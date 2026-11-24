//! The DXIL validator in `dxil.dll` (`IDxcValidator`), for tests and the
//! offline batch. Translated shaders do not need it: they carry the
//! container hash we compute (`container::hash`).
//!
//! The DLL is found at `VICTORY_DXIL` if set, else the newest Windows SDK's
//! `bin\<version>\x64\dxil.dll`, else wherever `LoadLibrary` finds it.

use std::ffi::c_void;
use std::path::PathBuf;

use windows_sys::Win32::Foundation::{FreeLibrary, HMODULE};
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};

type Hresult = i32;

#[repr(C)]
struct Guid(u32, u16, u16, [u8; 8]);

const CLSID_DXC_VALIDATOR: Guid = Guid(
    0x8ca3_e215,
    0xf728,
    0x4cf3,
    [0x8c, 0xdd, 0x88, 0xaf, 0x91, 0x75, 0x87, 0xa1],
);
const IID_IDXC_VALIDATOR: Guid = Guid(
    0xa6e8_2bd2,
    0x1fd7,
    0x4826,
    [0x98, 0x11, 0x28, 0x57, 0xe7, 0x97, 0xf4, 0x9a],
);
/// `DxcValidatorFlags_InPlaceEdit`: sign the container we pass.
const IN_PLACE_EDIT: u32 = 1;

type CreateInstance =
    unsafe extern "system" fn(*const Guid, *const Guid, *mut *mut c_void) -> Hresult;
type FreeModule = unsafe extern "system" fn(HMODULE) -> i32;

/// COM object: a pointer to a table of method pointers.
type Object = *mut *const usize;

unsafe fn method<F: Copy>(obj: Object, slot: usize) -> F {
    // SAFETY: the caller names a slot of the object's interface.
    unsafe { std::mem::transmute_copy(&*(*obj).add(slot)) }
}

unsafe fn release(obj: Object) {
    if !obj.is_null() {
        // SAFETY: slot 2 of every COM interface is Release.
        unsafe { method::<unsafe extern "system" fn(Object) -> u32>(obj, 2)(obj) };
    }
}

/// An `IDxcBlob` over our bytes: QueryInterface, AddRef, Release,
/// GetBufferPointer, GetBufferSize. Lives on the caller's stack.
#[repr(C)]
struct Blob {
    vtable: *const BlobVtable,
    data: *mut u8,
    len: usize,
}

/// IUnknown and IDxcBlob, as raw GUID bytes.
const IUNKNOWN: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 0xc0, 0, 0, 0, 0, 0, 0, 0x46];
const IDXC_BLOB: [u8; 16] = [
    0x08, 0xfb, 0xa5, 0x8b, 0x95, 0x51, 0xe2, 0x40, 0xac, 0x58, 0x0d, 0x98, 0x9c, 0x3a, 0x01, 0x02,
];

unsafe extern "system" fn blob_qi(
    this: *mut Blob,
    iid: *const [u8; 16],
    out: *mut *mut c_void,
) -> Hresult {
    // SAFETY: the validator passes valid pointers.
    unsafe {
        if *iid == IUNKNOWN || *iid == IDXC_BLOB {
            *out = this.cast();
            0
        } else {
            *out = std::ptr::null_mut();
            0x8000_4002_u32 as i32 // E_NOINTERFACE
        }
    }
}
unsafe extern "system" fn blob_ref(_: *mut Blob) -> u32 {
    1
}
unsafe extern "system" fn blob_ptr(this: *mut Blob) -> *mut u8 {
    // SAFETY: `this` is our blob.
    unsafe { (*this).data }
}
unsafe extern "system" fn blob_len(this: *mut Blob) -> usize {
    // SAFETY: `this` is our blob.
    unsafe { (*this).len }
}

#[repr(C)]
struct BlobVtable {
    qi: unsafe extern "system" fn(*mut Blob, *const [u8; 16], *mut *mut c_void) -> Hresult,
    add_ref: unsafe extern "system" fn(*mut Blob) -> u32,
    release: unsafe extern "system" fn(*mut Blob) -> u32,
    ptr: unsafe extern "system" fn(*mut Blob) -> *mut u8,
    len: unsafe extern "system" fn(*mut Blob) -> usize,
}

static BLOB_VTABLE: BlobVtable = BlobVtable {
    qi: blob_qi,
    add_ref: blob_ref,
    release: blob_ref,
    ptr: blob_ptr,
    len: blob_len,
};

struct LoadedModule {
    handle: HMODULE,
    free: FreeModule,
}

impl Drop for LoadedModule {
    fn drop(&mut self) {
        // SAFETY: construction transfers one reference to the module.
        unsafe { (self.free)(self.handle) };
    }
}

pub struct Validator {
    object: Object,
    // Field drops run after Validator::drop releases the COM interface.
    _module: Option<LoadedModule>,
}

// SAFETY: the validator object is free-threaded (DXC's validator has no
// thread affinity); calls are serialised by `&mut self`. The module reference
// belongs to the process and can be released from another thread.
unsafe impl Send for Validator {}

impl Validator {
    /// Where `dxil.dll` is looked for.
    pub fn default_path() -> Option<PathBuf> {
        if let Some(p) = std::env::var_os("VICTORY_DXIL") {
            return Some(p.into());
        }
        let kits = PathBuf::from(r"C:\Program Files (x86)\Windows Kits\10\bin");
        let mut versions: Vec<PathBuf> = std::fs::read_dir(&kits)
            .ok()?
            .filter_map(Result::ok)
            .map(|e| e.path().join(r"x64\dxil.dll"))
            .filter(|p| p.exists())
            .collect();
        versions.sort();
        versions.pop()
    }

    /// Loads `path`'s validator (or `dxil.dll` from the DLL search path).
    pub fn load(path: Option<&std::path::Path>) -> Result<Self, String> {
        let name: Vec<u16> = path
            .map_or_else(|| "dxil.dll".into(), |p| p.as_os_str().to_owned())
            .to_string_lossy()
            .encode_utf16()
            .chain([0])
            .collect();
        // SAFETY: plain Win32 calls with a NUL-terminated name.
        unsafe {
            let module = LoadLibraryW(name.as_ptr());
            if module.is_null() {
                return Err(format!("cannot load {}", String::from_utf16_lossy(&name)));
            }
            Self::from_module(module, FreeLibrary, |module| {
                let create = GetProcAddress(module, c"DxcCreateInstance".as_ptr().cast())
                    .ok_or("dxil.dll lacks DxcCreateInstance")?;
                let create: CreateInstance = std::mem::transmute(create);
                let mut object = std::ptr::null_mut();
                let hr = create(&CLSID_DXC_VALIDATOR, &IID_IDXC_VALIDATOR, &mut object);
                if hr < 0 || object.is_null() {
                    return Err(format!("DxcCreateInstance(validator) failed: {hr:#x}"));
                }
                Ok(object.cast())
            })
        }
    }

    // SAFETY: `module` is one owned reference accepted by `free`. On success,
    // `create` returns one owned reference to a live validator interface.
    unsafe fn from_module(
        module: HMODULE,
        free: FreeModule,
        create: impl FnOnce(HMODULE) -> Result<Object, String>,
    ) -> Result<Self, String> {
        let module = LoadedModule {
            handle: module,
            free,
        };
        Ok(Self {
            object: create(module.handle)?,
            _module: Some(module),
        })
    }

    /// Validates `container` and signs it in place; the error text if it
    /// is rejected.
    pub fn validate(&mut self, container: &mut [u8]) -> Result<(), String> {
        let mut blob = Blob {
            vtable: &BLOB_VTABLE,
            data: container.as_mut_ptr(),
            len: container.len(),
        };
        // SAFETY: IDxcValidator::Validate (slot 3) with our blob; then
        // IDxcOperationResult GetStatus (3) and GetErrorBuffer (5), and
        // IDxcBlob GetBufferPointer (3) / GetBufferSize (4).
        unsafe {
            let mut result: Object = std::ptr::null_mut();
            let validate: unsafe extern "system" fn(
                Object,
                *mut Blob,
                u32,
                *mut Object,
            ) -> Hresult = method(self.object, 3);
            let hr = validate(self.object, &mut blob, IN_PLACE_EDIT, &mut result);
            if hr < 0 || result.is_null() {
                return Err(format!("Validate failed: {hr:#x}"));
            }
            let mut status = 0;
            let get_status: unsafe extern "system" fn(Object, *mut Hresult) -> Hresult =
                method(result, 3);
            let hr = get_status(result, &mut status);
            if hr < 0 {
                release(result);
                return Err(format!("GetStatus failed: {hr:#x}"));
            }
            let mut text = String::new();
            if status < 0 {
                let mut errors: Object = std::ptr::null_mut();
                let get_errors: unsafe extern "system" fn(Object, *mut Object) -> Hresult =
                    method(result, 5);
                if get_errors(result, &mut errors) >= 0 && !errors.is_null() {
                    let ptr: unsafe extern "system" fn(Object) -> *const u8 = method(errors, 3);
                    let len: unsafe extern "system" fn(Object) -> usize = method(errors, 4);
                    let len = len(errors);
                    if len != 0 {
                        let ptr = ptr(errors);
                        if !ptr.is_null() {
                            let bytes = std::slice::from_raw_parts(ptr, len);
                            text = String::from_utf8_lossy(bytes)
                                .trim_end_matches('\0')
                                .trim()
                                .into();
                        }
                    }
                    release(errors);
                }
                if text.is_empty() {
                    text = format!("rejected ({status:#x})");
                }
            }
            release(result);
            if status < 0 { Err(text) } else { Ok(()) }
        }
    }
}

impl Drop for Validator {
    fn drop(&mut self) {
        // SAFETY: we own one reference.
        unsafe { release(self.object) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const E_FAIL: Hresult = 0x8000_4005_u32 as Hresult;
    const E_INVALIDARG: Hresult = 0x8007_0057_u32 as Hresult;

    #[derive(Default)]
    struct Scenario {
        validate_hr: Hresult,
        omit_result: bool,
        status_hr: Hresult,
        status: Hresult,
        errors_hr: Hresult,
        diagnostic: Option<&'static [u8]>,
    }

    #[repr(C)]
    struct MockHeader {
        vtable: *const usize,
        releases: u32,
    }

    #[repr(C)]
    struct Mock<T> {
        header: MockHeader,
        data: T,
    }

    impl<T> Mock<T> {
        fn new(vtable: &[usize], data: T) -> Self {
            Self {
                header: MockHeader {
                    vtable: vtable.as_ptr(),
                    releases: 0,
                },
                data,
            }
        }

        fn object(&mut self) -> Object {
            std::ptr::from_mut(self).cast()
        }
    }

    struct MockValidator {
        hr: Hresult,
        result: Object,
    }

    struct MockResult {
        status_hr: Hresult,
        status: Hresult,
        errors_hr: Hresult,
        errors: Object,
        error_queries: u32,
    }

    struct MockErrors {
        bytes: &'static [u8],
        pointer_queries: u32,
    }

    unsafe extern "system" fn mock_release(this: Object) -> u32 {
        // SAFETY: every mock object starts with MockHeader.
        unsafe { (*this.cast::<MockHeader>()).releases += 1 };
        0
    }

    unsafe extern "system" fn mock_validate(
        this: Object,
        blob: *mut Blob,
        flags: u32,
        out: *mut Object,
    ) -> Hresult {
        // SAFETY: validation invokes this slot on Mock<MockValidator> with a
        // live input blob and a writable result pointer.
        unsafe {
            let data = &(*this.cast::<Mock<MockValidator>>()).data;
            let vtable = &*(*blob).vtable;
            if flags != IN_PLACE_EDIT || (vtable.len)(blob) != 3 {
                return E_INVALIDARG;
            }
            *(vtable.ptr)(blob).add(1) = 0x5a;
            *out = data.result;
            data.hr
        }
    }

    unsafe extern "system" fn mock_status(this: Object, out: *mut Hresult) -> Hresult {
        // SAFETY: this slot belongs to Mock<MockResult> and out is writable.
        unsafe {
            let data = &(*this.cast::<Mock<MockResult>>()).data;
            if data.status_hr >= 0 {
                *out = data.status;
            }
            data.status_hr
        }
    }

    unsafe extern "system" fn mock_errors(this: Object, out: *mut Object) -> Hresult {
        // SAFETY: this slot belongs to Mock<MockResult> and out is writable.
        unsafe {
            let data = &mut (*this.cast::<Mock<MockResult>>()).data;
            data.error_queries += 1;
            if data.errors_hr >= 0 {
                *out = data.errors;
            }
            data.errors_hr
        }
    }

    unsafe extern "system" fn mock_error_ptr(this: Object) -> *const u8 {
        // SAFETY: this slot belongs to Mock<MockErrors>.
        unsafe {
            let data = &mut (*this.cast::<Mock<MockErrors>>()).data;
            data.pointer_queries += 1;
            if data.bytes.is_empty() {
                std::ptr::null()
            } else {
                data.bytes.as_ptr()
            }
        }
    }

    unsafe extern "system" fn mock_error_len(this: Object) -> usize {
        // SAFETY: this slot belongs to Mock<MockErrors>.
        let data = unsafe { &(*this.cast::<Mock<MockErrors>>()).data };
        data.bytes.len()
    }

    fn validate_mock(scenario: Scenario) -> (Result<(), String>, [u8; 3], [u32; 4]) {
        let release = mock_release as *const () as usize;
        let validate_vtable = [0, 0, release, mock_validate as *const () as usize];
        let result_vtable = [
            0,
            0,
            release,
            mock_status as *const () as usize,
            0,
            mock_errors as *const () as usize,
        ];
        let errors_vtable = [
            0,
            0,
            release,
            mock_error_ptr as *const () as usize,
            mock_error_len as *const () as usize,
        ];
        let mut errors = Mock::new(
            &errors_vtable,
            MockErrors {
                bytes: scenario.diagnostic.unwrap_or_default(),
                pointer_queries: 0,
            },
        );
        let mut result = Mock::new(
            &result_vtable,
            MockResult {
                status_hr: scenario.status_hr,
                status: scenario.status,
                errors_hr: scenario.errors_hr,
                errors: scenario
                    .diagnostic
                    .map_or(std::ptr::null_mut(), |_| errors.object()),
                error_queries: 0,
            },
        );
        let mut validator = Mock::new(
            &validate_vtable,
            MockValidator {
                hr: scenario.validate_hr,
                result: if scenario.omit_result || scenario.validate_hr < 0 {
                    std::ptr::null_mut()
                } else {
                    result.object()
                },
            },
        );
        let mut bytes = [0x11, 0x22, 0x33];
        let outcome = Validator {
            object: validator.object(),
            _module: None,
        }
        .validate(&mut bytes);
        assert_eq!(validator.header.releases, 1);
        let calls = [
            result.header.releases,
            result.data.error_queries,
            errors.header.releases,
            errors.data.pointer_queries,
        ];
        (outcome, bytes, calls)
    }

    #[test]
    fn failed_status_query_is_an_error_and_releases_the_result() {
        let (outcome, _, calls) = validate_mock(Scenario {
            status_hr: E_FAIL,
            ..Scenario::default()
        });
        assert_eq!(outcome, Err("GetStatus failed: 0x80004005".into()));
        assert_eq!(calls, [1, 0, 0, 0]);
    }

    #[test]
    fn empty_diagnostic_needs_no_buffer_pointer() {
        let (outcome, _, calls) = validate_mock(Scenario {
            status: E_INVALIDARG,
            diagnostic: Some(b""),
            ..Scenario::default()
        });
        assert_eq!(outcome, Err("rejected (0x80070057)".into()));
        assert_eq!(calls, [1, 1, 1, 0]);
    }

    #[test]
    fn validation_passes_the_live_bytes_and_releases_the_result() {
        let (outcome, bytes, calls) = validate_mock(Scenario {
            validate_hr: 1,
            status_hr: 1,
            status: 1,
            ..Scenario::default()
        });
        assert_eq!(outcome, Ok(()));
        assert_eq!(bytes, [0x11, 0x5a, 0x33]);
        assert_eq!(calls, [1, 0, 0, 0]);
    }

    #[test]
    fn failed_validation_or_missing_result_is_an_error() {
        for (validate_hr, expected) in [(E_FAIL, "0x80004005"), (0, "0x0")] {
            let (outcome, _, calls) = validate_mock(Scenario {
                validate_hr,
                omit_result: true,
                ..Scenario::default()
            });
            assert_eq!(outcome, Err(format!("Validate failed: {expected}")));
            assert_eq!(calls, [0; 4]);
        }
    }

    #[test]
    fn rejection_returns_trimmed_lossy_diagnostic_and_releases_both_objects() {
        let (outcome, _, calls) = validate_mock(Scenario {
            status: E_INVALIDARG,
            diagnostic: Some(b"  invalid \xff signature\r\n\0\0"),
            ..Scenario::default()
        });
        assert_eq!(outcome, Err("invalid \u{fffd} signature".into()));
        assert_eq!(calls, [1, 1, 1, 1]);
    }

    #[test]
    fn rejection_falls_back_to_status_without_a_diagnostic() {
        for (errors_hr, diagnostic, diagnostic_releases) in [
            (E_FAIL, Some(&b"unavailable"[..]), 0),
            (0, None, 0),
            (0, Some(&b" \r\n\0"[..]), 1),
        ] {
            let (outcome, _, calls) = validate_mock(Scenario {
                status: E_INVALIDARG,
                errors_hr,
                diagnostic,
                ..Scenario::default()
            });
            assert_eq!(outcome, Err("rejected (0x80070057)".into()));
            assert_eq!(calls, [1, 1, diagnostic_releases, diagnostic_releases]);
        }
    }

    #[test]
    fn blob_exposes_only_iunknown_and_idxcblob() {
        let mut bytes = [1, 2, 3];
        let mut blob = Blob {
            vtable: &BLOB_VTABLE,
            data: bytes.as_mut_ptr(),
            len: bytes.len(),
        };
        let this = std::ptr::from_mut(&mut blob);
        // SAFETY: the vtable belongs to the live stack blob and each query
        // uses an initialized IID and a writable output slot.
        unsafe {
            let mut out = std::ptr::null_mut();
            for iid in [IUNKNOWN, IDXC_BLOB] {
                assert_eq!((BLOB_VTABLE.qi)(this, &iid, &mut out), 0);
                assert_eq!(out, this.cast());
            }
            assert_eq!(
                (BLOB_VTABLE.qi)(this, &[0xff; 16], &mut out),
                0x8000_4002_u32 as Hresult
            );
            assert!(out.is_null());
            assert_eq!((BLOB_VTABLE.add_ref)(this), 1);
            assert_eq!((BLOB_VTABLE.release)(this), 1);
        }
    }

    #[derive(Default)]
    struct MockModule {
        object: Object,
        frees: u32,
        object_releases_at_free: u32,
    }

    unsafe extern "system" fn mock_free(module: HMODULE) -> i32 {
        // SAFETY: the test supplies a live MockModule and, when present, a
        // live object whose first field is a MockHeader.
        unsafe {
            let module = &mut *module.cast::<MockModule>();
            module.frees += 1;
            if !module.object.is_null() {
                module.object_releases_at_free = (*module.object.cast::<MockHeader>()).releases;
            }
        }
        1
    }

    #[test]
    fn failed_validator_creation_frees_the_module() {
        let mut module = MockModule::default();
        let handle = std::ptr::from_mut(&mut module).cast();
        // SAFETY: mock_free accepts this live mock module.
        let outcome =
            unsafe { Validator::from_module(handle, mock_free, |_| Err("creation failed".into())) };
        assert_eq!(outcome.err().as_deref(), Some("creation failed"));
        assert_eq!(module.frees, 1);
    }

    #[test]
    fn dropping_validator_releases_the_object_before_freeing_the_module() {
        let vtable = [0, 0, mock_release as *const () as usize];
        let mut object = Mock::new(&vtable, ());
        let object = object.object();
        let mut module = MockModule {
            object,
            ..MockModule::default()
        };
        let handle = std::ptr::from_mut(&mut module).cast();
        // SAFETY: the mock module, interface and vtable outlive the validator.
        let validator =
            unsafe { Validator::from_module(handle, mock_free, |_| Ok(object)) }.unwrap();
        assert_eq!(module.frees, 0);
        drop(validator);
        assert_eq!(module.frees, 1);
        assert_eq!(module.object_releases_at_free, 1);
    }
}
