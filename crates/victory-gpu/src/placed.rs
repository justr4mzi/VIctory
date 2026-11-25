//! Host textures placed in heaps of 64 MiB, for the textures the pass over
//! the descriptor heap makes by the hundred: a committed resource is a
//! kernel allocation of its own (about half a millisecond each), a placed
//! one is not.
//!
//! The range a texture has belongs to the COM object itself: a small object
//! given to it as private data, which it lets go of when it is destroyed (a
//! descriptor slot, a list and the title's texture each may hold it). From
//! then on the range waits for the submissions running now
//! ([`crate::host::retire`]) before another texture gets it: the system does
//! that for a committed resource.
// shortcut: no aliasing barrier between two textures that have a range one
// after the other: what used the first has finished when the second gets it,
// and every texture made here is uploaded whole before a shader reads it.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, MutexGuard};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{GUID, HRESULT, IUnknown, IUnknown_Vtbl, Interface};

/// A heap's bytes.
const HEAP: u64 = 64 << 20;
/// The largest texture that goes into one.
const LARGEST: u64 = 32 << 20;
/// Where textures may start: any, and small ones (most detailed level of at
/// most 64 KiB). Each kind has its own heaps.
const NORMAL: u64 = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64;
const SMALL: u64 = D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT as u64;
/// An empty heap is freed after this, but for one of each kind.
const EMPTY_MS: u64 = 5_000;

/// The free ranges of a heap, by offset: none touch.
struct Free(Vec<(u64, u64)>);

impl Free {
    /// The start of `size` bytes in the smallest range that has them.
    fn take(&mut self, size: u64) -> Option<u64> {
        let fits = (0..self.0.len()).filter(|&i| self.0[i].1 >= size);
        let best = fits.min_by_key(|&i| self.0[i].1)?;
        let (offset, has) = self.0[best];
        if has == size {
            self.0.remove(best);
        } else {
            self.0[best] = (offset + size, has - size);
        }
        Some(offset)
    }

    fn give(&mut self, offset: u64, size: u64) {
        let at = self.0.partition_point(|&(o, _)| o < offset);
        self.0.insert(at, (offset, size));
        if self.0.get(at + 1).is_some_and(|&(next, _)| next == offset + size) {
            self.0[at].1 += self.0.remove(at + 1).1;
        }
        if at > 0 && self.0[at - 1].0 + self.0[at - 1].1 == offset {
            self.0[at - 1].1 += self.0.remove(at).1;
        }
    }
}

struct Block {
    id: u64,
    heap: ID3D12Heap,
    align: u64,
    free: Free,
    free_bytes: u64,
    /// When something was last given back.
    given: u64,
}

static BLOCKS: Mutex<Vec<Block>> = Mutex::new(Vec::new());
/// Bytes of the heaps no texture has.
static FREE: AtomicU64 = AtomicU64::new(0);
/// For [`stats`]: heaps made, textures placed, textures committed instead.
static COUNTS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

fn blocks() -> MutexGuard<'static, Vec<Block>> {
    BLOCKS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A texture's range, given back when dropped.
struct Range {
    block: u64,
    offset: u64,
    size: u64,
}

impl Drop for Range {
    fn drop(&mut self) {
        let mut blocks = blocks();
        if let Some(b) = blocks.iter_mut().find(|b| b.id == self.block) {
            b.free.give(self.offset, self.size);
            b.free_bytes += self.size;
            b.given = crate::host::now_ms();
            FREE.fetch_add(self.size, Relaxed);
        }
    }
}

/// The object a texture holds as private data: `IUnknown` and no more.
#[repr(C)]
struct Ticket {
    vtable: &'static IUnknown_Vtbl,
    references: AtomicU32,
    range: Range,
}

static TICKET: IUnknown_Vtbl = IUnknown_Vtbl {
    QueryInterface: ticket_query,
    AddRef: ticket_add,
    Release: ticket_release,
};

fn ticket(range: Range) -> IUnknown {
    let ticket = Box::new(Ticket {
        vtable: &TICKET,
        references: AtomicU32::new(1),
        range,
    });
    // SAFETY: an object whose first field points to the three methods, with
    // one reference, which the caller gets.
    unsafe { IUnknown::from_raw(Box::into_raw(ticket).cast()) }
}

unsafe extern "system" fn ticket_query(this: *mut c_void, iid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    const E_NOINTERFACE: HRESULT = HRESULT(0x8000_4002_u32 as i32);
    // SAFETY: the caller passes a live ticket, an identifier and an out
    // pointer.
    unsafe {
        if *iid != IUnknown::IID {
            *out = std::ptr::null_mut();
            return E_NOINTERFACE;
        }
        ticket_add(this);
        *out = this;
    }
    HRESULT(0)
}

unsafe extern "system" fn ticket_add(this: *mut c_void) -> u32 {
    // SAFETY: a live ticket.
    unsafe { &*this.cast::<Ticket>() }.references.fetch_add(1, Relaxed) + 1
}

unsafe extern "system" fn ticket_release(this: *mut c_void) -> u32 {
    use std::sync::atomic::Ordering::AcqRel;
    let this = this.cast::<Ticket>();
    // SAFETY: a live ticket.
    let left = unsafe { &*this }.references.fetch_sub(1, AcqRel) - 1;
    if left == 0 {
        // SAFETY: made by `ticket` from a box, and the last reference.
        let ticket = unsafe { Box::from_raw(this) };
        // Not before the submissions that run now have finished.
        crate::host::retire(ticket.range);
    }
    left
}

/// Whether such textures are placed (`VICTORY_PLACED_TEX=1`; off by default).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PLACED_TEX").is_some_and(|v| v == "1"))
}

/// A texture as `desc` says (no target, depth or unordered access; in
/// `state`) placed in a heap; `None` if it must be committed.
pub(crate) fn make(
    device: &ID3D12Device,
    desc: &D3D12_RESOURCE_DESC,
    state: D3D12_RESOURCE_STATES,
) -> Option<ID3D12Resource> {
    let made = try_make(device, desc, state);
    COUNTS[if made.is_some() { 1 } else { 2 }].fetch_add(1, Relaxed);
    made
}

fn try_make(
    device: &ID3D12Device,
    desc: &D3D12_RESOURCE_DESC,
    state: D3D12_RESOURCE_STATES,
) -> Option<ID3D12Resource> {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    if desc.Flags.0 & TARGET_DEPTH_UNORDERED != 0 || desc.SampleDesc.Count > 1 {
        return None;
    }
    let mut desc = *desc;
    desc.Alignment = SMALL;
    // SAFETY: a plain query.
    let mut info = unsafe { device.GetResourceAllocationInfo(0, &[desc]) };
    if info.Alignment != SMALL || info.SizeInBytes == u64::MAX {
        desc.Alignment = 0;
        // SAFETY: as above.
        info = unsafe { device.GetResourceAllocationInfo(0, &[desc]) };
    }
    let (size, align) = (info.SizeInBytes, info.Alignment);
    if size == 0 || size > LARGEST || (align != SMALL && align != NORMAL) {
        return None;
    }
    let (heap, range) = take(device, size.next_multiple_of(align), align)?;
    let mut texture: Option<ID3D12Resource> = None;
    // SAFETY: a live heap for such textures, and a range of it that is ours.
    let made = unsafe { device.CreatePlacedResource(&heap, range.offset, &desc, state, None, &mut texture) };
    if let Err(e) = made {
        tracing::warn!(error = %e, ?desc, "no placed host texture");
    }
    let texture = texture?;
    const RANGE: GUID = GUID::from_u128(0x6d2f1c1e_5a0b_4c7e_9a64_0f8d3b7e2a91);
    let ticket = ticket(range);
    // SAFETY: a live texture, which takes a reference of its own.
    if let Err(e) = unsafe { texture.SetPrivateDataInterface(&RANGE, &ticket) } {
        // The range must not go back while the texture lives.
        tracing::warn!(error = %e, "a placed host texture keeps its range for the run");
        std::mem::forget(ticket);
    }
    Some(texture)
}

/// `size` bytes of a heap of kind `align`; a new heap if none has them.
fn take(device: &ID3D12Device, size: u64, align: u64) -> Option<(ID3D12Heap, Range)> {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let mut blocks = blocks();
    for b in blocks.iter_mut().filter(|b| b.align == align && b.free_bytes >= size) {
        if let Some(offset) = b.free.take(size) {
            b.free_bytes -= size;
            FREE.fetch_sub(size, Relaxed);
            let range = Range { block: b.id, offset, size };
            return Some((b.heap.clone(), range));
        }
    }
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: HEAP,
        Properties: D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        },
        Alignment: NORMAL,
        Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
    };
    let mut heap: Option<ID3D12Heap> = None;
    // SAFETY: plain creation.
    if let Err(e) = unsafe { device.CreateHeap(&desc, &mut heap) } {
        tracing::warn!(error = %e, "no heap for host textures");
    }
    let heap = heap?;
    COUNTS[0].fetch_add(1, Relaxed);
    FREE.fetch_add(HEAP - size, Relaxed);
    let id = NEXT.fetch_add(1, Relaxed);
    blocks.push(Block {
        id,
        heap: heap.clone(),
        align,
        free: Free(vec![(size, HEAP - size)]),
        free_bytes: HEAP - size,
        given: 0,
    });
    let range = Range { block: id, offset: 0, size };
    Some((heap, range))
}

/// Frees heaps that have been empty for a while.
pub(crate) fn let_go_idle() {
    let now = crate::host::now_ms();
    let mut gone = Vec::new();
    let mut blocks = blocks();
    for align in [NORMAL, SMALL] {
        let mut first = true;
        let mut i = 0;
        while i < blocks.len() {
            let b = &blocks[i];
            let empty = b.align == align && b.free_bytes == HEAP;
            if empty && !first && now.saturating_sub(b.given) > EMPTY_MS {
                FREE.fetch_sub(HEAP, Relaxed);
                gone.push(blocks.remove(i));
                continue;
            }
            first &= !empty;
            i += 1;
        }
    }
    drop(blocks);
    // Freed outside the lock.
    drop(gone);
}

/// Bytes of the heaps no texture has.
pub(crate) fn free_bytes() -> u64 {
    FREE.load(Relaxed)
}

/// Since the start: heaps made, textures placed, and textures asked for
/// that had to be committed.
pub(crate) fn stats() -> [u64; 3] {
    [0, 1, 2].map(|i| COUNTS[i].load(Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_come_back_whole() {
        let mut free = Free(vec![(0, HEAP)]);
        let sizes = [NORMAL, 3 * NORMAL, 16 * NORMAL, NORMAL];
        let at: Vec<u64> = sizes.iter().map(|&s| free.take(s).unwrap()).collect();
        for (i, &a) in at.iter().enumerate() {
            for (j, &b) in at.iter().enumerate() {
                assert!(i == j || a + sizes[i] <= b || b + sizes[j] <= a);
            }
        }
        // The smallest range that has room is taken first.
        free.give(at[1], sizes[1]);
        assert_eq!(free.take(2 * NORMAL), Some(at[1]));
        free.give(at[1], 2 * NORMAL);
        for i in [2, 0, 3] {
            free.give(at[i], sizes[i]);
        }
        assert_eq!(free.0, [(0, HEAP)]);
        assert_eq!(free.take(HEAP + 1), None);
    }

    /// Textures alive at once have ranges of their own, and a range comes
    /// back with the last reference to its texture, not before.
    #[test]
    fn a_range_lives_as_long_as_its_texture() {
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_BC7_UNORM, DXGI_SAMPLE_DESC};
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let make = |width: u32, levels| {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: width.into(),
                Height: width,
                DepthOrArraySize: 1,
                MipLevels: levels,
                Format: DXGI_FORMAT_BC7_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            try_make(&host.device, &desc, D3D12_RESOURCE_STATE_COMMON)
        };
        let big = make(4096, 1).unwrap();
        let small = make(64, 7).unwrap();
        // SAFETY: plain queries on live textures.
        let (a, b) = unsafe { (big.GetDesc(), small.GetDesc()) };
        assert_eq!((a.Width, b.MipLevels), (4096, 7));
        let free = free_bytes();
        let slot = big.clone();
        drop(big);
        assert_eq!(free_bytes(), free, "a slot still holds it");
        drop(slot);
        // Other tests' submissions may be running: then later.
        for _ in 0..200 {
            if free_bytes() > free {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(free_bytes() >= free + (16 << 20));
        assert!(make(8192, 14).is_none(), "too large to place");
    }
}
