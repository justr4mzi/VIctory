//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of the whole memory at every
//! submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: ID3D12Resource,
    data: *mut u8,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    last: Mutex<Last>,
    #[cfg(test)]
    gpu_uploads: std::sync::atomic::AtomicU64,
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not watched), or the write-watch asking it was detiled after.
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    asked: u64,
}

/// Room to read a watched texture's memory into, shared by all textures (a
/// buffer each would hold the sum of their sizes for good). Kept, already
/// touched, so refreshing does not fault its pages in again.
static SCRATCH: Mutex<Vec<u8>> = Mutex::new(Vec::new());

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

#[cfg(windows)]
impl Drop for Staging {
    fn drop(&mut self) {
        crate::host::release_late(self.buffer.clone());
    }
}

static STAGING: Mutex<Option<HashMap<usize, Arc<Staging>>>> = Mutex::new(None);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HostWrite {
    /// Write-watched memory: the write-watch asking when the host copy happened.
    Watched { start: usize, end: usize, asked: u64 },
    /// Non-watched memory (physical-page view): content hash at the host copy.
    Hashed {
        memory: usize,
        size: usize,
        hash: Option<blake3::Hash>,
    },
    /// Layout unknown or unsupported: keep host contents until released.
    Fixed,
}

/// Textures the host GPU wrote by a copy or resolve: their contents stay on
/// the host until the title writes their memory again.
static HOST_WRITTEN: Mutex<Option<HashMap<usize, HostWrite>>> = Mutex::new(None);

fn host_write_state(memory: *const u8, size: usize) -> HostWrite {
    let start = memory as usize;
    if let Some((now, _)) = crate::watch::writes(start, start + size, false) {
        return HostWrite::Watched {
            start,
            end: start + size,
            asked: now,
        };
    }
    let hash = committed(memory, size).then(|| {
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(memory, size) };
        blake3::hash(tiled)
    });
    HostWrite::Hashed {
        memory: start,
        size,
        hash,
    }
}

/// Whether `resource`'s host copy is still newer than its title memory.
fn still_host_written(resource: usize) -> bool {
    let state = HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .and_then(|map| map.get(&resource).copied());
    let Some(state) = state else {
        return false;
    };
    let unchanged = match state {
        HostWrite::Fixed => true,
        HostWrite::Watched { start, end, asked } => crate::watch::writes(start, end, false)
            .is_none_or(|(_, written)| written.iter().all(|&(_, w)| w <= asked)),
        HostWrite::Hashed { memory, size, hash } => {
            if !committed(memory as *const u8, size) {
                true
            } else {
                // SAFETY: committed memory of the texture's layout size.
                let tiled = unsafe { std::slice::from_raw_parts(memory as *const u8, size) };
                hash == Some(blake3::hash(tiled))
            }
        }
    };
    if !unchanged {
        let mut map = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(map) = map.as_mut()
            && map.get(&resource) == Some(&state)
        {
            map.remove(&resource);
        }
    }
    unchanged
}

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    if crate::resource::host_resource(resource).is_some_and(|t| !title_written(t)) {
        return;
    }
    let state = crate::resource::texture_memory(resource)
        .and_then(|(memory, desc)| {
            let layout = crate::resource::texture_layout(&desc)?.planes[0]
                .layout
                .clone();
            (!layout.desc.swizzle.xor()).then(|| host_write_state(memory, layout.size as usize))
        })
        .unwrap_or(HostWrite::Fixed);
    let mut map = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    map.get_or_insert_with(HashMap::new)
        .insert(resource as usize, state);
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
    let mut host_written = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(host_written) = host_written.as_mut() {
        host_written.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Registers a title-written texture with `list` (once per list) so that at
/// submission time its staging buffer is checked for CPU modifications and,
/// only when its contents changed (including first upload), copied into its
/// host texture on the list's prelude command list.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    if !list.first_use(resource as usize) || still_host_written(resource as usize) {
        return;
    }
    let Some(copy) = crate::resource::host_copy(resource) else {
        return;
    };
    let Some(staging) = staging(resource, texture) else {
        return;
    };
    // Once per submission however many of its lists sample the texture;
    // `copy` keeps the host texture alive even if the title releases it
    // before submission, and `CopyTextureRegion` is only recorded onto the
    // prelude when `staging.refresh()` reports dirty contents.
    list.defer_texture(staging, copy);
}

impl Staging {
    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all). Returns `true` when the
    /// staging buffer received new contents that must be copied to the host
    /// texture (including initial upload).
    pub(crate) fn refresh(&self) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let (read, dirty) = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
        dirty
    }

    /// Returns `(bytes_read, dirty)`.
    fn refresh_now(&self) -> (usize, bool) {
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        if let Some((now, written)) = crate::watch::writes(start, start + size, false) {
            if last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked) {
                return (0, false);
            }
            last.asked = now;
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            let mut read = SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
            read.resize(size, 0);
            crate::host::copy_mapped(read.as_mut_ptr(), self.memory, size, false);
            self.detile(&read);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return (size, true);
        }
        if !committed(self.memory, size) {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory is not committed"
            );
            return (0, false);
        }
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let hash = blake3::hash(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) {
            return (size, false);
        }
        last.hash = Some(hash);
        self.detile(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        (size, true)
    }

    /// Records `CopyTextureRegion` for every subresource from this staging
    /// buffer into `texture` (which must already be in `COPY_DEST`).
    pub(crate) fn record_copies(&self, list: &ID3D12GraphicsCommandList, texture: &ID3D12Resource) {
        for (i, footprint) in self.footprints.iter().enumerate() {
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: i as u32,
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above.
                pResource: unsafe { std::mem::transmute_copy(&self.buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: *footprint,
                },
            };
            // SAFETY: an open list; both locations name live resources.
            unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
            #[cfg(test)]
            self.gpu_uploads
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    #[cfg(test)]
    fn gpu_uploads(&self) -> u64 {
        self.gpu_uploads.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8]) {
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = pitch * rows as usize * f.Depth as usize;
            // SAFETY: the subresource's part of the mapped staging buffer.
            let linear = unsafe {
                std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len)
            };
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
        }
    }
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let end = p + len;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            return false;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    true
}

/// The staging state of a title texture, made on first use.
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&(resource as usize)) {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    let s = create_staging(host, resource, texture, layout, memory)?;
    tracing::debug!(?desc, "texture contents queued for staging upload");
    map.insert(resource as usize, s.clone());
    Some(s)
}

fn create_staging(
    host: &crate::host::Host,
    resource: This,
    texture: &ID3D12Resource,
    layout: victory_formats::texture::Layout,
    memory: *const u8,
) -> Option<Arc<Staging>> {
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let (buffer, data) = host.upload_buffer(total)?;
    crate::host::name(&buffer, || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    Some(Arc::new(Staging {
        buffer,
        data,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        memory,
        last: Mutex::default(),
        #[cfg(test)]
        gpu_uploads: std::sync::atomic::AtomicU64::new(0),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::Host;
    use crate::resource::HostTexture;
    use std::sync::atomic::{AtomicU32, Ordering};
    use victory_formats::texture::{Dimension, Swizzle, Texture, TextureDesc};
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};

    const SHADER_RESOURCE: u32 = (D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0
        | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0) as u32;

    fn make_rgba_texture(host: &Host, width: u32, height: u32) -> (Arc<HostTexture>, Texture) {
        let layout_tex = Texture::new(&TextureDesc {
            dimension: Dimension::Tex2d,
            format: DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32,
            width,
            height,
            depth_or_array: 1,
            mips: 1,
            samples: 1,
            swizzle: Swizzle::LINEAR,
        })
        .expect("linear RGBA layout");
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(width),
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            ..Default::default()
        };
        let mut resource: Option<ID3D12Resource> = None;
        // SAFETY: plain default-heap 2D texture creation on our test device.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut resource,
                )
                .expect("create host texture");
        }
        let copy = Arc::new(HostTexture {
            texture: resource.unwrap(),
            state: AtomicU32::new(D3D12_RESOURCE_STATE_COMMON.0 as u32),
            twin: None,
        });
        (copy, layout_tex)
    }

    /// Reads a 2D RGBA8 texture back from `host` into a tightly packed row buffer.
    fn read_back_rgba(host: &Host, copy: &Arc<HostTexture>, width: u32, height: u32) -> Vec<u8> {
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut rows = 0u32;
        let mut total = 0u64;
        // SAFETY: querying footprint of a live texture.
        let desc = unsafe { copy.texture.GetDesc() };
        // SAFETY: single-subresource footprint query.
        unsafe {
            host.device.GetCopyableFootprints(
                &desc,
                0,
                1,
                0,
                Some(&mut footprint),
                Some(&mut rows),
                None,
                Some(&mut total),
            );
        }
        let readback_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let readback_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: total,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain readback buffer creation.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &readback_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &readback_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .expect("create readback buffer");
        }
        let readback = readback.unwrap();
        let list = host.new_list().expect("readback list");
        let before = copy.state.load(Ordering::Relaxed);
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed COM pointer for the copy call.
            pResource: unsafe { std::mem::transmute_copy(&readback) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed COM pointer for the copy call.
            pResource: unsafe { std::mem::transmute_copy(&copy.texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        // SAFETY: recording a valid readback copy on an open list.
        unsafe {
            if before != D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32 {
                list.list.ResourceBarrier(&[crate::host::transition(
                    &copy.texture,
                    D3D12_RESOURCE_STATES(before as i32),
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
            }
            list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            if before != D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32 {
                list.list.ResourceBarrier(&[crate::host::transition(
                    &copy.texture,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATES(before as i32),
                )]);
            }
        }
        list.close();
        host.execute(&[&list]);

        let mut ptr = std::ptr::null_mut();
        // SAFETY: GPU finished writing `readback`; map for CPU read.
        unsafe {
            readback.Map(0, None, Some(&mut ptr)).expect("map readback");
            let pitch = footprint.Footprint.RowPitch as usize;
            let row_bytes = width as usize * 4;
            let src = std::slice::from_raw_parts(ptr.cast::<u8>(), total as usize);
            let mut out = Vec::with_capacity(row_bytes * height as usize);
            for y in 0..height as usize {
                let start = footprint.Offset as usize + y * pitch;
                out.extend_from_slice(&src[start..start + row_bytes]);
            }
            readback.Unmap(0, None);
            out
        }
    }

    fn fill_linear_rgba(layout: &victory_formats::texture::Layout, mem: &mut [u8], pixel: [u8; 4]) {
        let mut linear = vec![0u8; layout.desc.width as usize * layout.desc.height as usize * 4];
        for chunk in linear.as_chunks_mut::<4>().0 {
            *chunk = pixel;
        }
        let pitch = layout.desc.width as usize * 4;
        layout
            .tile(mem, 0, 0, &linear, pitch, linear.len())
            .expect("tile pixels");
    }

    fn all_pixels_eq(bytes: &[u8], expected: [u8; 4]) -> bool {
        let (chunks, rem) = bytes.as_chunks::<4>();
        rem.is_empty() && !chunks.is_empty() && chunks.iter().all(|&px| px == expected)
    }

    /// Exercises the real D3D12 WARP prelude upload path end to end:
    /// (a) initial use uploads title memory to GPU,
    /// (b) subsequent frames with unchanged title memory record zero GPU copies,
    /// (c) multi-list submissions after a modification upload only once,
    /// (d) modifying title memory triggers a fresh GPU upload with new pixels.
    #[test]
    fn warp_skips_clean_texture_uploads_and_reuploads_on_change() {
        let Some(host) = Host::new_warp() else {
            return; // WARP software adapter unavailable here
        };
        let (width, height) = (4, 4);
        let (copy, tex) = make_rgba_texture(&host, width, height);
        let layout = tex.planes[0].layout.clone();
        let mut title_mem = vec![0u8; layout.size as usize];
        let v1 = [0x11, 0x22, 0x33, 0xff];
        fill_linear_rgba(&layout, &mut title_mem, v1);

        let staging = create_staging(
            &host,
            std::ptr::null(),
            &copy.texture,
            layout.clone(),
            title_mem.as_ptr(),
        )
        .expect("create staging");
        assert_eq!(staging.gpu_uploads(), 0);

        let make_sampling_list = || {
            let list = host.new_list().expect("new list");
            list.defer_texture(staging.clone(), copy.clone());
            list.expect_texture_state(&copy, SHADER_RESOURCE);
            list.close();
            list
        };

        // (a) Submission 1: first use must upload onto the prelude and land on GPU.
        let list1 = make_sampling_list();
        host.execute(&[&list1]);
        assert_eq!(staging.gpu_uploads(), 1, "initial use must record 1 copy");
        assert_eq!(copy.state.load(Ordering::Relaxed), SHADER_RESOURCE);
        assert!(all_pixels_eq(
            &read_back_rgba(&host, &copy, width, height),
            v1
        ));

        // (b) Submission 2: unchanged title memory across two lists records ZERO copies.
        let list2a = make_sampling_list();
        let list2b = make_sampling_list();
        host.execute(&[&list2a, &list2b]);
        assert_eq!(
            staging.gpu_uploads(),
            1,
            "unchanged texture must record zero CopyTextureRegion uploads"
        );
        // Re-submitting the same closed list with unchanged memory also records zero copies.
        host.execute(&[&list2a]);
        assert_eq!(staging.gpu_uploads(), 1);
        assert!(all_pixels_eq(
            &read_back_rgba(&host, &copy, width, height),
            v1
        ));

        // (c) Submission 3: CPU modifies title memory; two lists sample it -> exactly 1 upload.
        let v2 = [0xaa, 0xbb, 0xcc, 0xff];
        fill_linear_rgba(&layout, &mut title_mem, v2);
        let list3a = make_sampling_list();
        let list3b = make_sampling_list();
        host.execute(&[&list3a, &list3b]);
        assert_eq!(
            staging.gpu_uploads(),
            2,
            "modified texture across two lists must upload exactly once"
        );
        assert_eq!(copy.state.load(Ordering::Relaxed), SHADER_RESOURCE);
        assert!(all_pixels_eq(
            &read_back_rgba(&host, &copy, width, height),
            v2
        ));

        // (d) Submission 4: unchanged again -> zero additional uploads.
        let list4 = make_sampling_list();
        host.execute(&[&list4]);
        assert_eq!(staging.gpu_uploads(), 2);
    }

    /// Verifies the write-watched (`MEM_WRITE_WATCH`) dirty-tracking path on
    /// real D3D12 WARP: first use uploads, clean frames skip, and a CPU page
    /// write triggers a single fresh upload.
    #[test]
    fn warp_write_watched_texture_uploads_only_when_written() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;

        let Some(host) = Host::new_warp() else {
            return;
        };
        let (width, height) = (4, 4);
        let (copy, tex) = make_rgba_texture(&host, width, height);
        let layout = tex.planes[0].layout.clone();
        let alloc_size = (layout.size as usize).max(crate::watch::PAGE);
        // SAFETY: allocating a private write-watched page buffer; kept for the rest of the test run.
        let raw = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                alloc_size,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as *mut u8;
        assert!(!raw.is_null());

        let v1 = [0x33, 0x44, 0x55, 0xff];
        // SAFETY: `raw` is committed read-write memory of `alloc_size` bytes.
        unsafe {
            let mem = std::slice::from_raw_parts_mut(raw, layout.size as usize);
            fill_linear_rgba(&layout, mem, v1);
        }

        let staging = create_staging(&host, std::ptr::null(), &copy.texture, layout.clone(), raw)
            .expect("create staging");

        let run_once = || {
            let list = host.new_list().expect("new list");
            list.defer_texture(staging.clone(), copy.clone());
            list.expect_texture_state(&copy, SHADER_RESOURCE);
            list.close();
            host.execute(&[&list]);
        };

        // Initial submission uploads.
        run_once();
        assert_eq!(staging.gpu_uploads(), 1);
        assert!(all_pixels_eq(
            &read_back_rgba(&host, &copy, width, height),
            v1
        ));

        // Unchanged write-watched memory skips GPU upload.
        run_once();
        assert_eq!(staging.gpu_uploads(), 1);

        // CPU write dirties the watched page -> next submission uploads.
        let v2 = [0xde, 0xad, 0xbe, 0xff];
        // SAFETY: `raw` is still committed read-write memory.
        unsafe {
            let mem = std::slice::from_raw_parts_mut(raw, layout.size as usize);
            fill_linear_rgba(&layout, mem, v2);
        }
        run_once();
        assert_eq!(staging.gpu_uploads(), 2);
        assert!(all_pixels_eq(
            &read_back_rgba(&host, &copy, width, height),
            v2
        ));

        // Clean again -> zero upload.
        run_once();
        assert_eq!(staging.gpu_uploads(), 2);

        // Kept for the rest of the test run so watch::JOURNAL does not pollute a reused VA in a later test under --test-threads=1.
    }

    #[test]
    fn host_written_texture_never_queues_cpu_upload() {
        let Some(host) = Host::new_warp() else {
            return;
        };
        let (copy, _) = make_rgba_texture(&host, 4, 4);
        let fake_resource = 0x1234_5678usize as This;
        host_written(fake_resource);
        let list = host.new_list().expect("new list");
        upload(&list, fake_resource, &copy.texture);
        assert!(
            !list.has_deferred_work(),
            "host-written textures must not queue CPU-to-GPU uploads"
        );
        forget(fake_resource as usize);
    }

    #[test]
    fn host_written_clears_after_write_watched_memory_is_written() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let len = crate::watch::PAGE;
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let ptr = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as *mut u8;
        assert!(!ptr.is_null());
        let key = ptr as usize;
        let state = host_write_state(ptr, len);
        assert!(matches!(state, HostWrite::Watched { .. }));
        {
            let mut map = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
            map.get_or_insert_with(HashMap::new).insert(key, state);
        }
        assert!(still_host_written(key), "untouched memory stays host-written");
        // SAFETY: committed page of ours.
        unsafe { ptr.write(0x42) };
        assert!(
            !still_host_written(key),
            "writing title memory clears host-written state"
        );
        assert!(!still_host_written(key));
        forget(key);
        // SAFETY: our allocation.
        unsafe { VirtualFree(ptr.cast(), 0, MEM_RELEASE) };
    }

    #[test]
    fn host_written_clears_after_non_watched_memory_changes() {
        let mut buf = vec![0x11u8; 256];
        let key = buf.as_ptr() as usize;
        let state = host_write_state(buf.as_ptr(), buf.len());
        assert!(matches!(state, HostWrite::Hashed { hash: Some(_), .. }));
        {
            let mut map = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
            map.get_or_insert_with(HashMap::new).insert(key, state);
        }
        assert!(still_host_written(key), "unchanged bytes stay host-written");
        buf[0] = 0x22;
        assert!(
            !still_host_written(key),
            "changed title bytes clear host-written state"
        );
        assert!(!still_host_written(key));
        forget(key);
    }
}
