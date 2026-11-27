//! Copies the Series X scene just before the title blends its HUD. A
//! recording only becomes eligible for frame generation when submitted.

use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::{
    D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT,
    D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
    D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    D3D12_RESOURCE_STATE_RENDER_TARGET, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R8G8B8A8_UNORM,
};
use windows::core::Interface;

use crate::host::{HostList, transition};
use crate::object::This;

const MAX_COPIES: usize = 4;
const FFX_TEXTURE_2D: u32 = 2;

pub(crate) struct Color {
    texture: ID3D12Resource,
    size: [u32; 2],
}

impl Color {
    pub(super) fn resource(&self) -> super::Resource {
        let [width, height] = self.size;
        super::Resource {
            resource: self.texture.as_raw(),
            description: [
                FFX_TEXTURE_2D,
                super::RGBA8_UNORM,
                width,
                height,
                1,
                1,
                0,
                0,
            ],
            state: super::FFX_COMPUTE_READ,
        }
    }
}

struct Submitted {
    source: ID3D12Resource,
    color: Arc<Color>,
}

static COPIES: Mutex<Vec<Arc<Color>>> = Mutex::new(Vec::new());
static SUBMITTED: Mutex<Vec<Submitted>> = Mutex::new(Vec::new());

pub(crate) fn enabled() -> bool {
    static ASKED: OnceLock<bool> = OnceLock::new();
    *ASKED.get_or_init(|| crate::switch::on("VICTORY_FRAMEGEN_HUDLESS"))
        && crate::console::current() == crate::console::Console::SeriesX
        && super::enabled()
        && !super::ml_enabled()
}

pub(super) fn is_hud_blend(pipeline: This) -> bool {
    let Some(pipeline) = crate::pipeline::host_pipeline(pipeline) else {
        return false;
    };
    let [vertex, pixel] = pipeline.stages.as_slice() else {
        return false;
    };
    vertex.shader.hash == "0602c2384acbb3b2" && pixel.shader.hash == "a78d088fa55aa495"
}

/// Called immediately before the matched blend. Draw preparation has
/// already put the source in RENDER_TARGET. Restore that state for the draw.
pub(crate) fn record(list: &HostList, pipeline: This, target: This) {
    if !enabled() || !is_hud_blend(pipeline) {
        return;
    }
    let Some(source) = crate::resource::host_copy(target) else {
        return;
    };
    let source = &source.texture;
    // SAFETY: the title resource has a live host copy.
    let mut desc = unsafe { source.GetDesc() };
    if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || !matches!(
            desc.Format,
            DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R8G8B8A8_TYPELESS
        )
        || desc.DepthOrArraySize != 1
        || desc.MipLevels != 1
        || desc.SampleDesc.Count != 1
        || desc.Width == 0
        || desc.Width > 3840
        || desc.Height == 0
        || desc.Height > 2160
    {
        return;
    }
    let size = [desc.Width as u32, desc.Height];
    let mut copies = COPIES.lock().unwrap_or_else(|e| e.into_inner());
    let color = if let Some(color) = copies
        .iter()
        .find(|color| color.size == size && Arc::strong_count(color) == 1)
    {
        color.clone()
    } else {
        // Free copies for another resolution can make room for this one.
        copies.retain(|color| Arc::strong_count(color) != 1);
        if copies.len() == MAX_COPIES {
            return;
        }
        let Some(host) = crate::host::host() else {
            return;
        };
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let mut texture = None;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.Flags = D3D12_RESOURCE_FLAG_NONE;
        // SAFETY: a single-sample RGBA8 texture with a compatible copy format.
        let made = unsafe {
            host.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                None,
                &mut texture,
            )
        };
        if let Err(error) = made {
            static REPORTED: std::sync::Once = std::sync::Once::new();
            REPORTED.call_once(
                || tracing::warn!(%error, "FidelityFX FG3: could not allocate a HUD-less copy"),
            );
            return;
        }
        let Some(texture) = texture else {
            return;
        };
        let color = Arc::new(Color { texture, size });
        copies.push(color.clone());
        color
    };
    drop(copies);

    let rt = D3D12_RESOURCE_STATE_RENDER_TARGET;
    let src = D3D12_RESOURCE_STATE_COPY_SOURCE;
    let dest = D3D12_RESOURCE_STATE_COPY_DEST;
    let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    // SAFETY: the open list holds the source in RT state. The pool lends
    // this copy only while it is unreferenced by lists and interpolation.
    // Both transition pairs restore the tracked states.
    unsafe {
        list.list.ResourceBarrier(&[
            transition(source, rt, src),
            transition(&color.texture, read, dest),
        ]);
        list.list.CopyResource(&color.texture, source);
        list.list.ResourceBarrier(&[
            transition(source, src, rt),
            transition(&color.texture, dest, read),
        ]);
    }
    let source = source.clone();
    list.defer_host(move || {
        let mut submitted = SUBMITTED.lock().unwrap_or_else(|e| e.into_inner());
        if !submitted.iter().any(|entry| entry.source == source) {
            submitted.push(Submitted {
                source: source.clone(),
                color: color.clone(),
            });
        }
    });
}

/// At a present boundary, take only the copy of the resource being shown.
/// Clear all candidates so a skipped or reused source cannot get stale color.
pub(crate) fn take(source: Option<&ID3D12Resource>) -> Option<Arc<Color>> {
    if !enabled() {
        return None;
    }
    let mut submitted = SUBMITTED.lock().unwrap_or_else(|e| e.into_inner());
    let color = submitted
        .iter()
        .find(|entry| source == Some(&entry.source))
        .map(|entry| entry.color.clone());
    submitted.clear();
    color
}
