#[cfg(test)]
mod batch {   // Batch::texture_due lives here; `mod tests` is already taken at the top of this file
    use super::super::Batch;
    use std::sync::Arc;

    #[test]
    fn a_texture_is_checked_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
        // The first list sampling each texture checks it ...
        assert!(batch.texture_due(&a));
        assert!(batch.texture_due(&b));
        // ... later lists of the same submission do not.
        assert!(!batch.texture_due(&a));
        assert!(!batch.texture_due(&a.clone()));
        assert!(!batch.texture_due(&b));
    }

    #[test]
    fn title_writes_make_textures_due_again() {
        let mut batch = Batch::default();
        let a = Arc::new(1u32);
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A copy into title memory or work that may write it may have
        // changed the texture's memory.
        batch.title_written();
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A fresh submission starts over.
        let mut next = Batch::default();
        assert!(next.texture_due(&a));
    }

    /// The hashing a submission does for textures in physical-page memory,
    /// with every list checking the textures it samples (before) and with
    /// each checked once (after). Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_checks`.
    #[test]
    #[ignore = "benchmark"]
    fn texture_checks_per_submission() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 4 << 20;
        const LISTS: usize = 12;
        let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
            .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
            .collect();
        let submit = |dedupe: bool| {
            let mut batch = Batch::default();
            let mut hashed = 0usize;
            let started = std::time::Instant::now();
            for _ in 0..LISTS {
                for t in &textures {
                    if !dedupe || batch.texture_due(t) {
                        std::hint::black_box(blake3::hash(t));
                        hashed += 1;
                    }
                }
            }
            (started.elapsed(), hashed)
        };
        submit(false); // warm up
        let (before, n_before) = submit(false);
        let (after, n_after) = submit(true);
        assert_eq!((n_before, n_after), (TEXTURES * LISTS, TEXTURES));
        println!(
            "{LISTS} lists sampling {TEXTURES} textures of {} MiB: \
             before {n_before} hashes in {before:?}, after {n_after} in {after:?} ({:.1}x)",
            SIZE >> 20,
            before.as_secs_f64() / after.as_secs_f64()
        );
    }

    #[test]
    fn vram_idle_ms_scales_with_pressure() {
        use super::super::vram_idle_ms;
        let gib = 1024 * 1024 * 1024;
        // Normal (< 75%)
        assert_eq!(vram_idle_ms(Some([10 * gib, 0, 0, 0])), 5_000);
        assert_eq!(vram_idle_ms(Some([10 * gib, 7 * gib, 0, 0])), 5_000);
        assert_eq!(vram_idle_ms(Some([100, 74, 0, 0])), 5_000);
        // Moderate (75%..85%)
        assert_eq!(vram_idle_ms(Some([100, 75, 0, 0])), 2_500);
        assert_eq!(vram_idle_ms(Some([100, 84, 0, 0])), 2_500);
        // High (>= 85%, including over-budget)
        assert_eq!(vram_idle_ms(Some([100, 85, 0, 0])), 1_000);
        assert_eq!(vram_idle_ms(Some([10 * gib, 12 * gib, 0, 0])), 1_000);
        // Missing or zero budget falls back to normal
        assert_eq!(vram_idle_ms(None), 5_000);
        assert_eq!(vram_idle_ms(Some([0, 100, 0, 0])), 5_000);
    }

    #[test]
    fn ram_idle_ms_scales_with_pressure() {
        use super::super::ram_idle_ms;
        let gib = 1024 * 1024 * 1024;
        // Normal (>= 4 GiB)
        assert_eq!(ram_idle_ms(Some(8 * gib)), 10_000);
        assert_eq!(ram_idle_ms(Some(4 * gib)), 10_000);
        // Moderate (2 GiB..4 GiB)
        assert_eq!(ram_idle_ms(Some(4 * gib - 1)), 5_000);
        assert_eq!(ram_idle_ms(Some(2 * gib)), 5_000);
        // High (< 2 GiB)
        assert_eq!(ram_idle_ms(Some(2 * gib - 1)), 2_000);
        assert_eq!(ram_idle_ms(Some(0)), 2_000);
        // Missing falls back to normal
        assert_eq!(ram_idle_ms(None), 10_000);
    }

    #[test]
    fn tile_segments_split_only_at_subsequent_tile_lists() {
        use super::super::tile_segments;
        assert_eq!(tile_segments(&[], &[], 0), vec![]);
        // No tile steps: one segment covering all lists including any trailing check list.
        assert_eq!(tile_segments(&[false, false], &[0, 2], 5), vec![(None, 0..5)]);
        // Tile steps only on the first list: still a single ExecuteCommandLists call.
        assert_eq!(tile_segments(&[true, false], &[0, 1], 3), vec![(Some(0), 0..3)]);
        // Tile steps on the second list split before list 1.
        assert_eq!(
            tile_segments(&[false, true], &[0, 2], 4),
            vec![(None, 0..2), (Some(1), 2..4)]
        );
        // Tile steps on lists 0 and 2 out of 3: lists 0+1 stay batched, list 2 splits.
        assert_eq!(
            tile_segments(&[true, false, true], &[0, 1, 3], 5),
            vec![(Some(0), 0..3), (Some(2), 3..5)]
        );
        // Consecutive tile lists each get their own segment.
        assert_eq!(
            tile_segments(&[false, true, true], &[0, 1, 2], 4),
            vec![(None, 0..1), (Some(1), 1..2), (Some(2), 2..4)]
        );
    }

    /// Multi-list submission `[List 0, List 1]` where `List 0` reads an unmapped
    /// tiled texture and `List 1` maps a recycled dirty tile onto it and overwrites
    /// it. `List 1`'s `defer_tiles` `UpdateTileMappings` must run AFTER `List 0`
    /// executes so `List 0` reads zeros rather than stale previous-owner VRAM.
    #[test]
    fn deferred_tile_mappings_on_second_list_do_not_expose_stale_vram_to_first_list() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;

        let Some(host) = super::super::host() else {
            return;
        };
        let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        // SAFETY: querying device tiled-resources tier.
        let ok = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                (&raw mut opts).cast(),
                size_of_val(&opts) as u32,
            )
        };
        if ok.is_err() || opts.TiledResourcesTier.0 < 2 {
            return; // requires Tier 2 null-tile zero reads
        }

        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: 256,
            Height: 256,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0u64;
        // SAFETY: querying copy footprint for a valid 256x256 R8 texture.
        unsafe {
            host.device.GetCopyableFootprints(&desc, 0, 1, 0, Some(&mut fp), None, None, Some(&mut total));
        }

        // SAFETY: creating a 64 KiB tile heap and two reserved textures on a live device.
        let (heap, donor, target): (ID3D12Heap, ID3D12Resource, ID3D12Resource) = unsafe {
            let mut heap: Option<ID3D12Heap> = None;
            host.device
                .CreateHeap(
                    &D3D12_HEAP_DESC {
                        SizeInBytes: 65536,
                        Properties: D3D12_HEAP_PROPERTIES {
                            Type: D3D12_HEAP_TYPE_DEFAULT,
                            ..Default::default()
                        },
                        Alignment: 65536,
                        Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
                    },
                    &mut heap,
                )
                .unwrap();
            let mut donor: Option<ID3D12Resource> = None;
            host.device
                .CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut donor)
                .unwrap();
            let mut target: Option<ID3D12Resource> = None;
            host.device
                .CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COPY_SOURCE, None, &mut target)
                .unwrap();
            (heap.unwrap(), donor.unwrap(), target.unwrap())
        };

        let map_tile = |queue: &ID3D12CommandQueue, tex: &ID3D12Resource, heap: Option<&ID3D12Heap>| {
            let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
            let size = D3D12_TILE_REGION_SIZE {
                NumTiles: 1,
                ..Default::default()
            };
            let flag = if heap.is_some() {
                D3D12_TILE_RANGE_FLAG_NONE
            } else {
                D3D12_TILE_RANGE_FLAG_NULL
            };
            let offset = 0u32;
            let count = 1u32;
            // SAFETY: single-tile mapping on a live 1-tile reserved resource.
            unsafe {
                queue.UpdateTileMappings(
                    tex,
                    1,
                    Some(&coord),
                    Some(&size),
                    heap,
                    1,
                    Some(&flag),
                    Some(&offset),
                    Some(&count),
                    D3D12_TILE_MAPPING_FLAG_NONE,
                );
            }
        };

        let make_readback = || -> ID3D12Resource {
            let mut rb: Option<ID3D12Resource> = None;
            // SAFETY: creating a readback buffer on a live device.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: D3D12_HEAP_TYPE_READBACK,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        &D3D12_RESOURCE_DESC {
                            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                            Width: total,
                            Height: 1,
                            DepthOrArraySize: 1,
                            MipLevels: 1,
                            Format: DXGI_FORMAT_UNKNOWN,
                            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                            ..Default::default()
                        },
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut rb,
                    )
                    .unwrap();
            }
            rb.unwrap()
        };

        let copy_tex_to_buf = |list: &super::super::HostList, tex: &ID3D12Resource, buf: &ID3D12Resource| {
            let dst = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: borrowed handle held alive by test scope.
                pResource: unsafe { std::mem::transmute_copy(buf) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp },
            };
            let src = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: borrowed handle held alive by test scope.
                pResource: unsafe { std::mem::transmute_copy(tex) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            // SAFETY: open command list and compatible texture/buffer footprints.
            unsafe { list.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None) };
        };
        let copy_buf_to_tex = |list: &super::super::HostList, buf: &ID3D12Resource, tex: &ID3D12Resource| {
            let dst = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: borrowed handle held alive by test scope.
                pResource: unsafe { std::mem::transmute_copy(tex) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            let src = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: borrowed handle held alive by test scope.
                pResource: unsafe { std::mem::transmute_copy(buf) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp },
            };
            // SAFETY: open command list and compatible buffer/texture footprints.
            unsafe { list.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None) };
        };

        // Step 1: Poison heap tile 0 with 0xa5 via donor texture, then unmap donor.
        let (up_poison, ptr_poison) = host.upload_buffer(total).unwrap();
        // SAFETY: writing to mapped upload buffer.
        unsafe { std::ptr::write_bytes(ptr_poison, 0xa5, total as usize) };
        let poison_list = host.new_list().unwrap();
        copy_buf_to_tex(&poison_list, &up_poison, &donor);
        poison_list.close();
        map_tile(&host.queue, &donor, Some(&heap));
        host.execute(&[&poison_list]);
        map_tile(&host.queue, &donor, None);

        // Step 2: Record [list1, list2] in one multi-list submission.
        // list1 reads `target` (currently unmapped -> must see 0x00).
        let rb_before = make_readback();
        let list1 = host.new_list().unwrap();
        copy_tex_to_buf(&list1, &target, &rb_before);
        list1.close();

        // list2 maps the dirty 0xa5 heap tile onto `target` via defer_tiles,
        // overwrites `target` with 0x3c, and reads back `target`.
        let (up_fresh, ptr_fresh) = host.upload_buffer(total).unwrap();
        // SAFETY: writing to mapped upload buffer.
        unsafe { std::ptr::write_bytes(ptr_fresh, 0x3c, total as usize) };
        let rb_after = make_readback();
        let list2 = host.new_list().unwrap();
        {
            let target_clone = target.clone();
            let heap_clone = heap.clone();
            list2.defer_tiles(move || {
                if let Some(h) = super::super::host() {
                    let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
                    let size = D3D12_TILE_REGION_SIZE {
                        NumTiles: 1,
                        ..Default::default()
                    };
                    let flag = D3D12_TILE_RANGE_FLAG_NONE;
                    let offset = 0u32;
                    let count = 1u32;
                    // SAFETY: live reserved resource and heap on the host queue.
                    unsafe {
                        h.queue.UpdateTileMappings(
                            &target_clone,
                            1,
                            Some(&coord),
                            Some(&size),
                            Some(&heap_clone),
                            1,
                            Some(&flag),
                            Some(&offset),
                            Some(&count),
                            D3D12_TILE_MAPPING_FLAG_NONE,
                        );
                    }
                }
            });
        }
        let to_dest = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    // SAFETY: borrowed handle held alive by test scope.
                    pResource: unsafe { std::mem::transmute_copy(&target) },
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_COPY_SOURCE,
                    StateAfter: D3D12_RESOURCE_STATE_COPY_DEST,
                }),
            },
        };
        let to_src = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    // SAFETY: borrowed handle held alive by test scope.
                    pResource: unsafe { std::mem::transmute_copy(&target) },
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_COPY_DEST,
                    StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                }),
            },
        };
        // SAFETY: open command list; valid transitions on live resource.
        unsafe {
            list2.list.ResourceBarrier(&[to_dest]);
            copy_buf_to_tex(&list2, &up_fresh, &target);
            list2.list.ResourceBarrier(&[to_src]);
            copy_tex_to_buf(&list2, &target, &rb_after);
        }
        list2.close();

        host.execute(&[&list1, &list2]);

        let read_first = |rb: &ID3D12Resource| -> u8 {
            let mut ptr = std::ptr::null_mut();
            // SAFETY: completed readback buffer.
            unsafe {
                rb.Map(0, None, Some(&mut ptr)).unwrap();
                let b = *ptr.cast::<u8>();
                rb.Unmap(0, None);
                b
            }
        };
        assert_eq!(
            read_first(&rb_before),
            0x00,
            "list1 must execute before list2's UpdateTileMappings binds recycled 0xa5 VRAM"
        );
        assert_eq!(read_first(&rb_after), 0x3c);
        map_tile(&host.queue, &target, None);
    }

    #[test]
    fn should_record_brought_only_when_later_flights_exist() {
        use super::super::should_record_brought;
        // Single/last flight overflow (after == 5, landed == Some(5)): after - 1 = 4 <= 5.
        assert!(!should_record_brought(5, Some(5)));
        // Next submission numbered 6 has not queued lists yet (after == 6, landed == Some(5)): after - 1 = 5 <= 5.
        assert!(!should_record_brought(6, Some(5)));
        // True overlap with flight 6 in the air (after == 7, landed == Some(5)): after - 1 = 6 > 5.
        assert!(should_record_brought(7, Some(5)));
        // Degenerate `after <= 1` never records.
        assert!(!should_record_brought(0, None));
        assert!(!should_record_brought(1, None));
        assert!(should_record_brought(2, None));
    }

    #[test]
    fn committed_caches_expire_sooner_when_commit_is_low() {
        use super::super::committed_cache_idle_ms;
        let gib = 1 << 30;
        assert_eq!(committed_cache_idle_ms(Some((8 * gib, 400 << 20))), 2_000);
        assert_eq!(committed_cache_idle_ms(Some((8 * gib, 2 * gib))), 5_000);
        assert_eq!(committed_cache_idle_ms(Some((gib, 8 * gib))), 2_000);
        assert_eq!(committed_cache_idle_ms(Some((8 * gib, 4 * gib))), 10_000);
        assert_eq!(committed_cache_idle_ms(None), 10_000);
    }
}

/// The host's own paths for a copy in the card's memory, end to end, on
/// the software adapter only: the title's bytes go up, a shader reads them
/// and writes, and what it wrote comes back into title memory. Run it with
/// the switches under test in the environment (they are read once per
/// process); it does nothing unless the host device is the software
/// adapter (`VICTORY_D3D12_WARP=1` in a diagnostics build) and the copy is
/// in the card's memory (`VICTORY_CARD_BUFFERS_MIN_MB=1`):
///
/// `VICTORY_D3D12_WARP=1 VICTORY_CARD_BUFFERS_MIN_MB=1 [VICTORY_QUEUE2=n]
///  [VICTORY_RELAY=2 | VICTORY_LATE_WAIT=3] cargo test -p victory-gpu
///  --features diagnostics queue2_tests`
///
/// The same title memory must come out whatever the switches: through the
/// direct queue alone, and through the copy queues.
#[cfg(test)]
mod queue2_tests {
    use super::super::*;

    /// `memory[to] = memory[from] + add`, after `spin` steps.
    const ADD: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0)"
cbuffer What : register(b0) { uint to; uint from; uint add; uint spin; };
RWByteAddressBuffer memory : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint h = 0u;
    [loop]
    for (uint i = 0u; i < spin; i++) {
        h = (h ^ memory.Load(from)) * 0x01000193u;
    }
    memory.Store(to, memory.Load(from) + add + (h & 0u));
}
"#;

    #[test]
    fn warp_title_bytes_go_up_and_shader_writes_come_back_whatever_the_queues() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;
        let Some(host) = host() else {
            return;
        };
        // Never on a real card.
        // SAFETY: a plain query.
        let software = host
            .adapter
            .as_ref()
            .and_then(|a| unsafe { a.GetDesc1() }.ok())
            .is_some_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0);
        if !software {
            eprintln!("queue2_tests: skipped, the host device is not the software adapter (VICTORY_D3D12_WARP=1, diagnostics)");
            return;
        }
        const PIECE: usize = 64 << 10;
        const LEN: usize = 4 << 20;
        const ROUNDS: usize = 12;
        // "Title memory": write-watched, as the title's is.
        // SAFETY: a plain allocation of our own.
        let title = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as *mut u8;
        assert!(!title.is_null());
        let dword = |at: usize| unsafe { title.add(at).cast::<u32>() };
        for i in 0..LEN / 4 {
            // SAFETY: inside the allocation.
            unsafe { dword(i * 4).write_volatile(0x5A00_0000 | i as u32) };
        }
        crate::shadow::register(0x51_5151, title as u64, LEN as u64);
        let shadow = crate::shadow::find(title as u64, LEN as u64).expect("a host copy");
        if !shadow.carded() {
            eprintln!("queue2_tests: skipped, the copy is not in the card's memory (VICTORY_CARD_BUFFERS_MIN_MB=1)");
            return;
        }
        let code = compile(ADD, windows::core::s!("main"), windows::core::s!("cs_5_1")).expect("the test's shader");
        // SAFETY: the shader's own root signature; plain creation.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).expect("a root signature");
            let pso = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .expect("a pipeline");
            (root, pso)
        };
        // Round k: the title writes a dword in piece 1+k; a submission's
        // shader adds k to it and stores the sum in piece 30+k, a range
        // declared as shader-written. Nothing is waited for between the
        // rounds where submissions are sent ahead.
        let from = |k: usize| (1 + k) * PIECE + 4 * k + 64;
        let to = |k: usize| (30 + k) * PIECE + 8 * k + 128;
        let titles = |k: usize| 0x0100_0000 + 0x1000 * k as u32 + 7;
        let mut lists = Vec::new();
        for k in 0..ROUNDS {
            // SAFETY: inside the allocation.
            unsafe { dword(from(k)).write_volatile(titles(k)) };
            let list = host.new_list().expect("a host list");
            shadow.refresh(&list, title as u64, LEN as u64);
            shadow.gpu_writes(&list, (title as usize + (30 + k) * PIECE) as u64, PIECE as u64);
            // SAFETY: an open list; the copy's buffer, with unordered
            // access; the shader's own root signature.
            unsafe {
                list.list.SetComputeRootSignature(&root);
                list.list.SetPipelineState(&pso);
                let constants = [to(k) as u32, from(k) as u32, k as u32, 30_000];
                list.list.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                list.list.SetComputeRootUnorderedAccessView(1, shadow.buffer.GetGPUVirtualAddress());
                list.list.Dispatch(1, 1, 1);
            }
            list.close();
            if late() {
                host.execute_own(&list);
            } else {
                host.execute(&[&list]);
            }
            lists.push(list);
        }
        settle_now(WHY_SYNC);
        for k in 0..ROUNDS {
            // SAFETY: inside the allocation.
            let (sum, kept) = unsafe { (dword(to(k)).read_volatile(), dword(from(k)).read_volatile()) };
            assert_eq!(kept, titles(k), "round {k}: the title's own dword changed");
            assert_eq!(
                sum,
                titles(k) + k as u32,
                "round {k}: the shader read {:#x} where the title had written {:#x}, or its write did not come back",
                sum.wrapping_sub(k as u32),
                titles(k)
            );
        }
        // Nothing else moved.
        let touched: std::collections::HashSet<usize> = (0..ROUNDS).flat_map(|k| [from(k), to(k)]).collect();
        for i in (0..LEN / 4).filter(|i| !touched.contains(&(i * 4))) {
            // SAFETY: inside the allocation.
            let got = unsafe { dword(i * 4).read_volatile() };
            assert_eq!(got, 0x5A00_0000 | i as u32, "dword {i} of title memory changed");
        }
        // The queues the switches name did the work.
        if let Some(second) = crate::queue2::second() {
            let (down, up) = (second.down.stats.jobs.load(Ordering::Relaxed), second.up.stats.jobs.load(Ordering::Relaxed));
            // (Sends that found no landing buffer went the old way.)
            let no_room = second.landing[1].load(Ordering::Relaxed);
            eprintln!(
                "queue2_tests: VICTORY_QUEUE2={} late {} relay {}: {down} fetches and {up} sends through the copy queues ({no_room} more found no landing buffer), the direct queue told to wait {} times",
                crate::queue2::bits(),
                late(),
                relay(),
                second.held[0].load(Ordering::Relaxed)
            );
            assert_eq!(down > 0, second.fetches(), "fetches through the copy queue");
            assert_eq!(up + no_room > 0, second.uploads(), "sends through the copy queue");
            assert_eq!(up, second.held[0].load(Ordering::Relaxed), "the direct queue waits for every send");
        } else {
            eprintln!("queue2_tests: the direct queue alone (late {}, relay {})", late(), relay());
        }
        drop(lists);
    }

    /// Our send-up list with several copies in it, in a call of its own and
    /// first in its submission's call ([`feed_one_call`]): three copies in the
    /// card's memory, each with title bytes to send up and shader writes
    /// declared in every round, so that one send-up list takes ranges of all
    /// three and hashes each. The title bytes reach the shaders and the writes
    /// come back, in every copy; the debug layer (on the software adapter) finds
    /// nothing wrong with the barriers.
    #[test]
    fn warp_three_copies_go_up_in_one_list_and_their_writes_come_back() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;
        let Some(host) = host() else {
            return;
        };
        // Never on a real card.
        // SAFETY: a plain query.
        let software = host
            .adapter
            .as_ref()
            .and_then(|a| unsafe { a.GetDesc1() }.ok())
            .is_some_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0);
        if !software {
            eprintln!("queue2_tests: skipped, the host device is not the software adapter (VICTORY_D3D12_WARP=1, diagnostics)");
            return;
        }
        const PIECE: usize = 64 << 10;
        const LEN: usize = 2 << 20;
        const ROUNDS: usize = 6;
        const COPIES: usize = 3;
        let mut outcomes: Vec<Vec<Vec<u8>>> = Vec::new();
        for one_call in [false, true] {
            ONE_CALL_HERE.set(if one_call { 2 } else { 1 });
            let joined = SENDUP_JOINED.get();
            let titles: Vec<*mut u8> = (0..COPIES)
                .map(|c| {
                    // SAFETY: a plain allocation of our own.
                    let title = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) }
                        as *mut u8;
                    assert!(!title.is_null());
                    for i in 0..LEN / 4 {
                        // SAFETY: inside the allocation.
                        unsafe { title.add(i * 4).cast::<u32>().write_volatile(0x3C00_0000 | (c as u32) << 20 | i as u32) };
                    }
                    crate::shadow::register(0x52_5250 + c + 8 * usize::from(one_call), title as u64, LEN as u64);
                    title
                })
                .collect();
            let dword = |c: usize, at: usize| unsafe { titles[c].add(at).cast::<u32>() };
            let shadows: Vec<_> = titles
                .iter()
                .map(|&t| crate::shadow::find(t as u64, LEN as u64).expect("a host copy"))
                .collect();
            if !shadows.iter().all(|s| s.carded()) {
                eprintln!("queue2_tests: skipped, the copies are not in the card's memory (VICTORY_CARD_BUFFERS_MIN_MB=1)");
                return;
            }
            let code = compile(ADD, windows::core::s!("main"), windows::core::s!("cs_5_1")).expect("the test's shader");
            // SAFETY: the shader's own root signature; plain creation.
            let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
                let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).expect("a root signature");
                let pso = host
                    .device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: code.as_ptr().cast(),
                            BytecodeLength: code.len(),
                        },
                        ..Default::default()
                    })
                    .expect("a pipeline");
                (root, pso)
            };
            let from = |k: usize| (1 + k) * PIECE + 4 * k + 64;
            let to = |k: usize| (20 + k) * PIECE + 8 * k + 128;
            let written = |c: usize, k: usize| 0x0200_0000 + 0x1000 * k as u32 + 0x10 * c as u32 + 3;
            let mut lists = Vec::new();
            for k in 0..ROUNDS {
                let list = host.new_list().expect("a host list");
                for (c, shadow) in shadows.iter().enumerate() {
                    // SAFETY: inside the allocation.
                    unsafe { dword(c, from(k)).write_volatile(written(c, k)) };
                    shadow.refresh(&list, titles[c] as u64, LEN as u64);
                    shadow.gpu_writes(&list, (titles[c] as usize + (20 + k) * PIECE) as u64, PIECE as u64);
                    // SAFETY: an open list; the copy's buffer, with unordered access.
                    unsafe {
                        list.list.SetComputeRootSignature(&root);
                        list.list.SetPipelineState(&pso);
                        let constants = [to(k) as u32, from(k) as u32, (k + c) as u32, 3_000];
                        list.list.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                        list.list.SetComputeRootUnorderedAccessView(1, shadow.buffer.GetGPUVirtualAddress());
                        list.list.Dispatch(1, 1, 1);
                    }
                }
                list.close();
                assert_eq!(host.log_messages(), None, "the debug layer found an error");
                if late() {
                    host.execute_own(&list);
                } else {
                    host.execute(&[&list]);
                }
                lists.push(list);
            }
            settle_now(WHY_SYNC);
            assert_eq!(host.log_messages(), None, "the debug layer found an error");
            for c in 0..COPIES {
                for k in 0..ROUNDS {
                    // SAFETY: inside the allocation.
                    let (sum, kept) = unsafe { (dword(c, to(k)).read_volatile(), dword(c, from(k)).read_volatile()) };
                    assert_eq!(kept, written(c, k), "copy {c}, round {k}: the title's own dword changed");
                    assert_eq!(sum, written(c, k) + (k + c) as u32, "copy {c}, round {k}: the shader's write");
                }
            }
            drop(lists);
            assert_eq!(
                SENDUP_JOINED.get() - joined,
                if one_call { ROUNDS as u64 } else { 0 },
                "send-up lists in their submission's own call"
            );
            // SAFETY: the allocations, nothing writes them now.
            outcomes.push(titles.iter().map(|&t| unsafe { std::slice::from_raw_parts(t, LEN) }.to_vec()).collect());
            for (c, &t) in titles.iter().enumerate() {
                crate::shadow::unregister(0x52_5250 + c + 8 * usize::from(one_call), t as u64);
                // SAFETY: `t` was allocated by `VirtualAlloc` above for this test iteration.
                assert!(unsafe { crate::watch::release_title_allocation(t.cast()) });
            }
        }
        ONE_CALL_HERE.set(0);
        assert!(outcomes[0] == outcomes[1], "the same bytes with the send-up list in a call of its own and in the submission's");
    }

    /// `memory[to] = first` at once; then `spin` steps, each a store to
    /// `memory[to + 4]`; then `memory[to + 4] = 0` and `memory[to] = last`.
    /// Whatever `spin` is, the bytes it leaves are the same.
    const TWICE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0)"
cbuffer What : register(b0) { uint to; uint first; uint last; uint spin; };
RWByteAddressBuffer memory : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    memory.Store(to, first);
    [loop]
    for (uint i = 0u; i < spin; i++) {
        memory.Store(to + 4u, i);
    }
    memory.Store(to + 4u, 0u);
    memory.Store(to, last);
}
"#;

    /// The hole a fetch on the copy queue had (`VICTORY_QUEUE2` bit 1 with
    /// the relay): flight A writes a piece; flight B, sent behind it,
    /// writes the same piece through a half-way value and leaves it as A
    /// left it. A is landed while B is half way: the fetch, beside B on
    /// the copy queue, brings B's half-way bytes into title memory. B's
    /// compare finds the piece as A's compare left it and flags nothing,
    /// so B's landing fetched nothing and title memory kept the half-way
    /// bytes. With the fix B's landing fetches the piece all the same
    /// (`BESIDE`).
    ///
    /// `VICTORY_D3D12_WARP=1 VICTORY_CARD_BUFFERS_MIN_MB=1 VICTORY_QUEUE2=1
    ///  VICTORY_RELAY=1 cargo test -p victory-gpu --features diagnostics
    ///  fetched_again -- --nocapture` (alone: it lands its two flights
    /// itself; `VICTORY_QUEUE2_REFETCH=0` shows the fault).
    #[test]
    fn warp_a_piece_fetched_while_a_later_flight_writes_it_is_fetched_again() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;
        let Some(host) = host() else {
            return;
        };
        // Never on a real card.
        // SAFETY: a plain query.
        let software = host
            .adapter
            .as_ref()
            .and_then(|a| unsafe { a.GetDesc1() }.ok())
            .is_some_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0);
        if !software {
            eprintln!("queue2_tests: skipped, the host device is not the software adapter (VICTORY_D3D12_WARP=1, diagnostics)");
            return;
        }
        if !relay() || !crate::queue2::second().is_some_and(|s| s.fetches()) {
            eprintln!("queue2_tests: skipped, fetches do not go to a copy queue beside the relay (VICTORY_RELAY=1 VICTORY_QUEUE2=1)");
            return;
        }
        const PIECE: usize = 64 << 10;
        const LEN: usize = 4 << 20;
        // SAFETY: a plain allocation of our own.
        let title = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as *mut u8;
        assert!(!title.is_null());
        let dword = |at: usize| unsafe { title.add(at).cast::<u32>() };
        for i in 0..LEN / 4 {
            // SAFETY: inside the allocation.
            unsafe { dword(i * 4).write_volatile(0x5A00_0000 | i as u32) };
        }
        crate::shadow::register(0x52_5252, title as u64, LEN as u64);
        let shadow = crate::shadow::find(title as u64, LEN as u64).expect("a host copy");
        if !shadow.carded() {
            eprintln!("queue2_tests: skipped, the copy is not in the card's memory (VICTORY_CARD_BUFFERS_MIN_MB=1)");
            return;
        }
        let code = compile(TWICE, windows::core::s!("main"), windows::core::s!("cs_5_1")).expect("the test's shader");
        // SAFETY: the shader's own root signature; plain creation.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).expect("a root signature");
            let pso = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .expect("a pipeline");
            (root, pso)
        };
        // A list whose shader writes the dword at 256 of `piece`, declared.
        let record = |piece: usize, first: u32, last: u32, spin: u32| {
            let list = host.new_list().expect("a host list");
            shadow.refresh(&list, title as u64, LEN as u64);
            shadow.gpu_writes(&list, (title as usize + piece * PIECE) as u64, PIECE as u64);
            // SAFETY: an open list; the copy's buffer, with unordered
            // access; the shader's own root signature.
            unsafe {
                list.list.SetComputeRootSignature(&root);
                list.list.SetPipelineState(&pso);
                let constants = [(piece * PIECE + 256) as u32, first, last, spin];
                list.list.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                list.list.SetComputeRootUnorderedAccessView(1, shadow.buffer.GetGPUVirtualAddress());
                list.list.Dispatch(1, 1, 1);
            }
            list.close();
            list
        };
        // SAFETY: a plain query.
        let completed = || unsafe { host.fence.GetCompletedValue() };
        let wait_for = |value: u64| {
            while completed() < value {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        };
        // How many steps keep the card busy for some seconds: timed on
        // another piece, waited for.
        const TRIAL: u32 = 1 << 20;
        let trial = record(40, 1, 2, TRIAL);
        let timing = std::time::Instant::now();
        host.execute(&[&trial]);
        let took = timing.elapsed().as_secs_f64().max(1e-4);
        let spin = (f64::from(TRIAL) * 3.0 / took).clamp(f64::from(TRIAL), 2.0e9) as u32;
        // SAFETY: inside the allocation.
        assert_eq!(unsafe { dword(40 * PIECE + 256).read_volatile() }, 2, "the trial's write did not come back");
        const AT: usize = 20 * PIECE + 256;
        const HALF_WAY: u32 = 0xDEAD_0001;
        const LEFT: u32 = 0x600D_0002;
        // Flight A, and the card has run it.
        let a = record(20, HALF_WAY, LEFT, 16);
        let (submitted, mut sent_a) = host.submit(&[&a], true).expect("flight A sent ahead");
        let fence_a = *submitted;
        drop(submitted);
        wait_for(fence_a);
        // Flight B: the same piece, through the half-way value, for seconds.
        let b = record(20, HALF_WAY, LEFT, spin);
        let (submitted, mut sent_b) = host.submit(&[&b], true).expect("flight B sent ahead");
        let fence_b = *submitted;
        drop(submitted);
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert!(
            completed() < fence_b,
            "flight B had run after 0.3 s ({spin} steps; the trial's {TRIAL} took {took:.3} s): nothing to show"
        );
        // A lands, as `Host::relay_land` lands it, while B is half way.
        let mut fetch = None;
        {
            let plan = sent_a.card.as_ref().expect("flight A has a plan for the card's copies");
            host.relay_card_back(&[(sent_a.number, plan)], &mut fetch, fence_a, sent_a.number);
        }
        for (copy, ranges) in &sent_a.back {
            copy.copy_back_done(ranges);
        }
        // SAFETY: inside the allocation.
        let seen = unsafe { dword(AT).read_volatile() };
        let b_ran = completed() >= fence_b;
        eprintln!(
            "queue2_tests: after A's landing title memory holds {seen:#x} (half way {HALF_WAY:#x}, left {LEFT:#x}); B had run: {b_ran}; {spin} steps"
        );
        assert_eq!(
            seen, HALF_WAY,
            "the fetch for A did not come down beside B (it waited for B, or B had not begun): this machine cannot show the fault"
        );
        // B has run; it left the piece as A left it, so its compare flags
        // nothing. Its landing must bring the piece down all the same.
        wait_for(fence_b);
        {
            let plan = sent_b.card.as_ref().expect("flight B has a plan for the card's copies");
            host.relay_card_back(&[(sent_b.number, plan)], &mut fetch, fence_b, sent_b.number);
        }
        for (copy, ranges) in &sent_b.back {
            copy.copy_back_done(ranges);
        }
        // SAFETY: inside the allocation.
        let (left, beside) = unsafe { (dword(AT).read_volatile(), dword(AT + 4).read_volatile()) };
        assert_eq!(
            left, LEFT,
            "title memory kept flight B's half-way bytes: B's compare found the piece as A left it and B's landing fetched nothing"
        );
        assert_eq!(beside, 0, "the dword beside it holds a half-way count");
        for sent in [&mut sent_a, &mut sent_b] {
            if let Some(plan) = sent.card.take() {
                host.card_done(plan);
            }
        }
        drop((a, b, trial));
    }
}

#[cfg(test)]
mod wide_refresh_tests {
    use super::super::wide_refresh_enabled;

    #[test]
    fn parallel_refresh_is_default_on_multicore_and_can_be_overridden() {
        assert!(wide_refresh_enabled(None, Some(4)));
        assert!(wide_refresh_enabled(None, Some(16)));
        assert!(!wide_refresh_enabled(None, Some(2)));
        assert!(!wide_refresh_enabled(None, None));
        assert!(wide_refresh_enabled(Some("1"), Some(1)));
        assert!(!wide_refresh_enabled(Some("0"), Some(16)));
        assert!(wide_refresh_enabled(Some("auto"), Some(4)));
    }
}
