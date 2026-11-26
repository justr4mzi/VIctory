# VIctory `v16` Fork Notes (`v16-rx6700xt` / `v16-beta`, AI-assisted)

This fork branch (`v16-rx6700xt`) was developed in an **AI-assisted** session targeting **Windows 11 + AMD Ryzen 5 5600 (6C/12T) + AMD Radeon RX 6700 XT (12 GB VRAM, RDNA2) + 48 GB RAM**.

## Base & Multi-Fork Integrations

- **Base (`39031cb4`):** `avatarmanz/VIctory-7900XTX` (async submission, `VICTORY_THIN`, `VICTORY_JOIN_PARTS`, `VICTORY_CARD_STEPS`, `VICTORY_UPLOAD_AHEAD`, `VICTORY_SHOW_AFTER`, `VICTORY_PORT_CONSTANTS`, `VICTORY_PORT_DESCRIPTORS`, AVX2 compares, sky-dome `ambient.rs`, parallel `land::each_view` heap pass, 16-sharded `RETIRED`).
- **Belobug (`4e1b16a8`, `81ebb7bd`):** magenta probe filter in `ambient.rs`, `82%`/`90%` VRAM eviction curve in `host.rs`, `room_lately()` guard on `keep_shown`, and lock-free compiler VEH + `warm-bad.txt` in `pipeline/make.rs`.
- **Upstream #2493 (`85b0b187`, `f3578e38`):** whole-wave `v_add_f16` inline float constant decoding (`Shape::Active` adaptation in `emit/wave.rs`).
- **Upstream #2449:** null CBV descriptor handling (`create_cbv` -> `View::Constant(0, 0)`) + unit test `null_cbv_clears_a_stale_view`.
- **ka1qux (`4bf4354d`, `f10ab5e1`):** hardware `aniso_point_mip` capability query wired into `sampler::console()` (+ unit test) and `tiles::let_go` / `give_back_let_go` in `HostTexture::drop` & `host::sweep`.
- **triippywhale (`8132aaac`, `0de380d8`):** system-commit-aware idle trim via `memory_status()` (`available_cache_ram`).

## Local Additions (`e6771f91`)

- **`Cargo.toml`:** `[profile.ship]` (`inherits = "release"`, `lto = "fat"`, `codegen-units = 1`).
- **`shadow.rs` & `descriptor/host.rs`:** cached `Shadow::uav_allowed()` (`OnceLock<bool>`) eliminating duplicate `GetDesc()` calls in `buffer_uav`; `EXACT_POOL` buffer reuse (`BUCKET_CAP = 512`, `TOTAL_CAP = 4096`) and adaptive prune cadence in `shadow::exact()`.
- **`list/thin.rs`:** calibrated default `VICTORY_THIN_THREADS` to `(cpus / 2).clamp(4, 8)` (`6` threads on 6C/12T Ryzen 5 5600).
- **`gameinput.rs`:** diagnostic file/focus input (`VICTORY_KEYS_FILE`, `VICTORY_KEYS_HOLD_MS`, `VICTORY_KEYS_ANYWHERE`) plus `WASD` / `IJKL` thumbstick emulation (+ unit test `keys_file_parses_consumes_holds_and_expires`).

## Verification & Honest Status

- **Policy check:** `cargo xtask policy` -> `asset policy: 251 files ok`.
- **Non-`victory-gpu` workspace tests:** **604 passed, 0 failed, 11 ignored**.
- **`victory-gpu` standard lib tests:** **340 passed, 1 failed, 34 ignored**.
- **`victory-gpu --features diagnostics` tests:** **388 lib + 1 replay = 389 passed, 1 failed, 35 ignored**.
  - The sole offline test failure is the untouched upstream test `descriptor::host::tests::views_are_made_beside_the_heap_and_copied_in` (`crates/victory-gpu/src/descriptor/host.rs:5209`): on AMD RDNA2 (`RX 6700 XT`), `CopyDescriptorsSimple` from a non-shader-visible staging heap to a shader-visible heap rewrites the first 16 driver-internal handle bytes (bytes `16..32` match and `GetDeviceRemovedReason` remains `Ok`).
- **Live GPU testing on RX 6700 XT:**
  - Reaches 3D scene rendering (`WORLD=2`, heap pass `3.3–73.9 ms` vs `399 ms` in `v15`), but triggers `DXGI_ERROR_DEVICE_RESET` (`0x887A0007`, empty DRED breadcrumbs `[]`, `0` submissions in flight) within `0–12 s` of entering the 3D world under all 4 tested flag presets (including RDNA2-safe overrides `VICTORY_VIEW_COPY=0`, `VICTORY_TEXTURE_POOL=0`, `VICTORY_HELD_LEVELS=0`, `VICTORY_SHOW_NEW_LATER=0`, `VICTORY_LIBRARY_MAP=0`, `VICTORY_UNDER_WIDE=0`, `VICTORY_LAND_ECHO=0`).
  - Tagged **`v16-beta`** honestly pending `VICTORY_D3D12_DEBUG=1` isolation; **`v15-ready`** remains the verified multi-hour playable release for RX 6700 XT.
