# VIctory `v16-ready` — AMD Radeon RX 6700 XT (RDNA2) Fork

> **AI-assisted development note:** All code integrations, conflict resolutions, bug fixes, benchmarks, and documentation in this fork (`justr4mzi/VIctory`, branch `v16-rx6700xt`, tag `v16-ready`) are AI-assisted. No title files, names, hashes, or offsets are included in this repository.

## Hardware Target & Tested Baseline

- **OS:** Windows 11 x64
- **CPU:** AMD Ryzen 5 5600 (6C / 12T)
- **RAM:** 48 GB DDR4
- **GPU:** AMD Radeon RX 6700 XT (RDNA2, 12 GB VRAM, ~11.2 GB DXGI budget)

## Root Cause Fix for RDNA2 (`0x887A0007` / Kernel Memory Exhaustion)

Diagnostic runs with `VICTORY_D3D12_DEBUG=1` + Windows Event Log showed **0** TDR driver resets (`amdkmdag`), and instead D3D12 Debug Layer `id=834` (*"D3D12: Kernel memory failure. There might be a memory leak."*), with >43,900 live resources and >8,200 live heaps before `0x887A0007`, exacerbated on RX 6700 XT when `VICTORY_RELAY`, `VICTORY_RELAY_COPY`, `VICTORY_UPLOAD_AHEAD`, `VICTORY_KEEP_FREED`, or `VICTORY_CARD_AWAKE` stay aggressive.

Key fixes & multi-fork additions in `v16-ready`:

1. **Exact Pool Bounds & Sweep (`crates/victory-gpu/src/shadow.rs`, `host.rs`):** Bounded `EXACT_POOL` (`POOL_BUCKET_CAP = 32`, `POOL_TOTAL_CAP = 256`, periodic prune) + `trim_exact_pool()` invoked from `sweep()`.
2. **12 GB RX 6700 XT VRAM Keep Thresholds (`crates/victory-gpu/src/host.rs`):** `vram_keep(budget, used)` triggers soft/hard trim earlier (`76%` -> `72%`, `85%` -> `75%`) when DXGI budget `<= 14 GiB`.
3. **Opt-in Occluded Meter Logging (`crates/victory-gpu/src/host.rs`):** `VICTORY_METER_OCCLUDED=1` records FPS lines even when the window is occluded (`DXGI_STATUS_OCCLUDED`), keeping default behavior unchanged (`0`).
4. **Merged `avatarmanz/main` (`0dec5dcc`) + Cherry-picks:**
   - `VICTORY_OUTPUT` (`1280x720`, `1664x936`, `1920x1080`, `2560x1440`, `3840x2160`) & `VICTORY_RENDER_SCALE` (`victory-loader/src/resolution.rs`)
   - `VICTORY_PORT_CBV=1` (`victory-gpu/src/port_cbv.rs`)
   - Descriptor fallback skip on marked runs (`victory-gpu/src/descriptor.rs`, upstream PR #178)
   - `VICTORY_PORTABLE_FLOAT_FAST=1` opt-in constant float clamp (`victory-shader/src/emit/lower.rs`, upstream PR #190)
   - `VICTORY_WARM_BAD` (`warm-bad.txt`) skip support while preserving `VICTORY_PIPELINE_LOW_MEMORY` (`victory-gpu/src/pipeline/make.rs`, upstream PR #170)
   - Size-tiered card-buffer headroom & fence-safe LRU demotion enabling `VICTORY_CARD_BUFFERS_MIN_KB=64` (`victory-gpu`, upstream PR #180)
   - Earlier `v16` integrations kept: `Belobug` magenta clamp in `ambient.rs`, `ka1qux` `aniso_point_mip` & `tiles::let_go`, `triippywhale` upload-cache trim, `OnceLock` `Shadow::uav_allowed()`, and `VICTORY_KEYS_FILE` / `VICTORY_KEYS_ANYWHERE` (`gameinput.rs`).

## Verified RX 6700 XT 3D Benchmarks (`draws >= 100`, `0` Errors)

| Profile | Resolution | Key Flags | 3D Samples (`n`) | FPS (med / mean / max) | Frame ms (med) | `heap_ms` (med) | `gpu` (med) |
|---|---|---|---|---|---|---|---|
| `v14` (baseline) | Native | `RELAY=2`, `ambient` | `2,042` (3h24m) | `1.90` / `1.93` / `4.50` | `532.25 ms` | `355.60 ms` | `136.20 ms` |
| `v15` (prev rec.) | Native | Marked heap, `LOOKUP_CACHE=1` | `1,902` (3h20m) | `1.90` / `1.91` / `4.00` | `534.30 ms` | `399.05 ms` | `132.85 ms` |
| `v16-safe` | `4K` | `RELAY=0`, `KEEP_FREED=0` | `60` | `1.80` / `1.93` / `4.50` | `570.85 ms` | **`2.35 ms`** | `225.00 ms` |
| `v16-norelay` | `4K` | `+VICTORY_THIN=1`, `RELAY=0` | `31` | `2.10` / `1.84` / `2.40` | `484.10 ms` | `12.10 ms` | `198.30 ms` |
| `v16-turbo` | `4K` | `+PORT_CONSTANTS=1 +PORT_DESCRIPTORS=1` | `28` | `2.30` / `2.20` / `3.00` | `443.00 ms` | `9.15 ms` | `199.05 ms` |
| **`v16` (Balanced)** | **`1920x1080`** | `+PORT_CBV=1 +CARD_BUFFERS_MIN_KB=64` | `51` | **`4.10`** / `3.39` / `4.60` | `241.90 ms` | **`2.50 ms`** | `109.10 ms` |
| **`v16 Turbo`** | **`1280x720`** | `+PORTABLE_FLOAT_FAST=1`, 4 thin threads | `30` | **`4.60`** / **`4.29`** / **`9.60`** (`p95 7.10`) | **`216.15 ms`** (`min 103.60`) | `10.55 ms` | **`96.30 ms`** |

## Recommended Crash-Free RX 6700 XT Settings

Keep these memory settings for stability on RDNA2 12 GB (`0` DXGI resets):

- `VICTORY_RELAY=0`, `VICTORY_RELAY_COPY=0`, `VICTORY_UPLOAD_AHEAD=0`
- `VICTORY_KEEP_FREED=0`, `VICTORY_BURY_LAG=16`, `VICTORY_TEXTURE_KEEP=0`, `VICTORY_CARD_AWAKE=0`
- `VICTORY_RESIDENCY=1`, `VICTORY_SWEEP_PER_HEAP=1`, `VICTORY_PIPELINE_LOW_MEMORY=1`
- `VICTORY_CARD_BUFFERS_MIN_KB=64`, `VICTORY_CARD_BUFFERS_MB=3072`
- `VICTORY_PORT_CONSTANTS=1`, `VICTORY_PORT_DESCRIPTORS=1`, `VICTORY_PORT_CBV=1`
- `VICTORY_OUTPUT=1920x1080` (Balanced, `~4.1 FPS` med) or `VICTORY_OUTPUT=1280x720` + `VICTORY_PORTABLE_FLOAT_FAST=1` (Turbo Max FPS, `4.6` med / up to `9.6 FPS`).
- Optional debug toggles for automated capture/diagnostics: `VICTORY_D3D12_DRED=light`, `VICTORY_D3D12_DEBUG=1`, `VICTORY_METER_OCCLUDED=1`.
