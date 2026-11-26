# VIctory `v17-ready` — AMD Radeon RX 6700 XT (RDNA2) & Zen 3 Fork

> **AI-assisted development note:** All code integrations, conflict resolutions, bug fixes, benchmarks, and documentation in this fork (`justr4mzi/VIctory`, branch `v17-rx6700xt`, tag `v17-ready`) are AI-assisted. No title files, names, hashes, or offsets are included in this repository.

## Hardware Target & Tested Baseline

- **OS:** Windows 11 x64
- **CPU:** AMD Ryzen 5 5600 (6C / 12T, Zen 3 — compiled with `-C target-cpu=native`)
- **RAM:** 48 GB DDR4
- **GPU:** AMD Radeon RX 6700 XT (RDNA2, 12 GB VRAM, ~11.47 GB DXGI budget)

## What Changed in `v17-ready` (Breakthrough Over `v16-ready`)

In `v16-ready`, setting `VICTORY_RELAY=0` stopped the RDNA2 `0x887A0007` device-removed resets but also serialized `prep + GPU + copy-back`, leaving the GPU idle ~49% of the frame and capping 3D throughput at `4.10–4.60 FPS` (`96–109 ms` GPU).

`v17-ready` combines the latest upstream renderer port, the RDNA2 send-ahead table-flush fix, VRAM auto-mip clamping, fence-safe LRU texture eviction, HLE memory/leak fixes, shader correctness PRs, and Zen 3 native codegen — unlocking **`7.80 FPS` median / `19.30 FPS` peak at 1080p** and **`9.30 FPS` median / `14.80 FPS` peak at 720p** with **`0` errors**:

1. **RDNA2 Send-Ahead Unlock (`VICTORY_LATE_WAIT=3` + `VICTORY_RELAY=0` + `VICTORY_LATE_FLUSH_TABLES=1`, upstream issue #113):**
   - `Detached::tables_done()`, `late_flush_tables()`, and `tables_back()` in `crates/victory-gpu/src/list.rs`, wired into `Host::land()`, `relay_land()`, and `settle()` (`crates/victory-gpu/src/host.rs`) so completed descriptor tables are recycled immediately instead of piling up across in-flight submissions.
   - Pairing `VICTORY_LATE_WAIT=3` with `VICTORY_RELAY=0`, `VICTORY_COPIES_IN_PLACE=0`, and `VICTORY_BURY_LAG=16` enables 3-frame send-ahead pipelining and free presents without spawning the RDNA2-unstable `relay_worker` thread — cutting GPU time **7.2x** (`109.10 ms` -> `15.10 ms` at 1080p) with **`0`** `0x887A0007` resets.
2. **Renderer Port #6 (`VICTORY_PORT_GUARD=1`, merged `origin/main` `a9400a6d`):**
   - Shared read/write lock on write guards with per-64 KiB block claims and concurrent submission planning (`VICTORY_PORT_GUARD_PLAN`), slashing title-thread stalls.
3. **Fence-Safe 2-Pass Texture LRU Eviction & Auto Mip Clamp (merged upstream PR #212):**
   - `VICTORY_TEXTURE_LRU_EVICT=1` tracks last-submitted fence per texture (`stamp_textures`) and evicts cold textures safely only after `completed_fence` passes.
   - `VICTORY_TEXTURE_AUTO_MIP_CLAMP=1` caps oversized textures (`<= 2048 px` on `<= 12 GiB` adapters), keeping RX 6700 XT VRAM cleanly around `5.4–7.6 GB / 11.47 GB` with zero shared-RAM spill.
4. **Idle Texture Memory Pressure Relief & Placed-View Padding Zeroing (merged upstream PR #206):**
   - Frees textures idle >5 s under PC memory pressure and zeroes trailing padding in placed-view shapes.
5. **Shader Correctness PRs (merged upstream PR #184 & PR #204):**
   - Wave-uniformity proof fix (`PR #184`) and LDS atomic ordering between divergent half-waves (`PR #204`).
6. **HLE Memory & Handle Leak Fixes (upstream issues #216, #217, #220):**
   - `xmem.rs` (#216): 32-bit `XMemFreeDefault` now notifies `title_memory_freed`.
   - `dstorage.rs` (#217): bounded/deduplicated `WALLS_FILES` tracking.
   - `net.rs` (#220): bounded `SEEN` set replacing unbounded linear scan.
7. **Zen 3 Native Codegen (`-C target-cpu=native`):**
   - Compiles with AVX2/FMA/BMI2/Zen 3 scheduling for both `--profile diag` and `--profile ship`.
8. **All `v16-ready` Fixes Preserved:**
   - Bounded `EXACT_POOL` + `trim_exact_pool()` in `shadow.rs`/`host.rs`, 4-tier 12 GB `vram_keep`, `VICTORY_OUTPUT`, `VICTORY_PORT_CONSTANTS=1`, `VICTORY_PORT_DESCRIPTORS=1`, `VICTORY_PORT_CBV=1`, `VICTORY_PORTABLE_FLOAT_FAST=1`, `VICTORY_CARD_BUFFERS_MIN_KB=64`, `VICTORY_WARM_BAD`, `VICTORY_PIPELINE_LOW_MEMORY=1`, `VICTORY_METER_OCCLUDED=1`, `VICTORY_KEYS_FILE` / `VICTORY_KEYS_ANYWHERE`, `Belobug` magenta clamp, `ka1qux` `aniso_point_mip` & `tiles::let_go`, and `triippywhale` upload-cache trim.

## Verified RX 6700 XT 3D Benchmarks (`draws >= 100`, `0` Errors on `v16`/`v17`)

All `v17` numbers below were recorded live with the **`diag`** build (`--features diagnostics` + frame PNG capture every 8 frames). The `--profile ship` build (`victory-v17.exe`) strips diagnostics and capture overhead.

| Profile | Resolution | Key Flags | 3D Samples (`n`) | FPS (med / mean / p95 / max) | Frame ms (med / min) | `prep` (med) | `gpu` (med) | `back` (med) | `heap_ms` (med) |
|---|---|---|---|---|---|---|---|---|---|
| `v14` (baseline) | Native | `RELAY=2`, `ambient` | `2,042` | `1.90` / `1.93` / `2.60` / `4.50` | `532.25` / `222.20` | `62.80 ms` | `136.20 ms` | `54.10 ms` | `355.60 ms` |
| `v15` (prev rec.) | Native | Marked heap, `LOOKUP_CACHE=1` | `1,902` | `1.90` / `1.91` / `2.50` / `4.00` | `534.30` / `249.80` | `67.60 ms` | `132.85 ms` | `50.40 ms` | `399.05 ms` |
| `v16` (Balanced) | `1920x1080` | `RELAY=0`, Ports 1+2+5, `MIN_KB=64` | `51` | `4.10` / `3.39` / `4.50` / `4.60` | `241.90` / `215.30` | `66.40 ms` | `109.10 ms` | `43.00 ms` | `2.50 ms` |
| `v16 Turbo` | `1280x720` | `+PORTABLE_FLOAT_FAST=1` | `30` | `4.60` / `4.29` / `7.10` / `9.60` | `216.15` / `103.60` | `61.25 ms` | `96.30 ms` | `37.45 ms` | `10.55 ms` |
| **`v17` (Ultime)** | **`1920x1080`** | **`LATE_WAIT=3`, Port #6, PR #212, `PLANES=0`** | `47` | **`7.80`** / **`8.06`** / **`18.30`** / **`19.30`** | **`128.50`** / **`51.90`** | **`50.00 ms`** | **`15.10 ms`** | **`33.40 ms`** | **`1.70 ms`** |
| **`v17 Turbo`** | **`1280x720`** | **`LATE_WAIT=3`, Port #6, PR #212, `PLANES=0`** | `54` | **`9.30`** / **`8.68`** / **`11.40`** / **`14.80`** | **`107.65`** / **`67.70`** | **`46.15 ms`** | **`14.40 ms`** | **`24.60 ms`** | **`1.10 ms`** |

- **`drop_ind`**: reduced from `632.00` (`v14`/`v15`) to **`0.00`** in both `v17` profiles.
- **Stability**: **`ERR=0`** across `47` samples (`~4m08s` 3D) at `1080p` and `54` samples (`~4m45s` 3D) at `720p`.

## Recommended Crash-Free RX 6700 XT Settings (`v17-ready`)

Core stability & send-ahead unlock (`0` DXGI resets):

- `VICTORY_LATE_WAIT=3`, `VICTORY_LATE_FLUSH_TABLES=1`, `VICTORY_COPIES_IN_PLACE=0`
- `VICTORY_RELAY=0`, `VICTORY_RELAY_COPY=0`, `VICTORY_UPLOAD_AHEAD=0`
- `VICTORY_KEEP_FREED=0`, `VICTORY_BURY_LAG=16`, `VICTORY_TEXTURE_KEEP=0`, `VICTORY_CARD_AWAKE=0`
- `VICTORY_TEXTURE_LRU_EVICT=1`, `VICTORY_TEXTURE_AUTO_MIP_CLAMP=1`
- `VICTORY_RESIDENCY=1`, `VICTORY_SWEEP_PER_HEAP=1`, `VICTORY_PIPELINE_LOW_MEMORY=1`
- `VICTORY_PORT_GUARD=1`, `VICTORY_PORT_CONSTANTS=1`, `VICTORY_PORT_DESCRIPTORS=1`, `VICTORY_PORT_CBV=1`, `VICTORY_PORTABLE_FLOAT_FAST=1`
- `VICTORY_CARD_BUFFERS_MIN_KB=64`, `VICTORY_CARD_BUFFERS_MB=1536`, `VICTORY_PLANES=0`
- `VICTORY_OUTPUT=1920x1080` (`7.80` med / `19.30` max FPS) or `VICTORY_OUTPUT=1280x720` (`9.30` med / `14.80` max FPS), `VICTORY_PANEL=0` (toggle FPS panel on demand with `F3`).
- Optional diagnostics: `VICTORY_D3D12_DRED=light`, `VICTORY_D3D12_DEBUG=1`, `VICTORY_METER_OCCLUDED=1`.
