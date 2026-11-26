# VIctory `v19-ready` (1080p Ultime) — AMD Radeon RX 6700 XT (RDNA2) & Zen 3 Fork

> **AI-assisted development note:** All code integrations, conflict resolutions, custom CPU optimizations, bug fixes, benchmarks, and documentation in this fork (`justr4mzi/VIctory`, branch `v19-rx6700xt`, tag `v19-ready`) are AI-assisted. No title files, names, hashes, or offsets are included in this repository.

## Hardware Target & Tested Baseline

- **OS:** Windows 11 x64
- **CPU:** AMD Ryzen 5 5600 (6C / 12T, Zen 3 — compiled with `-C target-cpu=native`)
- **RAM:** 48 GB DDR4
- **GPU:** AMD Radeon RX 6700 XT (RDNA2, 12 GB VRAM, ~11.47 GB DXGI budget)
- **Target Resolution:** `1920x1080` (`VICTORY_OUTPUT=1920x1080`)

## What Is New in `v19-ready` (Over `v18-ready`)

### 1. Two Custom Surgical CPU Optimizations (A/B Proven on Ryzen 5 5600 + RX 6700 XT)

1. **`VICTORY_WIDE_THREADS=<2..12>` (`crates/victory-gpu/src/shadow.rs`):**
   - Previously `shadow::wide()` hardcoded `n.get() / 4` (only 3 threads on a 12-thread Ryzen 5 5600).
   - Unlocking `VICTORY_WIDE_THREADS=8` together with `VICTORY_WIDE_MERGE=1` and `VICTORY_WIDE_BACK=1` cuts `copying back` from **`33.40 ms` -> `18.55 ms` (-44%)** in `1920x1080` with `0` errors.
2. **`VICTORY_ASYNC_LAND=1` (`crates/victory-gpu/src/list.rs` + `host.rs`):**
   - Offloads landed command-list reset/recycle (`land()`) to `late_helper` (with backpressure `< 24`), slashing `after a present` wait from **`14.1 ms` -> `2.4–8.9 ms`**.

### 2. Eleven Merged Upstream PRs (0 Conflicts, Verified Offline & Live)

- **VRAM, Memory & Queues:**
  - `PR #293`: Retry textures refused for VRAM instead of refusing permanently (`resource.rs`, plus `lru_evict_enabled()` safety guard).
  - `PR #288`: Bound `texstore` queue to at most 8 waiting textures.
  - `PR #285`: Pool empty blocks follow VRAM headroom (`VICTORY_SPARES_BY_HEADROOM=1`).
  - `PR #273`: Renderer Port #2 (`port_descriptors.rs`) clears stale fallback pages when the marked heap drops.
- **Shaders & Wave64 Correctness:**
  - `PR #279`: Fix SGPR destination past `exec_hi` crash (even in release builds).
  - `PR #286`: Fix Wave64 own-bits scalar-joined loops.
  - `PR #296`: Fix whole-wave `v_permlanex16` fetch-inactive lanes.
- **RDNA2 / Platform / UI:**
  - `PR #299`: RDNA2 (`RX 6700 XT`) descriptor copy test fix (`descriptor/host.rs`).
  - `PR #300`: `VICTORY_D3D12_ADAPTER` matches GPU model numbers (e.g. `6700`).
  - `PR #284`: `VICTORY_CACHE_CORES` reads the last CPU cache record.
  - `PR #295`: Fit dynamic UI/subtitles viewboxes on scaled outputs (`VICTORY_UI_VIEWBOX=1`).

### 3. Live `1920x1080` A/B Tuning Discoveries on RX 6700 XT (157 3D Samples, `ERR = 0`)

- **`VICTORY_CARD_WINDOWS_MB=1024`** moves 1 GB of vertex/index/constant windows directly into GDDR6 VRAM, slashing **`gpu` median from `15.10 ms` -> `8.80 ms` (-42%, 12.4x faster than `v16`)** and eliminating 3-second stalls (`mean 150.85 ms`, `max 571.70 ms`).
- **`VICTORY_COPY_BESIDE=1` + `VICTORY_STORE_BESIDE=1`** bypassed `239,703` copies and `274,004` stores in a single prologue run.
- **`VICTORY_FRAME_INTERVAL_MS=16.667`** unlocks smooth **`60.0 FPS`** menus, loading screens, and videos.
- **Settings proven WORSE on Ryzen 5 5600 + RX 6700 XT (kept OFF in launchers):**
  - `VICTORY_RELAY=1/2`, `VICTORY_THIN=1` + `VICTORY_ASYNC_SUBMIT=1`, `VICTORY_KEEP_FREED=0`, `VICTORY_TEXTURE_LRU_EVICT=1` -> trigger `0x887A0007` on RX 6700 XT.
  - `VICTORY_TEXTURE_KEEP=1` -> trap in `resource.rs` (`!matches!(var, Ok("0") | Ok("1"))`) that disables `vram_keep`! Leave `VICTORY_TEXTURE_KEEP` unset.
  - `VICTORY_ASK_SPREAD=1` / `VICTORY_ASK_NORESET=1` -> doubled write-watch cost (`6.1 ms` -> `10.0 ms`).
  - `VICTORY_SYNC_FINE=1` / `VICTORY_WIDE_SPIN=150` -> slowed down 6-core Zen 3.

## Verified RX 6700 XT `1920x1080` 3D Prologue Benchmarks (`draws >= 100`, `ERR = 0`, `diag` + PNG Capture)

| Version / Profile | Resolution | Key Flags | 3D Samples (`n`) | Draws (med) | FPS (med / mean / p95 / max) | Frame ms (med / mean / min) | `prep` (med) | `gpu` (med) | `back` (med) | `heap_ms` (med) |
|---|---|---|---|---|---|---|---|---|---|---|
| `v14` (baseline) | Native | `RELAY=2`, `ambient` | `2,042` | `4,893` | `1.90` / `1.93` / `2.60` / `4.50` | `532.25` / `541.46` / `222.20` | `62.80 ms` | `136.20 ms` | `54.10 ms` | `355.60 ms` |
| `v15` | Native | Marked heap, `LOOKUP_CACHE=1` | `1,902` | `4,848` | `1.90` / `1.91` / `2.50` / `4.00` | `534.30` / `545.70` / `249.80` | `67.60 ms` | `132.85 ms` | `50.40 ms` | `399.05 ms` |
| `v16` (Balanced) | `1920x1080` | `RELAY=0`, Ports `1+2+5`, `MIN_KB=64` | `51` | `4,924` | `4.10` / `3.39` / `4.50` / `4.60` | `241.90` / `420.89` / `215.30` | `66.40 ms` | `109.10 ms` | `43.00 ms` | `2.50 ms` |
| `v17` (Ultime) | `1920x1080` | `LATE_WAIT=3`, Port #6, `AUTO_MIP_CLAMP=1` | `47` | `4,450` | `7.80` / `8.06` / `18.30` / `19.30` | `128.50` / `258.92` / `51.90` | `50.00 ms` | `15.10 ms` | `33.40 ms` | `1.70 ms` |
| **`v19` (`V19-A`)** | **`1920x1080`** | `WIDE_THREADS=8`, `ASYNC_LAND=1`, `CARD_WINDOWS_MB=1024` | `52` | **`4,972`** | **`8.35`** / `7.51` / **`9.90`** / **`10.00`** | **`120.10`** / `238.17` / **`99.90`** | `57.55 ms` | `11.10 ms` | **`18.55 ms`** | **`0.60 ms`** |
| **`v19` (`V19-B`, smoothest)** | **`1920x1080`** | `UNDER=6`, `VIEW=6`, `CARD_WINDOWS_MB=1024`, no `ASK_*` | `59` | **`4,974`** | `8.10` / `7.53` / `9.60` / `9.90` | `124.00` / **`150.85`** / `100.70` | `66.60 ms` | **`8.80 ms`** | `19.80 ms` | **`0.60 ms`** |
| **`v19` (`V19-C`, light VRAM)** | **`1920x1080`** | `WIDE_THREADS=8`, `BUFFERS=1536`, `WINDOWS=512` | `46` | **`4,779`** | **`8.35`** / `6.70` / `9.20` / `9.30` | **`119.80`** / `364.03` / `108.00` | `60.45 ms` | `10.65 ms` | `19.75 ms` | **`0.65 ms`** |

> Note: All benchmarks above ran on `victory-v19-diag.exe` (`--features diagnostics`) with live PNG capture and `VICTORY_LOG=info`. The `--profile ship` binary (`victory-v19.exe`, `-C target-cpu=native`, `VICTORY_LOG=error`) removes diagnostic and logging overhead in real gameplay.

## Recommended RX 6700 XT `1920x1080` Settings (`v19-ready`)

- **Output & Pacing:** `VICTORY_OUTPUT=1920x1080`, `VICTORY_FRAME_INTERVAL_MS=16.667`, `VICTORY_PLANES=0` (`1` for `--console series-s`).
- **v19 CPU & Async Pipeline:** `VICTORY_WIDE_MERGE=1`, `VICTORY_WIDE_THREADS=8`, `VICTORY_UNDER_THREADS=6`, `VICTORY_VIEW_THREADS=6`, `VICTORY_ASYNC_LAND=1`, `VICTORY_WIDE_BACK=1`, `VICTORY_JOIN_COPIES=1`, `VICTORY_COPY_BESIDE=1`, `VICTORY_STORE_BESIDE=1`.
- **Ports #1–#7:** `VICTORY_PORT_JOURNAL=1`, `VICTORY_PORT_GUARD=1`, `VICTORY_PORT_CONSTANTS=1`, `VICTORY_PORT_DESCRIPTORS=1`, `VICTORY_PORT_CBV=1`, `VICTORY_PORTABLE_FLOAT_FAST=1`.
- **RDNA2 Send-Ahead & 12 GB VRAM:** `VICTORY_LATE_WAIT=3`, `VICTORY_LATE_WAIT_LISTS=512`, `VICTORY_LATE_FLUSH_TABLES=1`, `VICTORY_CARD_BUFFERS_MIN_KB=64`, `VICTORY_CARD_BUFFERS_MB=2048`, `VICTORY_CARD_WINDOWS_MB=1024`, `VICTORY_KEEP_FREED=1`, `VICTORY_BURY_LAG=16`, `VICTORY_TEXTURE_AUTO_MIP_CLAMP=1`, `VICTORY_TEXTURE_LRU_EVICT=0`, `VICTORY_SPARES_BY_HEADROOM=1`, `VICTORY_COPIES_IN_PLACE=0`, `VICTORY_RELAY=0`, `VICTORY_THIN=0`, `VICTORY_ASYNC_SUBMIT=0`.
- **Loading & Stability:** `VICTORY_DSTORAGE_STAGING_MB=256`, `VICTORY_XMA_REFILL=1`, `VICTORY_FALLBACK_HEAPS=1`, `VICTORY_RESIDENCY=1`, `VICTORY_SWEEP_PER_HEAP=1`, `VICTORY_PIPELINE_LOW_MEMORY=0`, `VICTORY_PIPELINE_WORKERS=8`, `VICTORY_PIPELINE_WAIT_MS=50`, `VICTORY_LOG=error`.
