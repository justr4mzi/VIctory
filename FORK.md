# VIctory `v18-ready` (Ultimate) — AMD Radeon RX 6700 XT (RDNA2) & Zen 3 Fork

> **AI-assisted development note:** All code integrations, conflict resolutions, bug fixes, benchmarks, and documentation in this fork (`justr4mzi/VIctory`, branch `v18-rx6700xt`, tag `v18-ready`) are AI-assisted. No title files, names, hashes, or offsets are included in this repository.

## Hardware Target & Tested Baseline

- **OS:** Windows 11 x64
- **CPU:** AMD Ryzen 5 5600 (6C / 12T, Zen 3 — compiled with `-C target-cpu=native`)
- **RAM:** 48 GB DDR4
- **GPU:** AMD Radeon RX 6700 XT (RDNA2, 12 GB VRAM, ~11.47 GB DXGI budget)

## What Is New in `v18-ready` (Over `v17-ready`)

1. **Renderer Port #7 (`VICTORY_PORT_JOURNAL=1`, merged `origin/main` `c9c932cb`):**
   - Splits the journal's notes into 64 sharded locks and records the required heap round at submit time (`-2.4 ms` on the submit lane).
2. **10–12 GB Card-Buffer Defaults (merged upstream PR #225):**
   - Automatically grants 10–12 GB cards (`> 9 GiB .. <= 12 GiB` DXGI budget) 64 KiB minimum / 1536 MiB card-buffer budget.
3. **Prepared Image Cache & Faster Warm-Up (merged upstream PR #239, PR #186, PR #188):**
   - Fixes race-free staging publication in `crates/victory-loader/src/publish.rs` (6 unit tests passing).
   - Wakes the `warm` pipeline watchdog immediately on completion (`PR #186`) and stops scanning shader archives at the warm-up deadline (`PR #188`).
4. **Audio, Memory & Async Stability Fixes (merged upstream PR #248, PR #247, PR #244):**
   - `PR #248` (`audio.rs`): Keeps freed audio heap blocks alive while the XMA decoder holds slices (prevents use-after-free).
   - `PR #247` (`xmem.rs`): Checks `n * PAGE` and `base + len` overflow in `XMemMapPhysicalPages`.
   - `PR #244` (`xasync.rs`): Waits for in-flight provider calls during cleanup and scopes `XTaskQueueTerminate` cancellations properly.
5. **Best of `codec-d/cool_fox` Enabled Cleanly on RDNA2:**
   - `VICTORY_DSTORAGE_STAGING_MB=256` (256 MiB DirectStorage staging so multi-MB chunks do not starve worker cores during loading/streaming).
   - `VICTORY_XMA_REFILL=1` (clean XMA voice refill & whole-block publishing).
   - `VICTORY_FALLBACK_HEAPS=1`, `VICTORY_RESIDENCY=1`, `VICTORY_SWEEP_PER_HEAP=1`, `VICTORY_AMBIENT_AIR=1.5`.
6. **Two Critical Launcher Fixes Discovered During Live RX 6700 XT Sessions:**
   - **Pipeline stall fix:** Setting `VICTORY_PIPELINE_LOW_MEMORY=0`, `VICTORY_PIPELINE_WORKERS=8`, and `VICTORY_PIPELINE_WAIT_MS=50` prevents the 500 ms/frame stall (`1.8 FPS`) caused by `slow_limit=1` when hundreds of pipelines remain queued on 3D entry.
   - **`THIN=1` + `LRU_EVICT` race fix:** Setting `VICTORY_TEXTURE_LRU_EVICT=0` while keeping `VICTORY_TEXTURE_AUTO_MIP_CLAMP=1` avoids unmaking 170+ heap slots concurrently with `VICTORY_THIN=1` command-list recording (`0x887A0007`), while keeping VRAM cleanly at `5.4–7.6 GB / 11.47 GB`.
   - **`Safe-Fast` fallback profile (`Lancer VIctory v18 Safe.bat`):** Runs `VICTORY_LATE_WAIT=3` + Ports #1–#7 + `AUTO_MIP_CLAMP=1` with `VICTORY_THIN=0` (proven 0-error baseline).

## Verified RX 6700 XT 3D Benchmarks (`draws >= 100`, `0` Errors on `v16`/`v17`)

| Profile | Resolution | Key Flags | 3D Samples (`n`) | FPS (med / mean / p95 / max) | Frame ms (med / min) | `prep` (med) | `gpu` (med) | `back` (med) | `heap_ms` (med) |
|---|---|---|---|---|---|---|---|---|---|
| `v14` (baseline) | Native | `RELAY=2`, `ambient` | `2,042` | `1.90` / `1.93` / `2.60` / `4.50` | `532.25` / `222.20` | `62.80 ms` | `136.20 ms` | `54.10 ms` | `355.60 ms` |
| `v15` (prev rec.) | Native | Marked heap, `LOOKUP_CACHE=1` | `1,902` | `1.90` / `1.91` / `2.50` / `4.00` | `534.30` / `249.80` | `67.60 ms` | `132.85 ms` | `50.40 ms` | `399.05 ms` |
| `v16` (Balanced) | `1920x1080` | `RELAY=0`, Ports 1+2+5, `MIN_KB=64` | `51` | `4.10` / `3.39` / `4.50` / `4.60` | `241.90` / `215.30` | `66.40 ms` | `109.10 ms` | `43.00 ms` | `2.50 ms` |
| `v16 Turbo` | `1280x720` | `+PORTABLE_FLOAT_FAST=1` | `30` | `4.60` / `4.29` / `7.10` / `9.60` | `216.15` / `103.60` | `61.25 ms` | `96.30 ms` | `37.45 ms` | `10.55 ms` |
| `v17` (Ultime) | `1920x1080` | `LATE_WAIT=3`, Port #6, `AUTO_MIP_CLAMP=1`, `PLANES=0` | `47` | `7.80` / `8.06` / `18.30` / `19.30` | `128.50` / `51.90` | `50.00 ms` | `15.10 ms` | `33.40 ms` | `1.70 ms` |
| `v17 Turbo` | `1280x720` | `LATE_WAIT=3`, Port #6, `AUTO_MIP_CLAMP=1`, `PLANES=0` | `54` | `9.30` / `8.68` / `11.40` / `14.80` | `107.65` / `67.70` | `46.15 ms` | `14.40 ms` | `24.60 ms` | `1.10 ms` |

## Recommended RX 6700 XT Settings (`v18-ready`)

- **Ports & Pipeline:** `VICTORY_PORT_JOURNAL=1`, `VICTORY_PORT_GUARD=1`, `VICTORY_PORT_CONSTANTS=1`, `VICTORY_PORT_DESCRIPTORS=1`, `VICTORY_PORT_CBV=1`, `VICTORY_PORTABLE_FLOAT_FAST=1`
- **RDNA2 Send-Ahead & Memory:** `VICTORY_LATE_WAIT=3`, `VICTORY_LATE_FLUSH_TABLES=1`, `VICTORY_COPIES_IN_PLACE=0`, `VICTORY_RELAY=0`, `VICTORY_KEEP_FREED=0`, `VICTORY_BURY_LAG=16`, `VICTORY_TEXTURE_AUTO_MIP_CLAMP=1`, `VICTORY_TEXTURE_LRU_EVICT=0`
- **Threading & `cool_fox` Features:** `VICTORY_THIN=1`, `VICTORY_THIN_THREADS=6`, `VICTORY_ASYNC_SUBMIT=1`, `VICTORY_JOIN_PARTS=6`, `VICTORY_DSTORAGE_STAGING_MB=256`, `VICTORY_XMA_REFILL=1`, `VICTORY_FALLBACK_HEAPS=1`, `VICTORY_RESIDENCY=1`, `VICTORY_SWEEP_PER_HEAP=1`
- **Fast Shader/Pipeline Loading (48 GB RAM / 12T):** `VICTORY_PIPELINE_LOW_MEMORY=0`, `VICTORY_PIPELINE_WORKERS=8`, `VICTORY_PIPELINE_WAIT_MS=50`, `VICTORY_LOG=warn`, `VICTORY_PANEL=0` (`F3` toggles FPS panel).
