# VIctory `v20-ready` (1080p Ultime) — AMD Radeon RX 6700 XT (RDNA2) & Zen 3 Fork

> **AI-assisted development note:** All code integrations, conflict resolutions, custom CPU optimizations, bug fixes, benchmarks, and documentation in this fork (`justr4mzi/VIctory`, branch `v20-rx6700xt`, tag `v20-ready`) are AI-assisted. No title files, names, hashes, or offsets are included in this repository.

## Hardware Target & Tested Baseline

- **OS:** Windows 11 x64
- **CPU:** AMD Ryzen 5 5600 (6C / 12T, Zen 3 — compiled with `-C target-cpu=native`)
- **RAM:** 48 GB DDR4
- **GPU:** AMD Radeon RX 6700 XT (RDNA2, 12 GB VRAM, ~11.47 GB DXGI budget)
- **Target Resolution:** `1920x1080` (`VICTORY_OUTPUT=1920x1080`)

## What Is New in `v20-ready` (Over `v19-ready`)

### 1. All 15 CPU/GPU Performance Commits from `XaviFortes/VIctory-RDNA`

1. **`VICTORY_PORT_BUFFERS=1` (`b2917b48` + `4487275b`):** Renderer Port #8 — places contiguous-section title buffers above 4 GiB directly in place (zero watch, zero `VirtualProtect`, zero refresh, zero copy-back), while keeping big GPU-heavy buffers in local GDDR6 VRAM so they do not stall over PCIe.
2. **AVX2 & Lock-Free Shadow Fast Paths (`6072c374`, `fdfb306d`):** Uses AVX2 `word_runs` in `Shadow::take` instead of scalar `dword_runs`, plus lock-free `has_pending` fast path and `RwLock<CopiedState>` for `nothing_new`.
3. **Write-Watch & Journal Sharding (`8abd3671`, `56638976`, `e1f8b656`, `83ba8e20`):** Coalesces unguard `VirtualProtect` runs, prebuilds null descriptor templates, short-circuits `ask_ahead` when `ask_once` is active, caches `Batch::last_ahead`, uses `AddressHash` for textures, lowers `WIDE_PLAN_STEPS`, narrows journal shards to 4 MiB (`SHARD_SHIFT=6`), scales `WIDE_JOB` fan-out to 256 KiB, and runs single-pass `ask_once`.
4. **Descriptor Heap & Draw Recording (`61477a81`, `dc1675b2`, `3c89cc40`, `f8eb38d5`, `13e1f5ed`):** CPU-side `strides`/`view_addr` shadow cache, per-slot `null_kind`/`has_kept` cache, packed `DynamicBits` bitset allocator (`is_taken` adapted for LRU eviction and `slot_story`), batched `release_dynamic_many`, `BufferSweep` deduplication and per-part span caching, and lock-free mirror fast path.
5. **Parallel Texture Readback (`e4250eb5`, `af65a791`):** Parallelizes flight texture readbacks, reuses thread-local scratch buffers, narrows `TICKETS` lock, and borrows `texture_backs` cleanly.

### 2. Five New Upstream PRs + `PastiiJay/klbf-lab` Intro Skip

- **`PR #304` (`VICTORY_MAP_LOCK_FINE=1`):** Replaces the global address-space mapping `RwLock` with per-range locking so unrelated writes do not stall draw-side readers.
- **`PR #306` (`VICTORY_PARTIAL_DELTA=1` + `VICTORY_REFRESH_ONCE=1`, ScarlettPort deltas):** Content-addressed per-subresource texture delta uploads and submit-scope refresh deduplication.
- **`PR #314` (`VICTORY_AMBIENT_REFLECT=1`):** Clamps traced reflections indoors to surrounding ambient light so eyes, glasses, and indoor glass panes no longer reflect the open sky when ray tracing is off.
- **`PR #311` (`VICTORY_BORDERLESS=1`):** Opt-in borderless fullscreen covering the primary display.
- **`PR #303`:** Per-pipeline wait budget across frames and panic-safe refusal.
- **`PastiiJay/klbf-lab` (`60eca6bb`, `VICTORY_SKIP_MOVIES=open`):** Opt-in Bink intro movie skip for faster boot into gameplay (`3/3` unit tests passing).
- **Policy Compliance (`913e1f3c`):** Extracted `host::refresh_tests` into `crates/victory-gpu/src/host/refresh_tests.rs` so `host.rs` stays under the 512 KiB `xtask policy` limit (`516,326 B < 524,288 B`).

### 3. Live A/B Tuning Validated on Ryzen 5 5600 (6C/12T) + RX 6700 XT

- **`VICTORY_TEXTURE_LRU_EVICT=1` re-enabled cleanly** (safe with `VICTORY_THIN=0`): keeps VRAM at `~8.9 / 10.2 GB` across zone transitions instead of hitting `no room for new ones`.
- **Proven WORSE on 6-core Ryzen 5 5600 (kept OFF):** `VICTORY_LANE_CORE=1`, `VICTORY_FAST_CORES=1`, and `VICTORY_WINDOWS_BAR_MB=1024` cost `-11.4 %` mean FPS (`prep` `+10 ms`) in live prologue A/B logs (`7.34 -> 6.50 FPS` at `>= 4,500 draws`).
- Retained from `v19`: `VICTORY_WIDE_THREADS=8`, `VICTORY_WIDE_MERGE=1`, `VICTORY_ASYNC_LAND=1`, `VICTORY_LATE_WAIT=3`, `VICTORY_CARD_BUFFERS_MB=2048`, `VICTORY_CARD_WINDOWS_MB=1024`, `VICTORY_FRAME_INTERVAL_MS=16.667`, Ports #1–#7, `VICTORY_DSTORAGE_STAGING_MB=256`, `VICTORY_PIPELINE_WORKERS=8`.

## Recommended RX 6700 XT `1920x1080` Settings (`v20-ready`)

- **Output & Boot:** `VICTORY_OUTPUT=1920x1080`, `VICTORY_FRAME_INTERVAL_MS=16.667`, `VICTORY_SKIP_MOVIES=open`, `VICTORY_AMBIENT_REFLECT=1`, `VICTORY_PLANES=0` (`1` for `--console series-s`).
- **v20 CPU & Ports #1–#8:** `VICTORY_PORT_BUFFERS=1` (`0` in Safe/LightVRAM profile), `VICTORY_PORT_JOURNAL=1`, `VICTORY_PORT_GUARD=1`, `VICTORY_PORT_CONSTANTS=1`, `VICTORY_PORT_DESCRIPTORS=1`, `VICTORY_PORT_CBV=1`, `VICTORY_PORTABLE_FLOAT_FAST=1`, `VICTORY_MAP_LOCK_FINE=1`, `VICTORY_PARTIAL_DELTA=1`, `VICTORY_REFRESH_ONCE=1`.
- **Threading & Async:** `VICTORY_WIDE_MERGE=1`, `VICTORY_WIDE_THREADS=8`, `VICTORY_UNDER_THREADS=6`, `VICTORY_VIEW_THREADS=6`, `VICTORY_ASYNC_LAND=1`, `VICTORY_WIDE_BACK=1`, `VICTORY_JOIN_COPIES=1`, `VICTORY_COPY_BESIDE=1`, `VICTORY_STORE_BESIDE=1`.
- **RDNA2 Send-Ahead & 12 GB VRAM:** `VICTORY_LATE_WAIT=3`, `VICTORY_LATE_WAIT_LISTS=512`, `VICTORY_LATE_FLUSH_TABLES=1`, `VICTORY_CARD_BUFFERS_MIN_KB=64`, `VICTORY_CARD_BUFFERS_MB=2048`, `VICTORY_CARD_WINDOWS_MB=1024`, `VICTORY_KEEP_FREED=1`, `VICTORY_BURY_LAG=16`, `VICTORY_TEXTURE_AUTO_MIP_CLAMP=1`, `VICTORY_TEXTURE_LRU_EVICT=1`, `VICTORY_SPARES_BY_HEADROOM=1`, `VICTORY_COPIES_IN_PLACE=0`, `VICTORY_RELAY=0`, `VICTORY_THIN=0`, `VICTORY_ASYNC_SUBMIT=0`.
- **Loading & Stability:** `VICTORY_DSTORAGE_STAGING_MB=256`, `VICTORY_XMA_REFILL=1`, `VICTORY_FALLBACK_HEAPS=1`, `VICTORY_RESIDENCY=1`, `VICTORY_SWEEP_PER_HEAP=1`, `VICTORY_PIPELINE_LOW_MEMORY=0`, `VICTORY_PIPELINE_WORKERS=8`, `VICTORY_PIPELINE_WAIT_MS=50`, `VICTORY_LOG=error`.
