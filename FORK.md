# Fork: `justr4mzi/VIctory` (`v15-ready` RX 6700 XT Edition, AI-assisted)

Personal fork of [Marvisak/VIctory](https://github.com/Marvisak/VIctory) (built atop `v14-ready` / [`fxceless/victory-fork`](https://github.com/fxceless/victory-fork) and curated PRs across Waves 1–2 for Windows 11 + AMD Ryzen 5 5600 + AMD Radeon RX 6700 XT RDNA2 + 48 GB RAM). Branches `v15-rx6700xt`, `v14-rx6700xt`, `v13-rx6700xt`, `v12-rx6700xt`, `v11-rx6700xt`, `v9-rx6700xt`, `ultimate-rx6700xt`, and `main` target this hardware profile.

Nothing here is upstream-approved; please direct upstream feedback to the original repositories and PR authors. **Zero game data, filenames, executables, hashes, or offsets are included** (`cargo xtask policy` verified: **`213 files ok`**). This fork, its conflict resolutions, and its local fixes/defaults are **AI-assisted**.

## `v15` (`v15-ready`, `v15-rx6700xt`) Changelog & Verification

- **Verified offline tests (`cargo test --workspace -- --test-threads=1`):** **855 passed, 0 failed, 24 ignored** (`victory-gpu --features diagnostics`: **314 passed, 0 failed, 13 ignored**; `cargo xtask policy`: **213 files ok**; all `.rs` files `<= 524,288` B via `host/tests.rs` split).
- **Builds (Thin LTO, `codegen-units = 1`):** Release `9,500,672` B (`SHA-256: 9BE1EDE8A6CC70C9E25DBE4511268CE71DEC0C0590499B0B0463D0C8CFD2D1F7`); Diagnostics `12,600,320` B (`SHA-256: 7725A3ED0E7A5A6BB94394D50F24FE23669C7164C013932224B1A2346B8486EC`).
- **Wave 1 (`b65376c6`):**
  - `heap_pass.rs` (`VICTORY_HEAP_PASS_MARKED` default ON), `MIN_KB=64` & `CARD_HASH_LEN=1` (`shadow.rs`), `VICTORY_LOOKUP_CACHE` (`RwLock<BufferMap>` + `LOOSE`), `VICTORY_LIST_REUSE`, `VICTORY_FILLS_PLAIN`, `VICTORY_STEP_INDEX`, `VICTORY_CARD_SYNC_BATCH`, `xmem` PR `#1185`.
- **Wave 2 (`d2cf592a`):**
  - `#2301`, `#2449`, `#911`/`#1206` 64 KiB CBV cap, `4d0c6345` dynamic free-run wrap-around, `#1210` `ResourceMinLODClamp`, `Belobug 81ebb7bd` magenta filter, `#1241` `alias_source_changed` + `VICTORY_DECLARE_ONCE`, `VICTORY_TARGET_VARIANTS` + `VICTORY_BIND_CACHE`, PR `#3025` graphics-first queue, and diagnostic `W/A/S/D` + `I/J/K/L` sticks.
- **Live RX 6700 XT Run (`v15-diag`, `1,902` 3D samples over 3h20m):**
  - `1.90 FPS` median (`534.3 ms`, `p95 667.1 ms`, `max 3,248.0 ms`), `gpu 132.85 ms` (`p95 188.3`), `back 50.40 ms`, `WARN=306`, `heap_ms=399.05 ms` (`mirror_loop` `247–556 ms` serial identified for `v16`).

## `v14` (`v14-ready`, `v14-rx6700xt`) Changelog & Verification

- **Verified offline tests (`cargo test --workspace -- --test-threads=1`):** **839 passed, 0 failed, 23 ignored** (`victory-gpu --features diagnostics`: **300 passed, 0 failed, 12 ignored**; `cargo xtask policy`: **211 files ok**; max `.rs` size: `lower.rs` `501,859` B, `host.rs` `495,085` B <= `524,288` B).
- **Release build (Thin LTO, `codegen-units = 1`):** `9,407,488` B (`SHA-256: B01727D6CFACD7450BC531FE55D1E6837F60759A7E0419F472713AE2BA3FEDE2`).
- **Wave 1 (`cb645b22`):**
  - `shadow.rs`: default `VICTORY_CARD_BUFFERS_MB=2048` + `VICTORY_CARD_CHANGED_LIST=1` dirty card range list.
  - `host.rs`: `!cfg!(test)` defaults for `VICTORY_LAND_ASIDE=1` & `VICTORY_WIDE_REFRESH=1` (when logical CPUs >= 4).
  - `descriptor/host.rs` + `draw.rs`: `buffers_evict_gen` gating `ranges.retain`, pooled `SPANS` map, and `HEAP_PASS_BREAKDOWN[6]` instrumentation (`sync_heaps`, `retain`, `mirror_loop`, `spans`, `refresh_many`, `tex`).
  - `watch.rs` + `xmem.rs`: `!cfg!(test)` defaults for `VICTORY_ABSENT_ONCE=1` + `VICTORY_ABSENT_MS=u64::MAX` with `XMemVirtualAlloc` commit invalidation.
- **Wave 2 (`5e2e05de`, `f3578e38`, `97f1b391`):**
  - `Belobug` reflection + sky-dome ambient chain (`ambient.rs`, `pipeline.rs`, `draw.rs`, `host.rs`): default `VICTORY_AMBIENT=0.6`, `VICTORY_AMBIENT_REFLECT=0.5` clamping `CS_UpdateHistory`, `VICTORY_AMBIENT_SKY=0.36,0.05,0.4` following `ps_sky_paraboloid_reflection`, `list.keep(&shadow.buffer)` in `root_view`, `HostList::transition_copy`, `KEEP_TIME=8s` on retired host objects (`!cfg!(test)`), and non-fatal warning on undeletable old save directories (`xgamesave.rs`).
  - `BCPack`: keep all 32 indices in first BC7 dictionary entry (`5e2e05de`).
  - `Shaders`: `#2493` `v_add_f16` inline float constants in whole-wave sim (`f3578e38`), `v_dot8_i32_i4` (`0x018`) opcode/lowering/clamped tests, and `VICTORY_SHADER_OPT_TRACE=1`.
  - `Formats/XMA`: `#2663` 10-bit VLC LUT + `u64` `peek32` + cached `quant_factor` LUT.

## `v13` (`v13-ready`, `v13-rx6700xt`) Changelog & Verification

- **Verified offline tests (`cargo test --workspace -- --test-threads=1`):** **834 passed, 0 failed, 23 ignored** (`victory-gpu --features diagnostics`: **299 passed, 0 failed, 11 ignored**; `cargo xtask policy`: **211 files ok**; `lower.rs`: `501,859` B <= `524,288` B).
- **Release build (Thin LTO, `codegen-units = 1`):** `9,345,536` B (`SHA-256: 038660A9E033D354F2947ED3B233F1EF7F5B3460AA3ED506F3D36B42599100E5`).
- **Wave A (`72ffa502`..`3adfb785`):** `mv#114` XAsync cleanup unlocked, `mv#139` SSA sums memoization, `mv#244` serialized WARP host operations, placeholder & `Sum` clone fixes.
- **Wave B (`677969b6`..`a6a98e10`):**
  - `fx#1101` display-source driver watch filter, `fx#1108` `bcpack` scratch reuse, `fx#1135` parallel texture refresh accounting.
  - `2c0e7735` (`fx#1117`, `#1120`, `#1128`, `#1146`): cached `gpu_handle`, stack `user_data`, inplace `VIEWS` `RwLock`/sections, `land_aside`, tabled page tracking.
  - `b49825e1` (`mv#2828`, `triippy`, `mv#2727`, `Belobug`): concurrent cache publication, present barrier skip, rect range clamp, ray activation skip.
  - `02b0e485` & `f0d151e2` (`mv#2788`): host audio thread scheduling, non-blocking XMA mixer, ended-voice resume.
  - `a6a98e10`: `fx#1142` `BufferMap` interval map in `shadow.rs`, `fx#1130` `VICTORY_TEXTURE_PAGES=1` `detile_window` page-granular detile, `fx#1137` `VICTORY_FEED_ONE_CALL=1` batched upload submits, `fx#1136` `VICTORY_PRESENT_QUEUED` queued presents (opt-in for `late_stage() > 2`), `mv#2824` `release_title_allocation` cleanup.
- **Wave C1 (`b1049466`):** `mv#2808` fast DXBC MD5, `fx#1126` DXIL bitcode opt passes (`DEFAULT` preserving AMD `0x1002` `REUSE` skip), `fx#1140` `emit/lds.rs` Wave32 LDS disambiguation, `fx#1124` `emit/group_wide.rs` + `ends_trivially`, `fx#1132` env-switch fingerprint, `fx#1123` `StaticSampler::filter_for_host`, `fx#1112` `image_with_sampler` + `mv#2816` split scalar mask tracking.
- **Wave C2 (`81470672`):** `mv#2825` Native FP16 (`half` SM 6.2 via `set_host_native16`, `VICTORY_F16_IN_F32=1` fallback, verified on hardware GPU single-rounding FMA), `mv#2817` DPP8 quad ops (`emit/quad_tests.rs`), `ka1qux 033da400` `Atom::Opaque(Value, Value)`, `mv#2812` `StaticSampler::make_comparison`, `mv#2823` bit-31 `Lower::fneg`, `flamingo #751` dead-phi empty latch fold, `flamingo #747` `v_mul_hi_u32_u24`/`i24`.

## `v12` (`v12-ready`, `v12-rx6700xt`) Changelog & Verification

- **Verified offline tests (`cargo test --workspace -- --test-threads=1`):** **749 passed, 0 failed, 19 ignored** (`victory-gpu --features diagnostics`: **277 passed, 0 failed, 11 ignored**; `cargo xtask policy`: **205 files ok**; `lower.rs`: `504,741` B <= `524,288` B).
- **Release build (Thin LTO, `codegen-units = 1`):** `8,985,088` B (`SHA-256: D0266814F7085A7229FAC325A78FAE41616038FD8CCCA074807F489CE3164CEB`).
- **Wave A (`0297ada2`):** `VICTORY_PLACED_TEX=1` (`placed.rs`), `skip_tabled_buffers` + `entry.null_buf` (`descriptor/host.rs`), Series S two-plane compositor (`over.rs`, wired into `host.rs` & `queue.rs`), `mv#2723` `(u32, u32)` handle keys, and `fx#1096` `sampler_choice`.
- **Wave B (`28ec55d9`):** `#1069` `Section::committed` bitset + `VICTORY_REMAP_LEAN=1`, `#1102` `VICTORY_JOURNAL_UNLOCKED=1`, `#1086` re-entrant `REMAPPING` & `VICTORY_REMAP_BATCH=1`, `#1078` `VICTORY_TEXTURE_SHORT_HOLD=1`, `#1097` `PLACEHOLDERS` / `VICTORY_PLACEHOLDER_FAST=1` / `VICTORY_HOT_GUARD=1` / `bake_room()`, `#1094` `mask_runs`, and `#1104` `GetAsyncKeyState` low-bit tap + `VICTORY_KEYS_HOLD_MS`.
- **Wave C (`65ca7980`):**
  - `texture.rs`: `VICTORY_STAGING_BLOCKS=1` shared suballocation (`cut_from_blocks`, `staging_buffer_alone`).
  - `descriptor/host.rs`: `VICTORY_DYNAMIC_SLOTS=1` (`1 << 19` = 524,288 slots on uncapped heaps before `null`).
  - `resource.rs` (`#1080`): subtract `placed::free_bytes()` from `vm[1]` before `vram_idle_ms()`, add `VICTORY_TEXTURE_IDLE_MS`, raise normal `<80%` default to `15_000` ms outside `cfg!(test)`.
  - `memory.rs` (`#1082`): copy reclamation & `QUARANTINE = 8` (`VICTORY_MEMORY_RECLAIM`).
  - `victory-loader/src/prepare.rs` + `apps/victory/src/run.rs` (`#798`): policy-clean `PreparedImage::patch` + `VICTORY_TITLE_PATCH` (`KNOWN_BUILDS = &[]`).
  - `ambient.rs` + `pipeline.rs` + `draw.rs` (`#1093`): opt-in indirect-light pass (`VICTORY_AMBIENT`).
  - `ranges.rs` (`#1097`), plus `emit/lower/valu16.rs` extraction (`lower.rs` down to `504,741` B) & `fx#1096` / `mv#2723` shader tests.

## `v11` (`v11-ready`, `50b537b4`) Changelog & Verification

- **Verified offline tests (`cargo test --workspace -- --test-threads=1`):** **724 passed, 0 failed, 15 ignored** (`victory-gpu --features diagnostics`: **262 passed, 0 failed, 7 ignored**; `cargo xtask policy`: **199 files ok**; `lower.rs`: `518,390` B <= `524,288` B).
- **Release build (Thin LTO, `codegen-units = 1`):** `8,843,776` B (`SHA-256: 4E829DAFCC30F8D83E0E24DA8F4C93E20CD8E88A631B6C70DC3365CF6B9ED1C3`).
- **Wave A (`8a3f9518`):** Merged `fxceless/main` (`d510ee4a`), restored `!cfg!(test)` default `VICTORY_TEXTURE_DROP_MIPS=0`, enabled Thin LTO + `codegen-units = 1`, added mirror fast path.
- **Wave B (`425460c9`):** `#1043`, `#1068`, `#1032`, `#1058`, `#1059`, `#1066`, `#1044`, `#1063`, `#768`, shader-meta VSM2, `bind_pipeline` updates, plus `v10` session log fixes for `exact`-leak & `wide-textures`.
- **Wave C (`50b537b4`):**
  - `Belobug 86292110` log fix: disable blend enable on integer render targets in graphics & mesh PSOs (`crates/victory-gpu/src/pipeline/host.rs`).
  - Runtime & GPU PRs: `#1039` (crash-isolated shader compiler + worker respawn), `#959` audio (`audio.rs` XMA ring-buffer wrap & loop skip), `#1054` (descriptor slot log `Once` throttle), `#1055` (volume slice overlap check), `#1048` (`StageBudget` clamp), `#1035` (capture 512 MiB cap), `#989` (staging subresource padding bounds check).
  - Shader / 16-bit / precision PRs & upstream commits: `#927` (`emit/halves.rs` 16-bit half views), `#1036` (16-bit compare `opsel`), `#1040` (`v_div_fixup_f16`), `#1042` (packed int16 clamp saturation), `#957` (2D `image_get_resinfo` depth), `#951` (signed `s_add_i32`/`s_sub_i32` SCC), `#948` (VOP3 unsigned add/sub clamp), `#1041` (`reverse_postorder`), `Marvisak 4015a182` (mesh quad check), `a1a2077a` (`UAVS_EVERY_STAGE` outside pixel/compute), `da82e444` (`emit/norm.rs` exact `Unorm`/`Snorm` loads with `-1.0` `at_low` clamp).

## Status & Live Testing History (RX 6700 XT, Windows 11)

- **`v9` (`101f92e`, `backup-v9-base2`):** `2.3–2.9 FPS` (`346.8–435 ms/frame`), `prepare_heap_buffers` at `185–456 ms/frame`.
- **`v10a` / `v10b` (upstream `a253706` + `fx-pr-901..903` + `prepare_heap_buffers` `1/32` sweep & `HEAP_BUF_SKIP_TABLED`):** `6.0–7.5 FPS` (`INFLIGHT=1, INDIRECT_WINDOW=0`) vs `6.4–7.6 FPS` (`INDIRECT_WINDOW=1, INFLIGHT=0, DRIVER_WATCH=1`, GPU wait `4.1 ms` vs `10.7 ms`). Both exposed the linear `evict_placed_in` `remapping` stall (`140–240 ms/frame`) causing streaming OOM.
- **`v10c` (`f1b7088`, `PLACED_LIVE` `BTreeMap` + `sync_heaps` non-invalidation + `shadow::exact` prune throttle):** **`remapping` dropped to `0.6 ms/frame`**, **84+ minutes with 0 crashes / 0 driver errors**; heavy 4,686-draw cell at `1.7–2.1 FPS` (`553 ms`).
- **`v10d` (`148951c`, 64-striped `HostHeap` + `for_draw` `seen_pages` preservation + `EXACT_POOL` recycling):** Heavy 4,845-draw cell jumped to **`2.7–3.4 FPS`** (`328 ms`, `sync_heaps` halved to `171.5 ms`), `0` crashes.
- **`v10e` (upstream `3973179` + `0b0074e`, `uwop_epilog` v2 fix `12e6cb8`, baked-in defaults `425492e`):** Opening rooms **`7.6–9.2 FPS`** (`107–123 ms` usual, `57 ms` shortest), heavy 2,548–2,727 draw cutscene/cell **`3.0–3.6 FPS`** (`281–333 ms` usual, `sync_heaps 67.5–96.8 ms`).
- **`v10f` (`v10-final`, upstream `465b94f` `INDIRECT_GATHER` + bit-3 split fix `43feaf1` + `#1002/#1005/#1006/#1015/#1019/#1023/#1024/#1026/#1030/#1031`):** Opening rooms **`9.7–10.2 FPS`** (`51` frames in `5.0 s`, `98–103 ms` usual frame, `67 ms` shortest), heavy cell median `297 ms`, crisp daylight prison yard lighting and shadows, **0 crashes, 0 driver errors**.

### Curated PR Comparison (`fxceless/victory-fork`, `v10-final`)

- **Integrated:** `a253706`, `0b0074e` (`COPY_PLAIN` fix, `FAST_CORES`), `3973179` (`JOIN_PARTS`, `BACK_OWN_VIEW`, 64-thread coalesced `CARD_COMPARE`, `clzero`), `465b94f` (`INDIRECT_GATHER`, plus bit-3 vs split bit-2 bugfix `43feaf1`), `#901`, `#902`, `#903` (`PLACED_TEX`), `#1002` (Wave32 tiled cascade shadows + updated `site_pairs` test assertion), `#1005` (`xmem::unmap` ordering), `#1006` (code table start), `#1015` (`ALIAS_ONCE`), `#1019` (typed float sNaN), `#1023` (middle-mapped mips), `#1024` (cube `image_get_lod`), `#1026` (legacy f32 MAD/MAC fuse+flush), `#1030` (zero inactive compute `ds_swizzle_b32`), `#1031` (bounded texture queue + batched tile mappings + `show_resident`).
- **Intentionally skipped (with reason):**
  - `#997` / `#1014`: already superseded upstream by `3973179`'s 64-thread coalesced `CARD_COMPARE`.
  - `#1020` / `#1029`: Intel-only emulated `MONITORX`/`MWAITX` sleep (target CPU Ryzen 5 5600 executes them natively).
  - `#1000` / `#1001` / `#1016` / `#998`: superseded by upstream `465b94f` (`VICTORY_INDIRECT_GATHER`).
  - `#1018` (`TEXTURE_DIRTY_GEN`, stacked on unmerged `#231/#872`), `#1017` (>=1 TiB experimental alias), `#1013` (RT off by default), `#1027` (replaced by existing `VICTORY_KEYS_FILE`), `#1028` (Linux llvmpipe), and runtime `QUEUE2=3` / `VIEW_KNOWN` / `REFRESH_ONCE` (measured zero gain upstream `3973179` / `v10d`).

## Progression & Verification History (`02f404e` -> `v8`)

Host: Windows 11 x64 AVX2, AMD Radeon RX 6700 XT, `dxil.dll` `10.0.26100.0` (corpus: `26,599` containers / `26,148` programs).

| Build | VS/PS Validated (Emitted) | CS Validated (Emitted) | MS Validated (Emitted) | GS Validated (Emitted) | Total Primary Validated | Rejected by DXIL Validator |
|---|---|---|---|---|---|---|
| Upstream base `02f404e` (`#685`) | `12,160` (`12,236`) | `1,417` (`1,440`) | `8,101` (`8,163`) | `0` (`0`) | **21,678** | `161` |
| **v1** (`8e79c80` + 13 PRs) | `12,883` (`12,925`) | `1,451` (`1,474`) | `8,102` (`8,163`) | `0` (`0`) | **22,436** | – |
| **v2** (`ffe310e` + 19 PRs + `#736`) | `12,890` (`12,932`) | `1,451` (`1,475`) | `8,153` (`8,214`) | `0` (`0`) | **22,494** | `127` |
| **v3** (+16 items) | `12,977` (`13,020`) | `1,471` (`1,505`) | `8,319` (`8,380`) | `0` (`0`) | **22,767** | `138` |
| **v4** (+3 fixes + 17 items) | `13,248` (`13,292`) | `1,539` (`1,573`) | `8,374` (`8,455`) | `195` (`195`) + `195` ES | **23,356** | `159` |
| **v5** (+22 merges + `#981` + `f64`) | `13,769` (`13,769`) | `1,847` (`2,505`) | `8,599` (`8,670`) | `195` (`195`) + `195` ES | **24,410** | `729` (`658` CS + `71` MS) |
| **v6** (`backup-v6-final`, `a0b1b07`) | `13,425` (`13,425`) | `2,494` (`2,494`) | `8,405` (`8,476`) | `195` (`195`) + `195` ES | **24,519** | `71` (`0` CS + `71` MS) |
| **v7** (`backup-v7-final`, `7fccf82`) | `14,137` (`14,248`) | **`2,729`** (`2,729`) | **`8,660`** (`8,765`) | `–` (+195 SGPR) | **25,526** | `216` (`111` VP + `0` CS + `105` MS) |
| **v8 (current `v8-final`)** | **`14,129`** (`14,240`) | **`2,729`** (`2,729`) | **`8,660`** (`8,765`) | `–` (+195 SGPR) | **`25,518`** | **`216`** (`111` VP + `0` CS + `105` MS) |

- **v8 Verification Results:**
  - `cargo test --workspace -- --test-threads=1`: **598 passed, 0 failed, 13 ignored** (`victory` 1, `victory-core` 11, `victory-formats` 25 + 4 ign, `victory-game` 3, `victory-gpu` 179 + 3 ign, `victory-hle` 104, `victory-loader` 27, `victory-pe` 7, `victory-pipeline` 4, `victory-shader` 233 + 6 ign, `xtask` 4).
  - `cargo test -p victory-gpu --features diagnostics -- --test-threads=1`: **186 passed, 0 failed, 3 ignored**.
  - `cargo xtask policy`: **`asset policy: 180 files ok`** (`lower.rs` at `522,554` B <= `524,288` B without modifying `xtask/src/policy.rs`).
  - `cargo build --release -p victory`: **ok** (`39.02s`, `8,279,552` bytes).
  - `translate --validate` note: the `8`-shader delta (`25,526 -> 25,518`) comes strictly from **`fx-pr-801`** restoring `Err(FIRST_HALF)` on 8 unhandled paired interpolation first halves rather than silently compiling them as `0.0`. Compute validation remains **`2,729 / 2,729` (100% of emitted CS)**.

---

## Complete Chronological Change Inventory (`v8` on `v7`, then `v7`, then `v1..v6`)

### v8 — RX 6700 XT Default-Tuned Edition (`v8-final`, built on `backup-v7-final` `7fccf82`)

- **Wave 1 — GPU, Memory, HLE & Loading (`backup-v8-wave1` @ `b5ea581`):**
  - **fxceless PR #803** (`39f6994`): `own_bytes` buffer extent + 16-byte raw alignment fix (resolves raw `first=2` buffer view refusals).
  - **fxceless PR #834** (`2624049`): split indirect dispatches past `65,535` groups in `x`.
  - **fxceless PR #827** (`4068349`): world-entry table-slot skip crash fix, coalesced host refreshes, `VICTORY_RAM_RELIEF_MB`.
  - **fxceless PR #824** (`2e57463`): `RangeSet`/`holes` fast-path for view-memory guards, `xmem` alias coalescing, texture waits, relay refinements.
  - **fxceless PR #802** (`058156f`): sync raw-copied heap slots before draw passes + page-straddling descriptor records.
  - **fxceless PR #819** (`aa99a34`): return caller output pointer from 4 `GetDesc` methods.
  - **fxceless PR #838** (`02b6d7f`): stop preparing heap views after host device removal.
  - **fxceless PR #836** (`6434c18`): HLE accounting for DirectStorage decompression intermediate buffers.
  - **fxceless PR #818** (`5fc1129`): fix texture bake budget precision.
  - **myokex PR #6** (`f02e6d6`): per-thread pipeline wait while world loads.
  - **fxceless PR #805** (`1d9f67a`): resolved `CardInflight::brought` `Arc<Shadow>` leak fix.
  - **fxceless PR #807** (`e43910c`): resolved `HostHeap::copied` fingerprint invalidation across API writes, unmakes, and raw zeroing.
  - **fxceless PR #822** (`7f0354a`, adapted `relieve()` to `available_memory()`): trim idle upload caches when system commit runs low.
  - **fxceless PR #809** (`b5ea581`, ported to coalesced `xmem::map_alias`): skip redundant page commits via `PageCommitment::Existing`.
  - **fxceless PR #804** (`b5ea581`, ported to `RangeSet` `Guards`): exclude `absent` from `Guards::knows()` and verify `is_view()` on private reallocations.
  - **Local `v8` descriptor fixes (`b5ea581`):** invalidate `seen_pages` whenever `HostHeap::forget_copied()` clears a slot (`take_forgotten_copied()`), and guard `HostHeap::unmake()` for sampler heaps (`self.null_sampler(index)` instead of invalid `CreateShaderResourceView` call).
  - **Skipped Wave 1 PRs (logged with reasons):**
    - `fx-pr-812`: superseded by `fx-pr-824` `RangeSet`/`holes` O(log intervals) guard fast path.
    - `zslayer-pr-4`: deep texture staging buffer pool conflict with `fx-pr-822`/`824`/`827` memory trimming.
    - `zslayer-pr-9`: superseded by `fx-pr-827` incremental texture slot sweep.

- **Wave 2 — Shaders & Wave64 (`backup-v8-wave2` @ `8c7257d`):**
  - **fxceless PR #808** (`e71850d`): packed dot products and clamps (`emit/dot.rs`).
  - **fxceless PR #806** (`09237a6`): trace descriptor waterfalls across joins and per-lane loads (`resource.rs`).
  - **fxceless PR #801** (`9163a2e`, test adjusted in `emit/mod.rs` for upstream `Interp::Weighted` standalone `v_interp_p1_f32`): restore `Err(FIRST_HALF)` so `with_first_half_pairs` retries instead of silently returning `0.0`.
  - **fxceless PR #837** (`6389714`): NaN guard (`0` / `±65504`) on pixel color exports and typed image/buffer stores (`VICTORY_NAN_GUARD=0` to disable).
  - **fxceless PR #800** (`3ad7340`, resolved `valu_tests.rs`): lower guarded uniform first-lane reads on split waves.
  - **fxceless PR #832** (`8c7257d`, resolved `lower_tests.rs`): track divergent re-entry into shader loop exits.

- **XBVK Ports (`b32f177`, from [`0x0007b/XBVK`](https://github.com/0x0007b/XBVK) `858cd6a` / `b776914`):**
  - Fast-path detiling (`walk<const DETILE>` / `copy_runs`) was already present upstream in `crates/victory-formats/src/texture.rs`.
  - Ported `evict_placed_in` + `ranges_overlap` into `crates/victory-gpu/src/resource.rs` and wired into `watch::title_memory_remapped()` for immediate VRAM release on remaps/unmaps.

- **Custom `v8` GPU Fixes & Baked-In RX 6700 XT Defaults (`bf10bb1`):**
  - **Unaligned CBVs (`descriptor/host.rs`):** `buffer_copy` and `make(&View::Constant(..))` fall back to `shadow::exact(address, size)` when `shadow.gpu(address) % 256 != 0` (+ unit test `unaligned_constant_buffer_uses_exact_256_byte_copy`).
  - **Placed 2D/3D Texture UAVs (`resource.rs` + `descriptor/host.rs`):** Added `resource::unordered_texture_at` (`flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS`) and extended `placed_texture` to handle `View::Placed(true, ...)` -> `View::Unordered` (+ unit test `placed_unordered_textures_2d_and_3d_create_host_uavs`).
  - **Default RX 6700 XT Tuning (`!cfg!(test)`, all overrideable via env vars):** `VICTORY_RELAY=2`, `VICTORY_TEXTURE_DROP_MIPS=1`, `VICTORY_LEVELS_ON_DEMAND=1`, `VICTORY_CARD_ROOM_FOR_BUFFERS=1`, `VICTORY_CARD_REHASH_RUNS=1`, `VICTORY_CARD_FETCH_INFLIGHT=1`, `VICTORY_PAGE_WRITES=1` (`3` = guards + slots), and `VICTORY_ACTIVATE_RAYS=0`.

---

## Complete Chronological Change Inventory (`v7` Layer 1–3 on `fxceless 9376fd6`, then `v1..v6`)

### v7 — RX 6700 XT World-Rendering Edition (Rebased on [`fxceless/victory-fork`](https://github.com/fxceless/victory-fork) `9376fd6`)

- **Base `fxceless/victory-fork` (`9376fd6`) includes natively:**
  - World rendering architecture (`VICTORY_RELAY=2` flight relay + `VICTORY_CARD_FETCH_INFLIGHT`, `VICTORY_VIEW_GUARD` multi-view memory sync, `myokex` `c6dc4d8`, Odom `10ea38e`, DirectStorage `ERROR_INVALID_USER_BUFFER` fix `8dd30ab`, `UNCAPPED_VIEWS = 1 << 21` + `probe_view_limit()` `31714de`, DXIL clean-up optimization passes `-43%` instructions, prewarm pipeline library cache, descriptor copy mirror PR `#305`, levels-on-demand `#747` + heap sync `#670`).

- **Layer 1 — AMD RX 6700 XT Stability & Anti-OOM (`backup-v7-layer1`):**
  - `44ffef6` (`crates/victory-gpu/src/pipeline/make.rs`): Combined **myokex** (`victory-amd`) pipeline queue protections (`without_structures` skipping ray-tracing resolve shaders by default unless `VICTORY_RT_RESOLVE=1` or `VICTORY_BVH_BUILDER=1`; `too_dear` optional skip for the giant ray-activation compute shader via `VICTORY_ACTIVATE_RAYS=0`; `slow()` 3 s / 10 s timeout so no frame stalls on a single slow compile; separate graphics `4 s` / compute `12 s` loading wait budgets bounded by `VICTORY_PIPELINE_GATE_S=120`), **myokex #5** (skip known AMD compiler-crashing PSO in `Pending::new` unless `VICTORY_ALLOW_CRASHING_PSO=1`, plus `VICTORY_SKIP_PIPELINES` enabled in release builds), and **ZSlayerHQ #7** (`for_structures`: skip BVH skinning and BVH trace passes when `VICTORY_BVH_BUILDER != 1` and `VICTORY_BVH_PASSES != 1`, cutting idle RT GPU work).
  - `2ed17e2` (**ztrx #61**, `host.rs`, `emit/host.rs`, `emit/mod.rs`): Detect DXGI adapter vendor ID; log adapter name/vendor at startup and automatically omit `bitcode::opt::REUSE` (`4`) on AMD GPUs (`0x1002`) unless overridden via `VICTORY_SHADER_OPT`, avoiding `amdxc64.dll` access violations inside `CreateComputePipelineState`.
  - `b019fa1` (**myokex #4**, `list.rs`): Skip redundant self-`CopyResource` calls when `dst` and `src` resolve to the same host texture (`to == from`).

- **Layer 2 — Curated `fxceless` & `ka1qux` PRs (`backup-v7-layer2`):**
  - **fxceless PR #795** (`da2fecf`): On-demand texture levels keep their tiles while anything names the texture.
  - **fxceless PR #794** (`65ca183`): Preserve untouched bytes during partial texture read-back.
  - **fxceless PR #784** (`c295b0b`): Order deferred `UpdateTileMappings` per command list in multi-list submissions (prevents stale VRAM visibility between lists).
  - **fxceless PR #792** (`f3c3890`): Publish copied descriptors from the captured record, fixing races between capture and host slot mirroring.
  - **fxceless PR #783** (`a13d874`): Skip heap pages whose bytes did not change and write copies back in one call.
  - **fxceless PR #793** (`24b774a`): Skip diagnostic indirect budget atomics and readbacks outside `diagnostics` builds (reduces release CPU/GPU overhead).
  - **fxceless PR #610** (`479fc02`, resolved `list.rs` alongside `myokex #4` and `fx-pr-794`): `VICTORY_TEXTURE_DROP_MIPS=N` drops the largest `N` mip levels on host copies (~75% VRAM reduction for 2D textures at `N=1`).
  - **fxceless PR #188** (`38de813`): Series S stretch / display overlay path (`--console series-s`), avoiding stretched edge bleed.
  - **fxceless PR #518** (`fbd7c73`, resolved `host.rs` & `indirect.rs` alongside `fx-pr-793` `CARD_FETCH` + `warp_card_fetch` dispatch update): Parallelized `CARD_COMPARE` shader to one 64-thread group per 64 KiB piece instead of one thread per piece.
  - **ka1qux `99cec4e`** (`717e28d`, upstream PR `#632` backport): Preserve replacement view in `texture_slot_gone` when an older unmapped check finishes out of order.
  - **ka1qux `6391441`** (`02dc324`, upstream PR `#606` backport + `11b18db` 2-arg `staging()` test adaptation): Mark overlapping staged textures dirty in `title_memory_remapped` so remapped textures upload again.

- **Layer 3 — Native Wave64 & Shader PRs + Policy Compliance (`backup-v7-layer3`):**
  - **fxceless PR #497** (`0e9739b`): Wave-uniform first-lane reads (`half_sum_present` fallback) + masked shift counts when proving wave-uniform thread IDs.
  - **fxceless PR #558** (`a5c2987`, resolved keeping `half_sum_present` + both Wave64 tests, `424d9f8` cleanup): Barrier-free all-lane retry on native wave64 hosts (`HOST_MAX_WAVE = 64`), critical for RX 6700 XT native wave64 compute shaders.
  - **ZSlayerHQ #6** (`48229db`, cherry-picked `b05b60e`): Immediate sampler chosen by control flow samples through the taken branch's sampler.
  - **ztrx #68** (`4c8a415`, cherry-picked `81428dc`): Lower standalone `v_interp_p1_f32`.
  - **ztrx #69** (`007bb65`, cherry-picked `0cad4fa`, integrated alongside `unused_descriptor_kind`): Lower single-dword (`IMAGE_SIZE_DWORD = 2`) texture size scalar reads via `image_size_word` + `heap_index_at`.
  - `e9a2e8f`: Extracted `#[cfg(test)] mod tests` from `crates/victory-shader/src/emit/lower.rs` (`531,192` B > `512 KiB`) into `crates/victory-shader/src/emit/lower_tests.rs` (`21,002` B), bringing `lower.rs` down to `520,122` B cleanly within `xtask/src/policy.rs` (`MAX_FILE_SIZE = 524,288` B) without altering `policy.rs`.

- **Skipped / Deferred PRs in `v7` (with exact reasons):**
  - **fxceless PR #98:** Skipped — redundant with commit `31714de` already present in `fxceless/main` `9376fd6` (`UNCAPPED_VIEWS = 1 << 21` + dynamic `probe_view_limit()`).
  - **fxceless PR #220:** Explicitly skipped — caused device losses and GPU page faults in upstream `fxceless` testing (`VICTORY_INDIRECT_WINDOW=1`).
  - **myokex #2 / #3:** Skipped — redundant with `fxceless` commit `8dd30ab` already present in `9376fd6` (`DirectStorage` `ERROR_INVALID_USER_BUFFER` fix).
  - **ZSlayerHQ #5 (`5573b3f`):** Skipped — superseded by `fxceless/main` `9376fd6`'s more general `unused_descriptor_kind`.
  - **myokex `4436f8f`/`892d259` (>65,535 group dispatch split):** Deferred to `v8` — bundled inside a 13-file monolithic diff intertwined with texture spans, depth bounds, and `USER_GROUP_BASE` that will be cleanly ported on top of `v7` if needed.

- **RX 6700 XT Launch Switches (configured in Desktop launchers):**
  - Enabled by default: `VICTORY_RELAY=2`, `VICTORY_NO_WATCHDOG=1`, `VICTORY_LOG=info` (+ `--console series-s` on the Series S launcher).
  - Optional VRAM / stability toggles documented (commented in launchers): `VICTORY_TEXTURE_DROP_MIPS=1`, `VICTORY_LEVELS_ON_DEMAND=1`, `VICTORY_CARD_FETCH_INFLIGHT=1`, `VICTORY_ACTIVATE_RAYS=0`, `VICTORY_D3D12_BACKGROUND=0`, `VICTORY_SKIP_PIPELINES=...`.

---

### Base Upstream Commits Synced (`v1..v6` Historical Lineage, Preserved on `backup-v6-final` / `v6-final`)
- Starting point: **`02f404e`** (`PR #685`), followed by **`8e79c80`** (`#696`, `#706`, `#88`, `#93`, CI), then **`ffe310e`** (`PR #566` waterfall loops over first-lane table offsets, `61299b9` signature adaptations).

### v1 — Initial Optimized Stack (13 PRs on `8e79c80`)
- **Shaders (7):** [#539](https://github.com/Marvisak/VIctory/pull/539) (64-bit image atomics), [#592](https://github.com/Marvisak/VIctory/pull/592) (packed 16-bit pixel inputs 2nd rows), [#569](https://github.com/Marvisak/VIctory/pull/569) (biased image samples with 16-bit coords), [#694](https://github.com/Marvisak/VIctory/pull/694) (`image_sample_d_cl` / `image_gather4_l`), [#554](https://github.com/Marvisak/VIctory/pull/554) (one interpolation mode per signature row), [#571](https://github.com/Marvisak/VIctory/pull/571) (LDS byte/half ops), [#655](https://github.com/Marvisak/VIctory/pull/655) (guard conditional exports).
- **GPU & Build (6):** [#688](https://github.com/Marvisak/VIctory/pull/688) (reclaim dynamic descriptor slots), [#609](https://github.com/Marvisak/VIctory/pull/609) (defer retirement of GPU resources), [#652](https://github.com/Marvisak/VIctory/pull/652) (MSAA resolve mip extents), [#346](https://github.com/Marvisak/VIctory/pull/346) (expand indirect rectangle-list draws on GPU), [#472](https://github.com/Marvisak/VIctory/pull/472) (grow write snapshot in place & cap idle pool memory), [#677](https://github.com/Marvisak/VIctory/pull/677) (Thin LTO for release/diag builds).

### v2 — Rebase to `ffe310e` (`#566`) + 6 PRs + `#736` Fix
- Rebased/merged upstream `ffe310e` ([#566](https://github.com/Marvisak/VIctory/pull/566) waterfall loops + local `#736` fix `bfff38b`).
- **Merged (6):** [#588](https://github.com/Marvisak/VIctory/pull/588) (shadow buffer UAV <-> VB/IB/CBV transitions), [#628](https://github.com/Marvisak/VIctory/pull/628) (`ds_min_f32` / `ds_max_f32` RDNA2), [#617](https://github.com/Marvisak/VIctory/pull/617) (reuse lane guards), [#240](https://github.com/Marvisak/VIctory/pull/240) (DSP reverb convolution), [#536](https://github.com/Marvisak/VIctory/pull/536) (XMA ring wrap), [#620](https://github.com/Marvisak/VIctory/pull/620) (trigger rumble).

### v3 — Stability & Rendering Expansion (16 items)
- **GPU / Loader / Formats:** [#454](https://github.com/Marvisak/VIctory/pull/454) (`VICTORY_NO_WATCHDOG=1`), [#190](https://github.com/Marvisak/VIctory/pull/190) (indirect-arg DXIL clamp), [#493](https://github.com/Marvisak/VIctory/pull/493) (bounded root SRVs), [#568](https://github.com/Marvisak/VIctory/pull/568) (typed null descriptors), [#469](https://github.com/Marvisak/VIctory/pull/469) (depth UAV twin), [#211](https://github.com/Marvisak/VIctory/pull/211) (PresentX multi-planes), [#779](https://github.com/Marvisak/VIctory/issues/779) (GetDesc return pointer), [#782](https://github.com/Marvisak/VIctory/issues/782) (allocation-info alignment), [#24](https://github.com/Marvisak/VIctory/pull/24) (`ExecuteIndirect DispatchMeshX`), [#206](https://github.com/Marvisak/VIctory/pull/206) (unaligned structured views -> raw), [#492](https://github.com/Marvisak/VIctory/pull/492) (`R9G9B9E5_SHAREDEXP` -> `R16G16B16A16_FLOAT`), [#776](https://github.com/Marvisak/VIctory/issues/776) (write-watch once per submission), [#783](https://github.com/Marvisak/VIctory/issues/783) (`VirtualQuery` per-region fallback), [#230](https://github.com/Marvisak/VIctory/pull/230) (skip unchanged texture uploads).
- **Shaders:** [#775](https://github.com/Marvisak/VIctory/issues/775) (`v_swaprel_b32` / `v_movrelsd_2_b32`), [#728](https://github.com/Marvisak/VIctory/issues/728) (SMEM descriptor reads, spilled lane mask).

### v4 — Geometry, Stride Table & Session Fixes (3 session fixes + 17 items)
- **Session fixes:** [#844](https://github.com/Marvisak/VIctory/issues/844) (`resync_title_writes`), [#855](https://github.com/Marvisak/VIctory/issues/855) (`243d6d4` single shared returning block, `e67bb3d` omit array slices in `CalculateLOD`).
- **GPU & Memory:** [#876](https://github.com/Marvisak/VIctory/issues/876)/[#882](https://github.com/Marvisak/VIctory/issues/882) (aspect-aware typeless view remapping), [#651](https://github.com/Marvisak/VIctory/pull/651) (depth/stencil SRV aspect & `PlaneSlice`), [#450](https://github.com/Marvisak/VIctory/pull/450) (typed null RTVs), [#780](https://github.com/Marvisak/VIctory/issues/780)/#352 (`mesh triangles` library key & clean `RTVFormats`), [#796](https://github.com/Marvisak/VIctory/issues/796) (`OMSetRenderTargets` single-range guards), [#790](https://github.com/Marvisak/VIctory/issues/790) (`< 4 GiB` `xmem` 32-bit guard), [#464](https://github.com/Marvisak/VIctory/pull/464) (cache unmapped `absent` blocks), [#703](https://github.com/Marvisak/VIctory/pull/703) (`BCPack` `Workspace` reuse & LUTs).
- **Shaders:** [#585](https://github.com/Marvisak/VIctory/pull/585) (descriptor-only `GetDimensions`, null/const root descriptors), [#645](https://github.com/Marvisak/VIctory/pull/645) (root args spilled through VGPRs), [#489](https://github.com/Marvisak/VIctory/pull/489) (root spill table reads), [#704](https://github.com/Marvisak/VIctory/pull/704)+[#193](https://github.com/Marvisak/VIctory/pull/193) (runtime stride table for unknown-stride buffers), [#453](https://github.com/Marvisak/VIctory/pull/453) (`SV_Barycentrics`), [#878](https://github.com/Marvisak/VIctory/issues/878) (mask dwords & `s_bcnt1`), [#877](https://github.com/Marvisak/VIctory/issues/877) (legacy ES-GS geometry pipelines), [#852](https://github.com/Marvisak/VIctory/issues/852) (2-dword heap handle key), [#853](https://github.com/Marvisak/VIctory/issues/853) (waterfall dimension-only queries).

### v5 — Shader & GPU Expansion (22 merges + local test/adaptation fixes)
- **GPU / Indirect / Views:** `v5-921` (resolve placed texture UAVs to host resources), `v5-912` (zero-element buffer views as null descriptors), `v5-981` (align buffer view on host copy element), [#956](https://github.com/Marvisak/VIctory/pull/956) (clear overwritten guest root constants after `ExecuteIndirect`), `pr-67` (skip texture-to-buffer readback when texel size differs), [#1162](https://github.com/Marvisak/VIctory/pull/1162) (`ClearUnorderedAccessViewUint`/`Float` for buffers & textures), [#1126](https://github.com/Marvisak/VIctory/pull/1126) (release descriptor tables after rejected bind, keeping `Lease` reclamation).
- **Shaders / DXIL:** `fix/dxil-f64-conversion` (`v_cvt_f32_f64` & DXIL f64 flags), `v5-920` (`image_store` partial channel masks), `v5-914` (`SV_RenderTargetArrayIndex` / `SV_ViewportArrayIndex` initial exports), `v5-908` (helper lane `v_readfirstlane`), [#1130](https://github.com/Marvisak/VIctory/pull/1130) (descriptor read as data takes entry use kind), [#1056](https://github.com/Marvisak/VIctory/pull/1056) (scalar loads from root view address), [#1073](https://github.com/Marvisak/VIctory/pull/1073) (scalar joining lane mask and number), [#1074](https://github.com/Marvisak/VIctory/pull/1074) (lane mask spilled to VGPR lane), [#1012](https://github.com/Marvisak/VIctory/pull/1012) (descriptors joined by nested control flow), [#1022](https://github.com/Marvisak/VIctory/pull/1022) (validator signature row order), [#1134](https://github.com/Marvisak/VIctory/pull/1134) (`v_permlanex16_b32`), [#1176](https://github.com/Marvisak/VIctory/pull/1176) (uninitialized pixel SGPRs read zero), [#1008](https://github.com/Marvisak/VIctory/pull/1008) (immediate samplers chosen by control flow), [#978](https://github.com/Marvisak/VIctory/pull/978) (f64 immediate decode / double-precision checks), [#1182](https://github.com/Marvisak/VIctory/pull/1182) (prior per-lane descriptor word across redundant masked moves, issue #750).

### v6 — Ultimate Merge Stack (64 merges across Batch A, Batch B/B+, Batch B++, and Fast Sweep)
- **v6 Batch A — GPU 3D & Stability (10 merges):** [#1228](https://github.com/Marvisak/VIctory/pull/1228) (order shader writes on pipeline syncs, #732), [#1210](https://github.com/Marvisak/VIctory/pull/1210) (re-upload host textures after title write), [#1150](https://github.com/Marvisak/VIctory/pull/1150) (`OMSetDepthBounds`), [#1267](https://github.com/Marvisak/VIctory/pull/1267) (keep only writable shader byte ranges once/submission), [#1215](https://github.com/Marvisak/VIctory/pull/1215) (`NtWriteVirtualMemory` copy-back & `VICTORY_PROFILE`), [#1265](https://github.com/Marvisak/VIctory/pull/1265) (`VICTORY_WIDE_SCISSOR` diagnostic), [#1277](https://github.com/Marvisak/VIctory/pull/1277) (track view memory until remapped), [#1298](https://github.com/Marvisak/VIctory/pull/1298) (replace stale descriptors on null CBV writes), [#1311](https://github.com/Marvisak/VIctory/pull/1311) (normalized rect corners & NaN half conversion), [#1040](https://github.com/Marvisak/VIctory/pull/1040) (drop unsupported UAV/RT flags & guard RTV, adapted).
- **v6 Batch B + B+ — Shaders & VALU (27 active merges, `#1099` reverted):**
  - Control flow, descriptors, images & LDS: [#1230](https://github.com/Marvisak/VIctory/pull/1230) (path-sensitive missing phi reachability, #736), [#1237](https://github.com/Marvisak/VIctory/pull/1237) (compute implicit samples via quad gradients), [#1256](https://github.com/Marvisak/VIctory/pull/1256) (cube-array slice/face decode, #720), [#1253](https://github.com/Marvisak/VIctory/pull/1253) (scalar index arithmetic in waterfall offsets), [#1271](https://github.com/Marvisak/VIctory/pull/1271) (heap UAV inference & `glc` coherence), [#1231](https://github.com/Marvisak/VIctory/pull/1231) (uniform phis under `v_readfirstlane`), [#1245](https://github.com/Marvisak/VIctory/pull/1245) (biased 16-bit sample reads own bias VGPR), [#1148](https://github.com/Marvisak/VIctory/pull/1148) (`image_atomic_{inc,dec,fcmpswap,fmin,fmax}`), [#984](https://github.com/Marvisak/VIctory/pull/984) (split flat byte-offset structured loads), [#1145](https://github.com/Marvisak/VIctory/pull/1145) (duplicate latch folding & bitcode `replace_const` fix), [#1202](https://github.com/Marvisak/VIctory/pull/1202) (4 DXIL validator fixes + `SV_RenderTargetArrayIndex`/`SV_ViewportArrayIndex`), [#1276](https://github.com/Marvisak/VIctory/pull/1276) (first-lane scalar load from code table), [#1280](https://github.com/Marvisak/VIctory/pull/1280) (gradient samples with level clamp & face-space cube grads), [#1288](https://github.com/Marvisak/VIctory/pull/1288) (typed fetch aligned pair), [#1291](https://github.com/Marvisak/VIctory/pull/1291) (zero padding halves in f16 pixel interpolation), [#1293](https://github.com/Marvisak/VIctory/pull/1293) (`v_mov_b32` copies in interpolation pairing), [#1313](https://github.com/Marvisak/VIctory/pull/1313) (zero for unused image query extents).
  - VALU & Wave instructions: [#1223](https://github.com/Marvisak/VIctory/pull/1223) (`v_ldexp_f16`), [#1136](https://github.com/Marvisak/VIctory/pull/1136) (`v_pk_fmac_f16`), [#1149](https://github.com/Marvisak/VIctory/pull/1149) (`v_div_fixup_{f32,f16}`), [#1197](https://github.com/Marvisak/VIctory/pull/1197) (`v_sad_*` / `v_msad_u8`), [#1190](https://github.com/Marvisak/VIctory/pull/1190) (`v_frexp_{mant,exp}_*`), [#1285](https://github.com/Marvisak/VIctory/pull/1285) (16-bit VALU SDWA placement & inline float consts), [#1295](https://github.com/Marvisak/VIctory/pull/1295) (16-bit int inline float consts), [#1312](https://github.com/Marvisak/VIctory/pull/1312) (`v_readlane`/`v_writelane` lane masking & EXEC bypass), [#1292](https://github.com/Marvisak/VIctory/pull/1292) (`v_dot2`/`dot4`/`dot4c`/`dot8`), [#1300](https://github.com/Marvisak/VIctory/pull/1300) (`v_{min3,max3,med3}_{i16,u16}` & `v_mad_{u32_u16,i32_i16}`), [#1314](https://github.com/Marvisak/VIctory/pull/1314) (`v_{fmaak,fmamk}_f16`, `v_cvt_pk_*16_*32`, `v_lerp_u8`).
- **v6 Batch B++ — New Upstream Shaders Sub-group 1 (11 merges):** [#1409](https://github.com/Marvisak/VIctory/pull/1409) (heap-loaded `s_buffer_load` as raw SRV + compute `image_sample` LOD 0 fallback), [#1385](https://github.com/Marvisak/VIctory/pull/1385) (`GS_ALLOC_REQ` M0 bounds & mesh export constant guards), [#1489](https://github.com/Marvisak/VIctory/pull/1489) (`image_gather4_c_lz` / `_o`), [#1488](https://github.com/Marvisak/VIctory/pull/1488) (`ds_add{_rtn}_f32` & `ds_cmpst{_rtn}_f32`), [#1493](https://github.com/Marvisak/VIctory/pull/1493) (key immediate samplers by comparison mode), [#1473](https://github.com/Marvisak/VIctory/pull/1473) (per-lane pixel export state across mixed joins), [#1484](https://github.com/Marvisak/VIctory/pull/1484) (prune unreachable blocks before SSA), [#1474](https://github.com/Marvisak/VIctory/pull/1474) (`WAVE_OPS` flag on `waveGetLaneIndex`), [#1460](https://github.com/Marvisak/VIctory/pull/1460) (distinct critical-edge blocks for duplicate branch targets), [#1471](https://github.com/Marvisak/VIctory/pull/1471) (fold empty loop latches with dead phis), [#1501](https://github.com/Marvisak/VIctory/pull/1501) (DS ops without LDS read 0 / drop writes, extended for HEAD's byte/half and float LDS ops).
- **v6 Clean Auto-Merge Sweep (16 merges across B++ / A++):**
  - **Shaders:** [#1404](https://github.com/Marvisak/VIctory/pull/1404) (structure assert support set of vcc branch), [#1443](https://github.com/Marvisak/VIctory/pull/1443) (D3D12 scalar quadmask tests), [#1486](https://github.com/Marvisak/VIctory/pull/1486) (`Sym::Own` source lane tracking for shuffles), [#1328](https://github.com/Marvisak/VIctory/pull/1328) (bitwise `WaveMultiPrefixOp`), [#1366](https://github.com/Marvisak/VIctory/pull/1366) (skip VS inputs on mesh stage I/O), [#1395](https://github.com/Marvisak/VIctory/pull/1395) (`s_bfe_u64`/`s_bfe_i64`/`s_bfm_b64`).
  - **GPU / Formats / Pipeline:** [#1446](https://github.com/Marvisak/VIctory/pull/1446) (bind initial PSO in `create_list`/`create_list_x`), [#1392](https://github.com/Marvisak/VIctory/pull/1392) (UAV barrier after dropped compute dispatch), [#1390](https://github.com/Marvisak/VIctory/pull/1390) (keep `A8_UNORM` out of `R8` typeless family), [#1424](https://github.com/Marvisak/VIctory/pull/1424) (skip empty host submissions), [#1346](https://github.com/Marvisak/VIctory/pull/1346) (validate `allocate_upload` alignments & oversized requests), [#1324](https://github.com/Marvisak/VIctory/pull/1324) (blocked Windows cache publication test), [#1472](https://github.com/Marvisak/VIctory/pull/1472) (flush only overlapping shadow refreshes before `Step::GpuWrite`), [#1463](https://github.com/Marvisak/VIctory/pull/1463) (`VICTORY_SUBMIT_MAX_RUNS` cap), [#1505](https://github.com/Marvisak/VIctory/pull/1505) (pipeline workers at title priority), [#1499](https://github.com/Marvisak/VIctory/pull/1499) (keep deep 3D mips out of thick mip tail, `VIC-BT-014`).

## Local Fixes & Test Adaptations (`v6` branch `backup-v6-final`)
- `ca9ea87`: Allowed `CubeArray` (`kind == 9`) gathers past `pr-1489`'s `2 | 5 | 7` guard (fixing `cube_array_gather_and_resinfo_match_rdna2`), and split `crates/victory-shader/src/emit/lower.rs` (`619,692` B > `512 KiB` policy limit) cleanly into `lower.rs` (`516,776` B) + `lower_mesh.rs` + `lower_tests.rs` so `cargo xtask policy` passes untouched.
- `f472bff`: Removed duplicate `sopc` helper in `valu_tests.rs` after `#1443` / `#1395` sweep.
- `ef54a9e`: Updated `heap_uav_inference_and_glc_coherence_set_handle_properties` expectation for `s_buffer_load_dword` from `ResourceHeap` (`(13, 0x1_0000)` -> `(0x000B, 0)` per `#1409`).
- `aa719f1` / `77bcae5`: Skipped non-WARP adapters in synthetic `image_gather4_c_lz` / `immediate_sampler` `View::Texture` (`RGBA32F`) tests when `D3D12_FEATURE_FORMAT_SUPPORT` lacks `SHADER_GATHER_COMPARISON` / `SHADER_SAMPLE_COMPARISON` (AMD RDNA2 strictly enforces D3D12 single-component format support on comparison sampling).
- `92f1773` / `8681d1f` / `177764b` / `203ad44`: Reconciled Batch A unit tests (`placed_uav_tests`, `shadow::tests`, `texture::tests`, `host_texture` feature queries) and `rectangles` `Snorm8`/`Snorm16` support with `#1040`/`#1210`/`#1265`.
- `92300ad` / `4cf3636` / `7f37f05` / `09b99b5`: Adapted Validator 1.6 signature sort test (`container.rs`), `half_int_src` usage in `#1300`, `HostShader` `table_views`/`stride_table` test literals, and `#1276` `first_lane` code-table test assertions.

## Reverted & Skipped PRs (`v1..v6` with exact reasons)
- **v2:** [#459](https://github.com/Marvisak/VIctory/pull/459), [#334](https://github.com/Marvisak/VIctory/pull/334), [#582](https://github.com/Marvisak/VIctory/pull/582), [#159](https://github.com/Marvisak/VIctory/pull/159), [#587](https://github.com/Marvisak/VIctory/pull/587) (merge conflicts).
- **v3:** [#436](https://github.com/Marvisak/VIctory/pull/436), [#515](https://github.com/Marvisak/VIctory/pull/515), [#393](https://github.com/Marvisak/VIctory/pull/393) (merge conflicts).
- **v4:** [#785](https://github.com/Marvisak/VIctory/issues/785) (needs skipped `#334`), [#757](https://github.com/Marvisak/VIctory/issues/757) (conflicts `#346`/`#190`/`#24`), [#699](https://github.com/Marvisak/VIctory/pull/699) (superseded by scoped `#736`), [#710](https://github.com/Marvisak/VIctory/pull/710) (deep conflicts across 8 files).
- **v6 Batch A/B/B+:**
  - `#1260`: superseded by HEAD `v5-981` (`shadow::aligned`).
  - `#1099` (merged `ec245ff`, reverted `a00313b`): broke soundness test `first_half_of_interpolation_stays_missing_when_live_and_lowers_when_overwritten`; superseded by `#1230` `BoolSet`.
  - `#1270` (aborted): failed `!l.wave_ops` (`selects_on_masks_the_same_in_every_lane_need_no_lane_index`) and WARP `WaveSize(64)`.
  - `#1310`: shader diff already 100% present in HEAD (generalized by `#1253`).
- **v6 Batch B++ / A++ / Batch C Skipped (62 PRs total across manual/dedup/sweep):**
  - Dedup / already covered (7): `#1417` (superseded by `#1202` + 9 `lower.rs` conflicts), `#1451` (overlaps `#1293` across 4 files), `#1480` (covered by `#1245`+`#1280`), `#1467` (`HandleKey` in HEAD), `#1477` (chose `#1493`), `#1400` (chose cleaner `#1414`), `#1494` (chose safer `VIC-BT-014` `#1499`), `#1354`, `#1336`.
  - Fast-track auto-sweep merge conflicts (50 PRs in B++/A++): `#1407`, `#1420`, `#1412`, `#1416`, `#1423`, `#1408`, `#1491`, `#1485`, `#1441`, `#1437`, `#1478`, `#1469`, `#1342`, `#1341`, `#1340`, `#1339`, `#1338`, `#1337`, `#1334`, `#1333`, `#1327`, `#1326`, `#1370`, `#1391`, `#1398`, `#1387`, `#1371`, `#1433` (Wave32-only multi-wave flow; RX 6700 XT runs wave64 natively), `#1448`, `#1444`, `#1428` (partial overlap `#796`), `#1427` (partial overlap `#568`), `#1422`, `#1406` (partial overlap `#876`), `#1414`, `#1388`, `#1374`, `#1352`, `#1349` (partial overlap `#688`/`#1126`), `#1322`, `#1453`, `#1351`/`#1350`/`#1332` (overlap `#776`/`#783`), `#1466`, `#1461`, `#1510`, `#1507`, `#1503`, `#1500`.
  - Batch C large architectural PRs skipped due to conflicts (4): `#1315`, `#1399` (merged ES+GS; HEAD already has `#877` `pr-1202` ES-GS), `#1198` (`+2335` lines address map/descriptors), `#1175` (spilled root UAV writes).

## Credits & Acknowledgments
- **[Marvisak/VIctory](https://github.com/Marvisak/VIctory)** — upstream project (`GPL-3.0-or-later`).
- **[fxceless/victory-fork](https://github.com/fxceless/victory-fork)** — base world-rendering fork (`9376fd6`) and PRs `#188`, `#497`, `#518`, `#558`, `#610`, `#783`, `#784`, `#792`, `#793`, `#794`, `#795`.
- **[myokex/victory-amd](https://github.com/myokex/victory-amd)** — AMD RX 6000/7000 pipeline skips, timeouts, and PRs `#4` / `#5`.
- **[ztrx](https://github.com/ztrx)** — AMD shader compiler `REUSE` bypass (`#61`) and shader lowering PRs `#68` / `#69`.
- **[ZSlayerHQ](https://github.com/ZSlayerHQ)** — BVH skinning/trace bypass (`#7`) and immediate sampler control-flow PR (`#6`).
- **[ka1qux](https://github.com/ka1qux)** — descriptor slot write-number race fix (`99cec4e` / `#632`) and remapped staged texture dirtying (`6391441` / `#606`).
