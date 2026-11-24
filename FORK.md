# Fork: ultimate-rx6700xt (AI-assisted)

Personal fork of [Marvisak/VIctory](https://github.com/Marvisak/VIctory). Branch `ultimate-rx6700xt` (mirrored to `main`) stacks community PRs and contributor fork fixes on top of upstream `main` (`ffe310e`) with merge conflicts resolved for Windows 11 + AMD Radeon RX 6700 XT (RDNA2, 12 GB VRAM). Nothing here is upstream-approved; please send reviews and feedback to the original PRs/issues. Zero game data or assets are included (`cargo xtask policy` verified).

## Status & Live Testing History (RX 6700 XT, Windows 11)

- **v2 live test:** Boot, settings, audio, HUD, subtitles, and opening sequences rendered; distant emissive lights appeared in-world (~1.0–1.4 FPS), then ended with `DXGI_ERROR_DEVICE_HUNG`.
- **v3 + 3 session fixes (`e67bb3d`) live test:** Held cleanly in-world with **0 `ERROR` lines** in logs, `1,892` host pipelines compiled, GPU ~74–94 ms/frame (~0.6–1.0 FPS overall, CPU prep/readback ~1,000 ms/frame). Sky tone, bloom/glare, subtitles, audio, and controller camera input worked; remaining 3D geometry still incomplete (`236` live shader refusals, typeless depth/view mapping gaps, and high readback overhead). Closed cleanly by user without GPU hang/reset.
- **v4 (current HEAD):** Built, unit-tested, policy-checked, and offline shader-validated (**23,356** primary shaders + **195** ES halves). **Not yet tested live in-game.**

## What's New in v4

### Intermediate Session Fixes (on top of v3)
| Source | Ref / Link | Description |
|---|---|---|
| `Slaktarn700` | [issue #844](https://github.com/Marvisak/VIctory/issues/844) (`c4f0f19`) | Mirror title-written descriptor heap slots before submission (`resync_title_writes`) |
| `Slaktarn700` | [issue #855](https://github.com/Marvisak/VIctory/issues/855) (`243d6d4`) | Emit a single shared returning block per shader program |
| `Chuantao717` | [issue #855](https://github.com/Marvisak/VIctory/issues/855) (`e67bb3d`) | Omit array slices from DXIL `CalculateLOD` queries |

### Group 1 — GPU Runtime, Render Targets & Memory (v4)
| # | Link | Description |
|---|---|---|
| 876 / 882 | [fork Xvania7](https://github.com/Xvania7/VIctory/tree/fix/issue-850-typeless-view) ([issue #876](https://github.com/Marvisak/VIctory/issues/876), [issue #882](https://github.com/Marvisak/VIctory/issues/882)) | Remap typeless views on typeless resources to aspect-aware concrete formats (incl. `53 => 56`, keeping `#492`) |
| 651 | [PR #651](https://github.com/Marvisak/VIctory/pull/651) | Depth/stencil SRV aspect selection and `PlaneSlice` propagation |
| 450 | [PR #450](https://github.com/Marvisak/VIctory/pull/450) | Typed null RTV descriptors (`capture` test gated on `diagnostics`) |
| 780 / 352 | [fork LAKSHYA-cmd-tech](https://github.com/LAKSHYA-cmd-tech/VIctory/tree/feature/pipeline-library-mesh-topology-and-clean-desc) ([issue #780](https://github.com/Marvisak/VIctory/issues/780)) | `"mesh triangles"` pipeline library key with topology + clean `RTVFormats` |
| 796 | [fork LAKSHYA-cmd-tech](https://github.com/LAKSHYA-cmd-tech/VIctory/tree/feature/om-set-render-targets-bounds) ([issue #796](https://github.com/Marvisak/VIctory/issues/796)) | Guard empty, null, and overflowing `OMSetRenderTargets` single-range bindings |
| 790 | [fork LAKSHYA-cmd-tech](https://github.com/LAKSHYA-cmd-tech/VIctory/tree/feature/xmem-guard-32bit-window) ([issue #790](https://github.com/Marvisak/VIctory/issues/790)) | Protect `< 4 GiB` low address ranges for title 32-bit `xmem` allocations |

### Group 2 — Shader Compiler & Live Refusal Fixes (v4)
| # | Link | Description |
|---|---|---|
| 585 | [PR #585](https://github.com/Marvisak/VIctory/pull/585) | Lower descriptor-only `GetDimensions`, plus `NULL` and constant root descriptors |
| 645 | [PR #645](https://github.com/Marvisak/VIctory/pull/645) | Trace SGPR root-argument addresses spilled through VGPRs (`v_readlane`) |
| 489 | [PR #489](https://github.com/Marvisak/VIctory/pull/489) | Trace root spill table reads in resource tracking |
| 704 / 193 | [PR #704](https://github.com/Marvisak/VIctory/pull/704), [PR #193](https://github.com/Marvisak/VIctory/pull/193) | Runtime stride table + raw views for structured buffers with unknown stride |
| 453 | [PR #453](https://github.com/Marvisak/VIctory/pull/453) | Lower pixel `SV_Barycentrics` values and computed barycentric interpolation |
| 878 | [fork permadiego](https://github.com/permadiego/VIctory/tree/shader/mask-dword-joins) ([issue #878](https://github.com/Marvisak/VIctory/issues/878)) | Mask dwords at data joins (`undef`) and per-lane `s_bcnt1` SCC tracking |
| 877 | [fork permadiego](https://github.com/permadiego/VIctory/tree/shader/geometry-shaders) ([issue #877](https://github.com/Marvisak/VIctory/issues/877)) | Translate legacy ES-GS geometry programs to DXIL + build VS+GS+PS host pipelines |
| 852 | [fork NicolaiVdS](https://github.com/NicolaiVdS/VIctory/tree/shader/atomic-handle-type) ([issue #852](https://github.com/Marvisak/VIctory/issues/852)) | Key heap handle cache on both property dwords so `u32` image atomics don't reuse `f32` query handles |
| 853 | [fork FabianDU](https://github.com/FabianDU/VIctory/tree/claude/elegant-newton-vly8mr) ([issue #853](https://github.com/Marvisak/VIctory/issues/853)) | Track waterfall loops that only query per-lane descriptor dimensions |

### Group 3 — Performance & CPU Overhead (v4)
| # | Link | Description |
|---|---|---|
| 703 | [PR #703](https://github.com/Marvisak/VIctory/pull/703) | Eliminate BCPack per-call allocations (`Workspace`) + streaming `lanes_stream` + LUTs (combined with `bucket_of` window table) |
| 464 | [PR #464](https://github.com/Marvisak/VIctory/pull/464) | Remember unmapped guarded view blocks (`absent`) until `title_memory_remapped`, preserving `#776` overflow guards |

## Skipped PRs (with reasons)
- **v4:** [#785](https://github.com/Marvisak/VIctory/issues/785) (depends on skipped `#334` / `BUFFER_READ`), [#757](https://github.com/Marvisak/VIctory/issues/757) (deep conflicts with `#346`/`#190`/`#24`), [#699](https://github.com/Marvisak/VIctory/pull/699) (superseded by scoped `#736` already in HEAD), [#710](https://github.com/Marvisak/VIctory/pull/710) (deep conflicts across 8 files with `#645`/`#493`/`#568`/`#728`).
- **v3:** [#436](https://github.com/Marvisak/VIctory/pull/436), [#515](https://github.com/Marvisak/VIctory/pull/515), [#393](https://github.com/Marvisak/VIctory/pull/393).
- **v2:** [#459](https://github.com/Marvisak/VIctory/pull/459), [#334](https://github.com/Marvisak/VIctory/pull/334), [#582](https://github.com/Marvisak/VIctory/pull/582), [#159](https://github.com/Marvisak/VIctory/pull/159), [#587](https://github.com/Marvisak/VIctory/pull/587).

## Already Integrated from v1 / v2 / v3

### Added in v3 (16 items)
| # | Link | Description |
|---|---|---|
| 454 | [PR #454](https://github.com/Marvisak/VIctory/pull/454) | loader watchdog opt-out (`VICTORY_NO_WATCHDOG=1`) |
| 190 | [PR #190](https://github.com/Marvisak/VIctory/pull/190) | indirect-argument clamp in DXIL |
| 493 | [PR #493](https://github.com/Marvisak/VIctory/pull/493) | bounded root SRVs |
| 568 | [PR #568](https://github.com/Marvisak/VIctory/pull/568) | typed null descriptors |
| 469 | [PR #469](https://github.com/Marvisak/VIctory/pull/469) | UAV twin for depth resources |
| 211 | [PR #211](https://github.com/Marvisak/VIctory/pull/211) | PresentX multi-plans |
| 779 | [fork BlueCatsz](https://github.com/BlueCatsz/VIctory/tree/fix/gpu-getdesc-return-ptr) ([issue #779](https://github.com/Marvisak/VIctory/issues/779)) | GetDesc return pointer |
| 782 | [fork AndyLatimer](https://github.com/AndyLatimer/VIctory/tree/fix/gpu-allocation-info-alignment) ([issue #782](https://github.com/Marvisak/VIctory/issues/782)) | allocation-info alignment |
| 24 | [PR #24](https://github.com/Marvisak/VIctory/pull/24) | ExecuteIndirect DispatchMeshX |
| 206 | [PR #206](https://github.com/Marvisak/VIctory/pull/206) | unaligned structured views -> raw view |
| 492 | [PR #492](https://github.com/Marvisak/VIctory/pull/492) | `R9G9B9E5_SHAREDEXP` -> `R16G16B16A16_FLOAT` |
| 776 | [fork schmiticle](https://github.com/schmiticle/VIctory/tree/gpu/write-watch-once-per-submission) ([issue #776](https://github.com/Marvisak/VIctory/issues/776)) | write-watch once per submission |
| 783 | [gist rocobalt1](https://gist.github.com/rocobalt1/1edb3c0dad03f54c05f972b95475b14c) ([issue #783](https://github.com/Marvisak/VIctory/issues/783)) | per-region VirtualQuery fallback |
| 230 | [PR #230](https://github.com/Marvisak/VIctory/pull/230) | skip clean texture uploads |
| 775 | [fork talakasse](https://github.com/talakasse/VIctory/tree/shader/swaprel) ([issue #775](https://github.com/Marvisak/VIctory/issues/775)) | shader: swaprel |
| 728 | [fork Wasler](https://github.com/Wasler/VIctory/tree/fix/shader-top5) ([issue #728](https://github.com/Marvisak/VIctory/issues/728)) | shader: SMEM reads, spilled lane mask |

### Base v1 / v2
- Shaders: [#539](https://github.com/Marvisak/VIctory/pull/539) [#592](https://github.com/Marvisak/VIctory/pull/592) [#569](https://github.com/Marvisak/VIctory/pull/569) [#694](https://github.com/Marvisak/VIctory/pull/694) [#554](https://github.com/Marvisak/VIctory/pull/554) [#571](https://github.com/Marvisak/VIctory/pull/571) [#655](https://github.com/Marvisak/VIctory/pull/655) [#628](https://github.com/Marvisak/VIctory/pull/628) [#617](https://github.com/Marvisak/VIctory/pull/617)
- GPU / build: [#688](https://github.com/Marvisak/VIctory/pull/688) [#609](https://github.com/Marvisak/VIctory/pull/609) [#652](https://github.com/Marvisak/VIctory/pull/652) [#346](https://github.com/Marvisak/VIctory/pull/346) [#472](https://github.com/Marvisak/VIctory/pull/472) [#677](https://github.com/Marvisak/VIctory/pull/677) [#588](https://github.com/Marvisak/VIctory/pull/588)
- Audio / input: [#240](https://github.com/Marvisak/VIctory/pull/240) [#536](https://github.com/Marvisak/VIctory/pull/536) [#620](https://github.com/Marvisak/VIctory/pull/620)
- Upstream base: [#566](https://github.com/Marvisak/VIctory/pull/566) (`ffe310e`) + scoped `#736` fix

## Verification (v4 — Offline)

Host: Windows 11, x86_64 AVX2, AMD Radeon RX 6700 XT, `dxil.dll` `10.0.26100.0`.

- `cargo test --workspace -- --test-threads=1`: **447 passed, 0 failed, 6 ignored** (`victory` 1, `victory-core` 11, `victory-formats` 26 + 4 ignored, `victory-game` 2, `victory-gpu` 93 + 1 ignored, `victory-hle` 105 + 1 integration, `victory-loader` 22 + 3 integration, `victory-pe` 7, `victory-pipeline` 4, `victory-shader` 168 + 1 ignored, `xtask` 4).
- `cargo test -p victory-gpu --features diagnostics -- --test-threads=1`: **97 passed, 0 failed, 1 ignored**.
- `cargo xtask policy`: **`asset policy: 162 files ok`**.
- `cargo build --release -p victory` (Thin LTO): **ok** (`54.22 s`).
- `translate --validate` across all shipped title shader containers (`26,599` containers / `26,148` programs):

| Build | VS/PS Validated (Emitted) | CS Validated (Emitted) | MS Validated (Emitted) | GS Validated (Emitted) | Total Primary Validated | Rejected by DXIL Validator |
|---|---|---|---|---|---|---|
| Upstream `02f404e` | – | – | – | – | 21,678 | – |
| v2 (`ultimate-rx6700xt-v2`) | 12,890 (12,932) | 1,451 (1,475) | 8,153 (8,214) | 0 (0) | 22,494 | 127 |
| v3 | 12,977 (13,020) | 1,471 (1,505) | 8,319 (8,380) | 0 (0) | 22,767 | 138 |
| **v4 (current)** | **13,248** (13,292) | **1,539** (1,573) | **8,374** (8,455) | **195** (195) + **195** ES halves | **23,356** | **159** |

- Delta v3 → v4: **+589** primary validated shaders (plus **195/195** ES halves); **+1,678** over upstream `02f404e`.
- Not yet tested in-game.
