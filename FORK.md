# Fork: ultimate-rx6700xt (AI-assisted)

Personal fork of [Marvisak/VIctory](https://github.com/Marvisak/VIctory). Branch `ultimate-rx6700xt` stacks community PRs/forks on top of upstream `main` (ffe310e) with conflicts resolved. Nothing here is upstream-approved; please use the original PRs/issues for review.

## How far it goes (tested by the owner, RX 6700 XT, Windows 11)
- v2 live test: menus, settings, audio, HUD/minimap, subtitles and the opening credits render; first distant emissive lights appear. About 1 FPS in-world. Ended with a GPU device hang (DXGI_ERROR_DEVICE_HUNG).
- v3 (this branch): built and validated offline only, **not yet tested in game**. Results will be added after testing.

## Contents

Windows 11, x86_64 AVX2, AMD Radeon RX 6700 XT, dxil.dll 10.0.26100.0. Local branch built on `main` ffe310e (includes #566). No game data in this comment (asset policy).

### New in v3 (16 items)
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
| 492 | [PR #492](https://github.com/Marvisak/VIctory/pull/492) | R9G9B9E5_SHAREDEXP -> R16G16B16A16_FLOAT |
| 776 | [fork schmiticle](https://github.com/schmiticle/VIctory/tree/gpu/write-watch-once-per-submission) ([issue #776](https://github.com/Marvisak/VIctory/issues/776)) | write-watch once per submission |
| 783 | [gist rocobalt1](https://gist.github.com/rocobalt1/1edb3c0dad03f54c05f972b95475b14c) ([issue #783](https://github.com/Marvisak/VIctory/issues/783)) | per-region VirtualQuery fallback |
| 230 | [PR #230](https://github.com/Marvisak/VIctory/pull/230) | skip clean texture uploads |
| 775 | [fork talakasse](https://github.com/talakasse/VIctory/tree/shader/swaprel) ([issue #775](https://github.com/Marvisak/VIctory/issues/775)) | shader: swaprel |
| 728 | [fork Wasler](https://github.com/Wasler/VIctory/tree/fix/shader-top5) ([issue #728](https://github.com/Marvisak/VIctory/issues/728)) | shader: SMEM reads, spilled lane mask |

Skipped (conflicts): [#436](https://github.com/Marvisak/VIctory/pull/436), [#515](https://github.com/Marvisak/VIctory/pull/515), [#393](https://github.com/Marvisak/VIctory/pull/393) (v3) ; #459, #334, #582, #159, #587 (v2).

### Already on the branch from v1/v2
Shaders: [#539](https://github.com/Marvisak/VIctory/pull/539) [#592](https://github.com/Marvisak/VIctory/pull/592) [#569](https://github.com/Marvisak/VIctory/pull/569) [#694](https://github.com/Marvisak/VIctory/pull/694) [#554](https://github.com/Marvisak/VIctory/pull/554) [#571](https://github.com/Marvisak/VIctory/pull/571) [#655](https://github.com/Marvisak/VIctory/pull/655) [#628](https://github.com/Marvisak/VIctory/pull/628) [#617](https://github.com/Marvisak/VIctory/pull/617) ; GPU/build: [#688](https://github.com/Marvisak/VIctory/pull/688) [#609](https://github.com/Marvisak/VIctory/pull/609) [#652](https://github.com/Marvisak/VIctory/pull/652) [#346](https://github.com/Marvisak/VIctory/pull/346) [#472](https://github.com/Marvisak/VIctory/pull/472) [#677](https://github.com/Marvisak/VIctory/pull/677) [#588](https://github.com/Marvisak/VIctory/pull/588) ; audio/input : [#240](https://github.com/Marvisak/VIctory/pull/240) [#536](https://github.com/Marvisak/VIctory/pull/536) [#620](https://github.com/Marvisak/VIctory/pull/620) ; base [#566](https://github.com/Marvisak/VIctory/pull/566) ; fix #736.

### Verification
- `cargo test --workspace -- --test-threads=1` : 0 failures. `victory-gpu --features diagnostics` : 86 passed, 0 failures, 1 ignored.
- `cargo xtask policy` : 160 files ok. Release Thin LTO build: ok.
- `translate --validate` : **22 767 validated** (VS/PS 13 020 emitted / 12 977 ; CS 1 505 / 1 471 ; MS 8 380 / 8 319). v2 = 22 494 (+273) ; base 02f404e = 21 678 (+1 089).
- Emitted but rejected by the validator : 138 (v2 : 127). Not analysed per shader.
- Not tested in game.
