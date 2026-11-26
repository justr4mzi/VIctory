# VIctory v16 (RX 6700 XT Fork)

Fork of [avatarmanz/VIctory-7900XTX](https://github.com/avatarmanz/VIctory-7900XTX) (originally forked from [Marvisak/VIctory](https://github.com/Marvisak/VIctory)), tuned for **Windows 11 + AMD Ryzen 5 5600 (6C/12T) + AMD Radeon RX 6700 XT (12 GB VRAM, RDNA2) + 48 GB RAM**.

**This repository contains no game files, code or assets.** You need your own copy of the title. It is not affiliated with or endorsed by any developer or publisher.

See [FORK.md](FORK.md) for the complete changelog, multi-fork integrations, offline test results, and live RX 6700 XT status.

## Status on RX 6700 XT (RDNA2, 2026-11-26)

- **Build & Offline Tests (`v16-beta`):**
  - `cargo xtask policy`: **passed (`251 files ok`)**.
  - `[profile.ship]` (fat LTO, `codegen-units = 1`) and `[profile.diag]` (`diagnostics` feature) build cleanly.
  - Non-`victory-gpu` workspace tests: **604 passed, 0 failed, 11 ignored**.
  - `victory-gpu` (`diagnostics`): **388 lib + 1 replay passed, 1 failed (`descriptor::host::tests::views_are_made_beside_the_heap_and_copied_in`, pre-existing upstream RDNA2 CPU-vs-SV descriptor byte assertion), 35 ignored**.
- **Live GPU Status on RX 6700 XT:**
  - Enters the 3D scene cleanly (`WORLD=2`, ~10x faster heap pass `3.3–73.9 ms` vs `v15`'s `399 ms`, crisp lighting), but hits `DXGI_ERROR_DEVICE_RESET` (`0x887A0007`) within 0–12 s of entering 3D on RDNA2 even with `VICTORY_VIEW_COPY=0`, `VICTORY_TEXTURE_POOL=0`, `VICTORY_HELD_LEVELS=0`, `VICTORY_SHOW_NEW_LATER=0`, `VICTORY_LIBRARY_MAP=0`, `VICTORY_UNDER_WIDE=0`, and `VICTORY_LAND_ECHO=0`.
  - Use **`v15-ready`** for multi-hour stable gameplay on RX 6700 XT until the RDNA2 `0x887A0007` device reset is isolated with `VICTORY_D3D12_DEBUG=1`.

## How to run

You need Windows 10/11 x64, a CPU with AVX2, a Direct3D 12 GPU with Shader Model 6.6 and mesh shaders, Rust `1.98.1`, MSVC build tools, and your own copy of the title.

```powershell
cargo build --locked --profile diag -p victory --features diagnostics
cargo build --locked --profile ship -p victory
```

For RDNA2 (`RX 6700 XT`) live testing, avoid forcing `VICTORY_CARD_BUFFERS_MB=2048` (let the automatic 12 GB VRAM budget apply) and keep `VICTORY_VIEW_COPY=0`, `VICTORY_TEXTURE_POOL=0`, `VICTORY_HELD_LEVELS=0`, `VICTORY_SHOW_NEW_LATER=0`, `VICTORY_LIBRARY_MAP=0`, `VICTORY_MAP_THREADS=1`, and `VICTORY_TITLE_PRIORITIES=0`.

## Credits & License

- **[Marvisak/VIctory](https://github.com/Marvisak/VIctory)** — upstream loader, HLE, D3D12 backend, and shader recompiler.
- **[avatarmanz/VIctory-7900XTX](https://github.com/avatarmanz/VIctory-7900XTX)** — base architecture (`VICTORY_THIN`, `VICTORY_JOIN_PARTS`, `VICTORY_PORT_CONSTANTS`, `VICTORY_PORT_DESCRIPTORS`, `VICTORY_SHOW_AFTER`, `VICTORY_UPLOAD_AHEAD`, `ambient.rs`, `replay`).
- **[Belobug/VIctory-Belobug-AMD](https://github.com/Belobug/VIctory-Belobug-AMD)** (`4e1b16a8`) — magenta filter, VRAM eviction thresholds, `keep_shown` room guard, lock-free compiler VEH + `warm-bad.txt`.
- **[ka1qux](https://github.com/ka1qux/VIctory)** (`4bf4354d`, `f10ab5e1`) — anisotropic point-mip filtering probe and `tiles::let_go` deferred reclaim.
- **[triippywhale](https://github.com/triippywhale/VIctory)** (`0de380d8`) — commit-headroom idle cache trimming.
- **This fork (`v16-rx6700xt`)** — **AI-assisted** integration, profiling, and RX 6700 XT tuning (`EXACT_POOL`, cached `Shadow::uav_allowed()`, `VICTORY_THIN_THREADS` calibration, `VICTORY_KEYS_FILE` / `WASD` diag input, `[profile.ship]`).

GPL-3.0-or-later, as VIctory. See [LICENSE](LICENSE) and [docs/asset-policy.md](docs/asset-policy.md).
