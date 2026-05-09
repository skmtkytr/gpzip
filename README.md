# gpzip

GPU-accelerated archiver. 7zip-style CLI focused on **fast** compression of formats Ark already understands.

## Status

Pre-alpha. CPU backend in progress; GPU backend (wgpu) is a stub.

## Goals

- **GPU-accelerated compression** for `zip`, `tar.gz`, `tar.zst` (cross-vendor via wgpu)
- **CPU decompression** for everything Ark opens: `zip`, `tar.gz`, `tar.zst`, `tar.xz`, `tar.bz2`, `rar`, `7z`
- Cooperative CPU + GPU pipeline: input is split into chunks, both devices race to compress them, results are reassembled in order (Chunk-Member Profile, inspired by [cozip](https://github.com/bea4dev/cozip))
- Library-first architecture (`gpzip-core`) so a GUI frontend can be added without touching codecs

## Non-goals (initial scope)

- GPU decompression of Deflate/Zstd: even cozip leaves this on CPU; the algorithms are too sequential to be worth the GPU port
- RAR/7z compression: not legally possible (RAR) or out of scope (7z)

## Crates

| Crate | Purpose |
|---|---|
| `gpzip-core` | UI/GPU-independent traits, archive I/O, progress events |
| `gpzip-codec-cpu` | CPU codec (flate2 / zstd / xz2 / bzip2 / unrar) |
| `gpzip-codec-gpu` | wgpu-based GPU compression backend |
| `gpzip-cli` | `gpzip a`, `gpzip x`, `gpzip l` |

## Build

```sh
cargo build --release            # CPU + GPU
cargo build --no-default-features --features cpu  # CPU-only
```

## License

MIT OR Apache-2.0. RAR support relies on the `unrar` C library; see its UnRAR License.
