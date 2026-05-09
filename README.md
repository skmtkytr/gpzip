# gpzip

A GPU-first archiver. Speed is the feature.

```sh
gpzip a out.tar.gz src/      # 4x faster than serial gzip on a 16-core box
gpzip x out.tar.gz -o dest/
gpzip l out.tar.gz
```

## Formats

Compress: `zip`, `tar.gz`, `tar.zst`

Extract: `zip`, `tar`, `tar.gz`, `tar.zst`, `tar.xz`, `tar.bz2`, `rar`, `7z`

The compressed output is a normal gzip / zstd / zip stream — chunks are
emitted as concatenated gzip members or zstd frames, both of which are
defined by the respective format specs. Anything that decodes the format
decodes gpzip's output.

## How it goes fast

Input is split into fixed-size chunks (default 2 MiB) and each chunk is
compressed independently. Workers race on a shared queue; finished chunks
are reassembled in order at the output. Chunk independence costs a tiny
sliver of compression ratio and buys you all your cores.

Measured on a 16-core box, 200 MB mixed input, level 5:

| | serial | parallel |
|---|---|---|
| `tar.gz`  | 1.5 s | **0.38 s** (4.0x) |
| `tar.zst` | 0.36 s | **0.28 s** (1.3x) |

## GPU pipeline

`--backend gpu` runs LZ77 match-finding on the GPU via wgpu (Vulkan / Metal
/ DX12 — cross-vendor). The host then encodes the GPU-emitted token stream
into a fixed-Huffman DEFLATE block (RFC 1951 §3.2.6) and frames it as a
standard gzip member (RFC 1952). Output is normal gzip; `gzip -t` and
`tar tzf` accept it.

The shader is brute-force today (O(window) per byte, 4 KiB window), so the
GPU path is functional but slower than the CPU path on every input we've
benchmarked. The point is the integration: a hash-table LZ77 shader and a
GPU-side Huffman pass plug into the same pipeline without disturbing the
gzip output contract.

## Flags

```
gpzip <a|x|l> ARCHIVE [INPUTS...]

  -l, --level N         compression level 0..=9            (default 5)
  -o, --output DIR      extract destination                (default .)
      --threads N       worker count, 0 = all cores        (default 0)
      --chunk-size B    chunk bytes                        (default 2097152)
      --backend BE      cpu | gpu | auto                   (default auto)
```

## Crates

| Crate | Purpose |
|---|---|
| `gpzip-core` | Codec traits, archive I/O, backend registry, progress events |
| `gpzip-codec-cpu` | CPU codec + chunk-parallel writer for gzip / zstd |
| `gpzip-codec-gpu` | wgpu GPU codec (in progress) |
| `gpzip-cli` | The `gpzip` binary |

Library-first: the CLI is a thin shell over `gpzip-core`. A GUI frontend
plugs into the same API.

## Build

```sh
cargo build --release
cargo test --workspace
```

Output: `./target/release/gpzip`.

## Status

Pre-alpha. CPU pipeline works and is fast. GPU pipeline produces valid
gzip but is slow until the brute-force LZ77 shader gets replaced with a
hash-table variant.

## License

MIT OR Apache-2.0.
