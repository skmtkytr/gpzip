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
into a dynamic-Huffman DEFLATE block (RFC 1951 §3.2.7) and frames it as a
standard gzip member (RFC 1952). Output is normal gzip; `gzip -t` and
`tar tzf` accept it.

The match-finder is a two-pass hash-table shader: pass 1 writes each
position into a 4-way bucket (atomicMin per sub-slot; oldest-wins lock-free
build), pass 2 looks up all 4 sub-slots and picks the closest prior
position. Closer matches than ideal would need a real hash chain
(atomicCAS-based linked-list inserts), and the encoder side could use
package-merge length-limiting instead of frequency scaling — both are
future work.

## Hybrid CPU + GPU

`--backend auto` (the default) wires both devices into a single chunk
queue: each chunk closure tries to acquire a GPU permit, falls through
to the CPU encoder if the GPU is busy. Aggregate throughput is
`cpu_speed + gpu_speed`, which is the only way the GPU path (slower per
chunk than CPU today) can help end-to-end wall time.

Today the math doesn't work — the GPU path is so much slower per chunk
that the few chunks that land on it dominate the tail and hybrid loses
to plain CPU. Pass `--backend cpu` for the fast path until the GPU
shaders close the gap. The hybrid scaffolding stays so the speedup
arrives automatically once the per-chunk gap shrinks.

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

Pre-alpha. CPU pipeline is fast and produces standard output. GPU
pipeline is functional and produces standard output — speed and ratio
both still trail CPU on every workload measured.

## License

MIT OR Apache-2.0.
