# gpzip

A GPU-first archiver. Speed is the feature.

```sh
gpzip a out.tar.gz src/      # GPU LZ77 + chunk-parallel host pipeline
gpzip x out.tar.gz -o dest/
gpzip l out.tar.gz
```

The end goal: push the LZ77 inner loop — the compressor's bottleneck —
onto the GPU, while keeping output a standard gzip / zstd / zip stream
that any tool can decode. The chunk-parallel scaffolding around it is
shared with the CPU backend, which is fast in its own right and serves
as the working baseline while the GPU pipeline catches up.

## Formats

Compress: `zip`, `tar.gz`, `tar.zst`

Extract: `zip`, `tar`, `tar.gz`, `tar.zst`, `tar.xz`, `tar.bz2`, `rar`, `7z`

The compressed output is a normal gzip / zstd / zip stream — chunks are
emitted as concatenated gzip members or zstd frames, both of which are
defined by the respective format specs. Anything that decodes the format
decodes gpzip's output.

## How it goes fast

Input is split into fixed-size chunks (default 2 MiB) and each chunk is
compressed independently. Workers race on a shared queue; finished
chunks are reassembled in order at the output. Chunk independence costs
a tiny sliver of compression ratio and buys you all your cores —
straightforward for the CPU backend, and the *only* way the GPU
backend's per-chunk pipeline can be useful at all.

## GPU pipeline

`--backend gpu` runs LZ77 match-finding on the GPU via wgpu (Vulkan /
Metal / DX12 — cross-vendor). The host then encodes the GPU-emitted
token stream into a dynamic-Huffman DEFLATE block (RFC 1951 §3.2.7) and
frames it as a standard gzip member (RFC 1952). Output is normal gzip;
`gzip -t` and `tar tzf` accept it.

### Match finder

Two-pass segmented-hash LZ77:

1. **Build** — every position writes `(p+1)` into `seg_oldest[hash][p>>12]`
   (atomicMin) AND `seg_newest[hash][p>>12]` (atomicMax), giving each
   (hash, 4 KiB segment) a guaranteed earliest-and-latest candidate.
2. **Lookup** — walks segments backward from p's own segment until
   distance exceeds the 32 KiB window. Tries the newest candidate first
   (closer in distance → shorter Huffman code), falls back to oldest
   only when the newest doesn't yield a long-enough match.

The segmented design is what survived after a true atomicExchange hash
chain failed: GPU workgroups race at the chain head, so the chain ends
up ordered by execution rather than position and lookups can't reach
close-window matches reliably. Segmenting bounds candidate distance per
segment, so a window-eligible candidate is guaranteed regardless of the
build race.

### Pipeline

`BatchedLz77` runs a submitter and a completer thread. The submitter
batches up to 8 chunks into one GPU submission and immediately moves
on; the completer waits via `Maintain::WaitForSubmissionIndex` (per-
submission, so submitting batch N+1 doesn't block on N's read-back).
A bounded channel (depth 4) backpressures the submitter so the GPU
queue + buffer pool stays bounded.

Tokens are packed `(length<<16 | distance)` u32 — half the readback
bytes of the older `vec2<u32>` layout. The host-side dynamic-Huffman
encoder (`encode_block_fast`) was rewritten in this same pass:
pre-reversed Huffman codes, u64 bit accumulator, direct LUT for length
and distance symbols. Single-stage profile dropped from 11.5 ms to
1.3 ms per 512 KiB random chunk (8.6×).

### Where the GPU stands today

Functionally correct (round-trips verified, tests pass) but on the
benchmark box (Ryzen 7800X3D + RTX 4090) the per-chunk wall time and
the output ratio both still trail the CPU path. See the table below.

The dynamic-Huffman emit phase is now also available on the GPU
(D-3, opt-in via `GPZIP_GPU_ENCODE=1`). Single-call benches showed
the GPU emit at ~2× host_dyn, but under production load it's
roughly net-neutral — host encoding parallelises across rayon
workers while the GPU encoder serialises through one worker. The env
var is a measurement / fallback knob, not the production default.

## Hybrid CPU + GPU

`--backend auto` (the default) wires both devices into a single chunk
queue: each chunk closure tries to acquire a GPU permit (1 by default),
falls through to the CPU encoder if the GPU is busy. With the GPU's
current per-chunk cost the hybrid wall is dominated by the CPU path,
and the GPU permit is held at 1 so a future GPU speedup contributes
without a code change.

`LazyGpuBackend` defers wgpu init (~200 ms) until the first GPU chunk
arrives. With a 4-chunk warm-up threshold, inputs ≤ 8 MiB never touch
the GPU at all (they finish on CPU before crossing the threshold). For
`--backend gpu` extract / list (which never compress), GPU init is
skipped entirely.

## Real-world bench

Ryzen 7800X3D (8C/16T) + RTX 4090, level 5, 3 trials averaged. Inputs
harvested from local source trees, man pages, journal logs, and
`/usr/bin` binaries; sizes shown.

|              | source (23M) | text (47M)  | logs (9M)   | binmix (272M) |
|---           |---           |---          |---          |---            |
| gpzip-cpu    | 26 / .130    | 56 / .269   | 12 / .046   | 233 / .316    |
| gpzip-hybrid | 227 / .133   | 234 / .271  | 212 / .050  | 411 / .317    |
| gpzip-gpu    | 267 / .162   | 300 / .309  | 246 / .058  | 631 / .346    |
| gzip serial  | 256 / .129   | 1110 / .264 | 33 / .045   | 6745 / .306   |
| pigz -p 16   | 20 / .127    | 49 / .267   | 6 / .046    | 235 / .314    |
| zstd -T0     | 24 / .111    | 45 / .221   | 10 / .037   | 143 / .278    |

(numbers are wall-time ms / output ratio; lower is better in both)

The CPU backend is gzip-compatible and competitive: 11–30× faster than
serial gzip and within noise of `pigz` on real workloads. For zstd
output, gpzip-cpu's chunk-parallel `tar.zst` runs ~1.3–1.4× faster
than `zstd -T0` at a 4–10% larger output (the chunk-independence ratio
cost). The GPU backend is the active research direction.

### Decompression

`gpzip x` parallel-decodes its own gzip output by scanning for member
boundaries (gpzip's gzip member header is fixed at 10 bytes) and
running `flate2` per member via rayon. Falls back to serial
`MultiGzDecoder` for non-gpzip inputs (system gzip with FNAME etc.).
On the binmix workload (86 MiB compressed → 272 MiB raw): `gpzip x`
319 ms, `pigz -dc` 341 ms, `gzip -dc` 635 ms.

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
| `gpzip-codec-cpu` | CPU codec + chunk-parallel writer + parallel gzip decompress |
| `gpzip-codec-gpu` | wgpu GPU codec — segmented-hash LZ77, packed tokens, two-stage pipeline |
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

Pre-alpha. CPU backend is fast and gzip-compatible; GPU backend is
functional and produces standard output but is slower per chunk than
the CPU on the workloads measured so far. Active development is on the
GPU pipeline — closing the per-chunk gap is the project's main
direction.

## License

MIT OR Apache-2.0.
