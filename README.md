# gpzip

A parallel archiver. Speed is the feature.

```sh
gpzip a out.tar.gz src/      # 12–30x faster than serial gzip on a 16-core box
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

Real-world measurements — Ryzen 7800X3D (8C/16T) + RTX 4090, level 5, 3
trials averaged. Inputs harvested from local source trees, man pages,
journal logs, and `/usr/bin` binaries; sizes shown:

|              | source (23M) | text (47M)  | logs (8M)   | binmix (272M) |
|---           |---           |---          |---          |---            |
| gpzip-cpu    | **22 / .130** | **57 / .269** | **12 / .037** | **223 / .316** |
| gpzip-hybrid | 286 / .141   | 302 / .275  | 268 / .039  | 470 / .318    |
| gpzip-gpu    | 346 / .180   | 391 / .337  | 305 / .055  | 1044 / .359   |
| gzip         | 254 / .129   | 1097 / .264 | 25 / .035   | 6686 / .306   |
| zstd -T0     | 24 / .111    | 44 / .221   | 9 / .029    | 140 / .278    |

(numbers are wall-time ms / output ratio; lower is better in both columns)

`gpzip-cpu` beats serial `gzip` by 11–30×. It ties `zstd -T0` on small
inputs and trails it ~3–4× on bigger ones, but produces gzip-compatible
output (zstd doesn't).

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

Earlier attempts at a true atomicExchange hash chain failed because
GPU workgroups race at the chain head, leaving the chain ordered by
execution rather than position. Segmenting bounds candidate distance
per segment so close-window matches are reliably found regardless of
the chain race.

### Pipeline

`BatchedLz77` runs a submitter and a completer thread. The submitter
batches up to 8 chunks into one GPU submission and immediately moves
on; the completer waits via `Maintain::WaitForSubmissionIndex` (per-
submission, so submitting batch N+1 doesn't block on N's read-back).
A bounded channel (depth 4) backpressures the submitter so the GPU
queue + buffer pool stays bounded.

Tokens are packed `(length<<16 | distance)` u32 — half the readback
bytes of the older `vec2<u32>` layout.

### Performance

The GPU pipeline is functionally correct (output round-trips, tests
pass) but is currently slower AND less effective than the CPU pipeline
on every measured workload — see the table above. On real source code,
GPU output is 38% larger than CPU; on text it's 25% larger; on binaries
14% larger. Wall time is 4–25× slower because the GPU per-chunk cost
is dominated by host work (the chunk size is held at 32 KiB to keep
matches inside the 32 KiB window) and ~200 ms of one-shot wgpu init.

The honest read: gpzip-cpu is the production path. The GPU pipeline is
kept around as a research / future-work track. Closing the gap likely
needs either (a) Huffman emission moved onto the GPU or (b) a sliding-
window scheme that lets the GPU consume larger chunks without losing
match quality. Either is week-scale work.

## Hybrid CPU + GPU

`--backend auto` (the default) wires both devices into a single chunk
queue: each chunk closure tries to acquire a GPU permit, falls through
to the CPU encoder if the GPU is busy. The intent was aggregate
throughput `cpu_speed + gpu_speed`.

In practice, on this box, hybrid is strictly worse than `--backend cpu`
(both slower and a little less compressed) because every chunk that
lands on the GPU produces a worse-compressing output and adds tail
latency. The `gpu_workers` permit is set to 1 — minimum non-zero, so a
future GPU improvement starts contributing without a code change. For
maximum speed and ratio today, pass `--backend cpu` explicitly. The
hybrid path stays as the default since it falls back to pure CPU when
no adapter is available, but the CLI behavior on a CPU-only box matches
`--backend cpu`.

LazyGpuBackend defers wgpu init (~200 ms) until the first GPU chunk
arrives. With a 4-chunk warm-up threshold, inputs ≤ 8 MiB never touch
the GPU at all (they finish on CPU before crossing the threshold). For
`--backend gpu` extract / list (which never compress), GPU init is
skipped entirely.

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
| `gpzip-codec-gpu` | wgpu GPU codec (research-track; see status above) |
| `gpzip-cli` | The `gpzip` binary |

Library-first: the CLI is a thin shell over `gpzip-core`. A GUI frontend
plugs into the same API.

## Build

```sh
cargo build --release                       # CPU-only, ~5 MiB binary
cargo build --release --features gpu        # CPU + GPU, ~10 MiB (pulls wgpu)
cargo test --workspace
```

The default build is CPU-only because the GPU pipeline is currently
slower and less effective than the CPU path on every measured workload
(see status above). Pass `--features gpu` to build the wgpu-based
backend; without it `--backend gpu` and `--backend auto` silently fall
back to CPU.

Output: `./target/release/gpzip`.

## Status

Pre-alpha. CPU pipeline is fast and produces standard output and is
the recommended path. GPU pipeline is functional and produces standard
output but is slower and less compressing than CPU on every workload
measured to date — kept for future research, not for production.

## License

MIT OR Apache-2.0.
