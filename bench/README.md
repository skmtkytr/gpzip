# bench/

A small benchmark harness for `gpzip` and the compressors it shares a
namespace with. No fixtures live in the repo — every workload is
generated fresh under `/tmp` per run, and results print as a single
ASCII table.

## Quick start

```sh
./bench/bench.sh                     # 100 MB workloads, all detected tools
./bench/bench.sh --size 32           # smaller (32 MB) workloads
./bench/bench.sh --tools gpzip-cpu,gzip,zstd
./bench/bench.sh --workloads rep
```

## What it measures

For each `(tool, workload)` pair: wall-clock compression time and final
output size. No decompression measurement (that's a separate concern;
single-stream gzip/zstd decode is sequential by nature).

## Workloads

| name | content | what it stresses |
|---|---|---|
| `rand` | bytes from `/dev/urandom` | raw throughput; nothing compresses |
| `rep`  | one short string repeated | extreme redundancy; LZ77 dictionary quality |
| `bin`  | tiled `/usr/bin/bash` | mid-entropy real-world-ish payload |

## Tools auto-detected

| name | how |
|---|---|
| `gpzip-cpu` / `gpzip-gpu` / `gpzip-hybrid` | this repo's CLI with `--backend …` |
| `gzip` | system `gzip -c` |
| `pigz` | parallel gzip if `pigz` is on `PATH` |
| `zstd` | system `zstd -T0` (multi-threaded) |
| `cozip` | the `cozip_runner` helper at `/tmp/cozip-bench/run_on_file/target/release/cozip_runner` (build separately) |

Anything not on `PATH` (or not built) is skipped silently.

## Sample output

```
  tool             rand  (ms / out / ratio)    rep  (ms / out / ratio)     bin  (ms / out / ratio)
  ----             ----                        ----                        ----
  gpzip-cpu            89ms    32.0M  1.000        18ms     0.1M  0.003       101ms    16.7M  0.522
  gpzip-gpu           407ms    32.0M  1.001       287ms    17.8M  0.557       342ms    22.6M  0.707
  gpzip-hybrid        339ms    32.0M  1.000       254ms     2.4M  0.076       335ms    17.7M  0.553
  gzip                503ms    32.0M  1.000        38ms     0.1M  0.003      1320ms    16.6M  0.518
  zstd                 20ms    32.0M  1.000        14ms     0.0M  0.000        18ms     1.9M  0.058
  cozip               342ms    32.0M  1.000       246ms     0.2M  0.006       359ms    18.5M  0.578
```

(Times are dominated by thread count, page cache state, and
disk/PCIe contention — treat them as rough order-of-magnitude, not
absolute.)

## Notes

- Output containers differ between tools (gpzip writes `tar.gz`, gzip
  writes `.gz`, zstd writes `.zst`, cozip writes its own `.czdf`). The
  size column compares apples to slightly-different apples; the
  per-byte compression overhead is small relative to the codec's
  output.
- The `gpzip-gpu` and `gpzip-hybrid` rows are the parts of this
  project that are still finding their feet — their times and ratios
  will move as the GPU shaders mature.
