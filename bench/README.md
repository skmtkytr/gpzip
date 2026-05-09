# bench/

Benchmark harness for `gpzip` and the compressors it shares a namespace
with. No fixtures live in the repo — every workload is generated fresh
under `/tmp` per run.

## Quick start

```sh
./bench/bench.sh                        # 100 MB workloads, all detected tools
./bench/bench.sh --size 32              # smaller (32 MB) workloads
./bench/bench.sh --tools gpzip-cpu,gzip,zstd
./bench/bench.sh --workloads rep
```

`--size` accepts `K`/`M`/`G` suffixes (1024-based). Bare digits assume MiB,
so `--size 32` means 32 MiB.

## Knob sweeps

```sh
# Chunk-size sweep — gpzip backends only
./bench/bench.sh --tools gpzip-cpu --chunk-sizes 64K,256K,1M,4M

# Level sweep — every tool that has a compression level
./bench/bench.sh --levels 1,5,9

# Both at once: cartesian product
./bench/bench.sh --tools gpzip-cpu --chunk-sizes 256K,1M --levels 1,5,9
```

## Stability

```sh
# 5 runs, table reports median
./bench/bench.sh --runs 5
```

System-load jitter is usually 10-20% for sub-100ms cells. Three runs cuts
that noise to single-digit; five runs gets you within a percent or two.

## Decompression

```sh
./bench/bench.sh --decompress
```

Adds a second timing column per cell. cozip's runner doesn't expose a
decompress entry point, so cozip's decompress column reads `0`.

## Output formats

```sh
./bench/bench.sh --format table   # default, human-readable
./bench/bench.sh --format csv     # one row per (tool,chunk,level,workload)
./bench/bench.sh --format json    # array of objects
```

CSV / JSON include `out_bytes`, `in_bytes`, `ratio`. CSV is good for
piping into `awk` / `column -ts,` / spreadsheets.

## Workloads

| name | content | what it stresses |
|---|---|---|
| `rand` | bytes from `/dev/urandom` | raw throughput; nothing compresses |
| `rep`  | one short string repeated | extreme redundancy; LZ77 dictionary quality |
| `bin`  | tiled `/usr/bin/bash` | mid-entropy real-world-ish payload |
| `text` | tiled `/usr/share/dict/words` | natural-language redundancy |
| `log`  | synthetic timestamped log lines | structured-log redundancy |

## Tools auto-detected

| name | how |
|---|---|
| `gpzip-cpu` / `gpzip-gpu` / `gpzip-hybrid` | this repo's CLI with `--backend …` |
| `gzip` | system `gzip -c` |
| `pigz` | parallel gzip if `pigz` is on `PATH` |
| `zstd` | system `zstd -T0` (multi-threaded) |
| `cozip` | the `cozip_runner` helper at `/tmp/cozip-bench/run_on_file/target/release/cozip_runner` (build separately) |

## Sample output

```
  size=100.0M  runs=2  decompress=off

  tool                  rep  (ms / out / ratio)  bin  (ms / out / ratio)
  ----                  ----                    ----
  gpzip-cpu ck=64.0K       24ms  794.9K 0.008    133ms   54.4M 0.544
  gpzip-cpu ck=256.0K      23ms  682.5K 0.007    135ms   53.7M 0.537
  gpzip-cpu ck=1.0M        25ms  654.6K 0.006    137ms   53.5M 0.535
  gpzip-cpu ck=4.0M        27ms  647.5K 0.006    162ms   53.5M 0.535
```

The chunk-size sweep above shows the natural trade: bigger chunks let
LZ77 see more dictionary, ratio improves slightly, but parallelism per
worker drops and wall time creeps up. 256K-1M is the sweet spot here.

## Notes

- Output containers differ between tools (gpzip writes `tar.gz`, gzip
  writes `.gz`, zstd writes `.zst`, cozip writes its own `.czdf`). The
  size column compares apples to slightly-different apples; the
  per-byte compression overhead is small relative to the codec's
  output.
- Cells are independent — no cache priming between runs. Add `--runs N`
  for stability if comparing close numbers.
