#!/usr/bin/env bash
# gpzip benchmark harness.
#
# Generates a few test workloads (random / repetitive / binary), runs every
# available compressor on each, and prints a compact table of wall time and
# output size. Intended to be cheap to run from a fresh checkout — no
# special hardware assumptions, no fixtures committed.
#
# Usage:
#   ./bench/bench.sh                       # 100 MB workloads, all tools
#   ./bench/bench.sh --size 32             # smaller (32 MB) workloads
#   ./bench/bench.sh --tools gpzip,gzip    # restrict to listed tools
#   ./bench/bench.sh --workloads rep,rand  # restrict workloads
#   ./bench/bench.sh --keep-out            # don't delete /tmp work dir
#
# Recognised tools (auto-skipped if binary not found):
#   gpzip-cpu       - this repo's CLI, --backend cpu
#   gpzip-gpu       - this repo's CLI, --backend gpu
#   gpzip-hybrid    - this repo's CLI, --backend auto (default)
#   gzip            - system gzip (single-threaded baseline)
#   pigz            - parallel gzip if installed
#   zstd            - system zstd
#   cozip           - cozip_runner if /tmp/cozip-bench/run_on_file built
#
# Recognised workloads:
#   rand   - random bytes (uncompressible; tests pure throughput)
#   rep    - one short string repeated (extreme redundancy)
#   bin    - bash binary tiled to fill (mid entropy, real-world-ish)

set -euo pipefail

SIZE_MB=100
TOOLS=""
WORKLOADS="rand,rep,bin"
KEEP_OUT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --size) SIZE_MB="$2"; shift 2 ;;
        --tools) TOOLS="$2"; shift 2 ;;
        --workloads) WORKLOADS="$2"; shift 2 ;;
        --keep-out) KEEP_OUT=1; shift ;;
        -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPZIP="$ROOT/target/release/gpzip"
COZIP_RUNNER="/tmp/cozip-bench/run_on_file/target/release/cozip_runner"

if [[ ! -x "$GPZIP" ]]; then
    echo "==> building gpzip (release)..." >&2
    (cd "$ROOT" && cargo build --release --bin gpzip --quiet)
fi

WORK="$(mktemp -d)"
if [[ "$KEEP_OUT" == 0 ]]; then
    trap 'rm -rf "$WORK"' EXIT
else
    echo "work dir: $WORK"
fi

SIZE=$(( SIZE_MB * 1024 * 1024 ))

gen_workload() {
    local name="$1"
    local out="$WORK/data/$name/payload.bin"
    mkdir -p "$(dirname "$out")"
    # `yes | head -c N` and the bash-tile pipeline both rely on the upstream
    # producer exiting with SIGPIPE when `head` closes its read end. That
    # exit-by-signal trips `set -o pipefail`, so disable it locally.
    set +o pipefail
    case "$name" in
        rand)
            head -c "$SIZE" /dev/urandom > "$out" ;;
        rep)
            yes "the quick brown fox jumps over the lazy dog $$" \
                | head -c "$SIZE" > "$out" ;;
        bin)
            local seed=/usr/bin/bash
            local seedsz; seedsz=$(stat -c %s "$seed")
            local copies=$(( SIZE / seedsz + 1 ))
            for ((i=0; i<copies; i++)); do cat "$seed"; done | head -c "$SIZE" > "$out"
            ;;
        *) set -o pipefail; echo "unknown workload: $name" >&2; exit 1 ;;
    esac
    set -o pipefail
}

detect_tools() {
    [[ -x "$GPZIP" ]] && echo "gpzip-cpu"
    [[ -x "$GPZIP" ]] && echo "gpzip-gpu"
    [[ -x "$GPZIP" ]] && echo "gpzip-hybrid"
    command -v gzip >/dev/null && echo "gzip"
    command -v pigz >/dev/null && echo "pigz"
    command -v zstd >/dev/null && echo "zstd"
    [[ -x "$COZIP_RUNNER" ]] && echo "cozip"
}

run_one() {
    local tool="$1"
    local in_dir="$WORK/data/$2"
    local in_file="$in_dir/payload.bin"
    local out="$WORK/out/${tool}_$2"
    mkdir -p "$(dirname "$out")"

    local in_size; in_size=$(stat -c %s "$in_file")
    local start end ms

    case "$tool" in
        gpzip-cpu)
            out="$out.tar.gz"
            start=$(date +%s%N)
            "$GPZIP" --backend cpu -q a "$out" "$in_dir" >/dev/null 2>&1
            end=$(date +%s%N) ;;
        gpzip-gpu)
            out="$out.tar.gz"
            start=$(date +%s%N)
            "$GPZIP" --backend gpu -q a "$out" "$in_dir" >/dev/null 2>&1
            end=$(date +%s%N) ;;
        gpzip-hybrid)
            out="$out.tar.gz"
            start=$(date +%s%N)
            "$GPZIP" --backend auto -q a "$out" "$in_dir" >/dev/null 2>&1
            end=$(date +%s%N) ;;
        gzip)
            out="$out.gz"
            start=$(date +%s%N)
            gzip -c "$in_file" > "$out"
            end=$(date +%s%N) ;;
        pigz)
            out="$out.gz"
            start=$(date +%s%N)
            pigz -c "$in_file" > "$out"
            end=$(date +%s%N) ;;
        zstd)
            out="$out.zst"
            start=$(date +%s%N)
            zstd -q -T0 "$in_file" -o "$out" -f
            end=$(date +%s%N) ;;
        cozip)
            out="$out.czdf"
            start=$(date +%s%N)
            "$COZIP_RUNNER" "$in_file" "$out" >/dev/null 2>&1
            end=$(date +%s%N) ;;
        *) echo "FAIL"; return ;;
    esac

    ms=$(( (end - start) / 1000000 ))
    local out_size; out_size=$(stat -c %s "$out")
    echo "$ms $out_size $in_size"
}

available=$(detect_tools)
if [[ -n "$TOOLS" ]]; then
    selected=$(echo "$TOOLS" | tr ',' '\n')
    available=$(echo "$available" | grep -Fx -f <(echo "$selected") || true)
fi
if [[ -z "$available" ]]; then
    echo "no tools available" >&2; exit 1
fi

workloads=$(echo "$WORKLOADS" | tr ',' ' ')

echo "==> generating ${SIZE_MB} MB workloads..."
for w in $workloads; do
    gen_workload "$w"
done

printf "\n  %-15s" "tool"
for w in $workloads; do
    printf "  %-26s" "$w  (ms / out / ratio)"
done
printf "\n"
printf "  %-15s" "----"
for w in $workloads; do
    printf "  %-26s" "----"
done
printf "\n"

for t in $available; do
    printf "  %-15s" "$t"
    for w in $workloads; do
        result=$(run_one "$t" "$w" || echo "FAIL")
        if [[ "$result" == "FAIL" ]]; then
            printf "  %-26s" "FAIL"
        else
            ms=$(echo "$result" | awk '{print $1}')
            sz=$(echo "$result" | awk '{print $2}')
            insz=$(echo "$result" | awk '{print $3}')
            sz_mb=$(awk -v s="$sz" 'BEGIN { printf "%.1fM", s/1024/1024 }')
            ratio=$(awk -v s="$sz" -v i="$insz" 'BEGIN { printf "%.3f", s/i }')
            printf "  %6dms %8s %6s    " "$ms" "$sz_mb" "$ratio"
        fi
    done
    printf "\n"
done
echo
