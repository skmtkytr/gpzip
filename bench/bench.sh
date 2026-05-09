#!/usr/bin/env bash
# gpzip benchmark harness.
#
# Generates synthetic workloads under /tmp, runs every available compressor
# across a grid of (chunk_size × level × workload × run), and prints a table
# (or CSV / JSON) of compress and decompress times plus output size. Median
# over `--runs` smooths the system-load jitter.
#
# Usage:
#   ./bench/bench.sh                                  # 100 MB defaults
#   ./bench/bench.sh --size 32                        # smaller workloads
#   ./bench/bench.sh --tools gpzip-cpu,gzip,zstd      # restrict tools
#   ./bench/bench.sh --workloads rep,rand             # restrict workloads
#   ./bench/bench.sh --chunk-sizes 64K,256K,1M,4M     # gpzip chunk sweep
#   ./bench/bench.sh --levels 1,5,9                   # level sweep
#   ./bench/bench.sh --runs 3                         # median of 3
#   ./bench/bench.sh --decompress                     # also time decode
#   ./bench/bench.sh --format csv                     # machine-readable
#   ./bench/bench.sh --keep-out                       # keep /tmp work dir
#
# Sizes accept K/M/G suffixes (1024-based). Multiple values comma-separated.
#
# Recognised tools (auto-skipped if absent):
#   gpzip-cpu, gpzip-gpu, gpzip-hybrid (this repo's CLI)
#   gzip, pigz (system)
#   zstd (system, multi-threaded with -T0)
#   cozip (the cozip_runner helper at /tmp/cozip-bench/...)
#
# Workloads (`--workloads`, default rand,rep,bin):
#   rand     random bytes
#   rep      one short string repeated
#   bin      bash binary tiled
#   text     dictionary words tiled (if /usr/share/dict/words exists)
#   log      synthetic log lines

set -euo pipefail

# ----- defaults -----
SIZE_SPEC=100M
TOOLS=""
WORKLOADS="rand,rep,bin"
CHUNK_SIZES=""        # default = each tool's own default
LEVELS=""             # default = each tool's own default
RUNS=1
DECOMPRESS=0
FORMAT=table
KEEP_OUT=0

# ----- arg parsing -----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --size) SIZE_SPEC="$2"; shift 2 ;;
        --tools) TOOLS="$2"; shift 2 ;;
        --workloads) WORKLOADS="$2"; shift 2 ;;
        --chunk-sizes) CHUNK_SIZES="$2"; shift 2 ;;
        --levels) LEVELS="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --decompress) DECOMPRESS=1; shift ;;
        --format) FORMAT="$2"; shift 2 ;;
        --keep-out) KEEP_OUT=1; shift ;;
        -h|--help) sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# ----- helpers -----

# Parse SIZE strings like "200M", "64K", "1G" → bytes. Bare digits assume
# MiB (so `--size 32` means 32 MiB, matching the prior CLI behaviour).
parse_size() {
    local s="$1"
    local n="${s%[KkMmGg]}"
    case "$s" in
        *K|*k) echo $(( n * 1024 )) ;;
        *M|*m) echo $(( n * 1024 * 1024 )) ;;
        *G|*g) echo $(( n * 1024 * 1024 * 1024 )) ;;
        *)     echo $(( n * 1024 * 1024 )) ;;
    esac
}

# Format bytes as 1.2M / 64K / 17M etc.
fmt_bytes() {
    awk -v b="$1" 'BEGIN {
        if (b >= 1024*1024*1024) printf "%.1fG", b/1024/1024/1024;
        else if (b >= 1024*1024) printf "%.1fM", b/1024/1024;
        else if (b >= 1024)      printf "%.1fK", b/1024;
        else                     printf "%dB", b;
    }'
}

# Median of a list of integers passed via stdin (one per line).
median() {
    awk '
        { a[NR] = $1 }
        END {
            n = NR;
            asort(a);
            if (n % 2 == 1) print a[(n+1)/2];
            else            printf "%d\n", (a[n/2] + a[n/2+1]) / 2;
        }
    '
}

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

SIZE=$(parse_size "$SIZE_SPEC")

# ----- workload generation -----

gen_workload() {
    local name="$1"
    local out="$WORK/data/$name/payload.bin"
    [[ -f "$out" ]] && return
    mkdir -p "$(dirname "$out")"
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
        text)
            local dict=/usr/share/dict/words
            if [[ ! -r "$dict" ]]; then
                echo "no $dict; --workloads text unavailable" >&2; exit 1
            fi
            local seedsz; seedsz=$(stat -c %s "$dict")
            local copies=$(( SIZE / seedsz + 1 ))
            for ((i=0; i<copies; i++)); do cat "$dict"; done | head -c "$SIZE" > "$out"
            ;;
        log)
            (
                local i=0
                while :; do
                    i=$((i + 1))
                    printf '2026-05-09T12:34:56Z host[%d]: connection from 192.168.%d.%d port=%d ok\n' \
                        "$((i % 4096))" "$((i % 256))" "$((i % 256))" "$((i % 65535))"
                done
            ) | head -c "$SIZE" > "$out"
            ;;
        *) echo "unknown workload: $name" >&2; exit 1 ;;
    esac
    set -o pipefail
}

# ----- tool detection -----

# Also classify which knobs each tool honours. CHUNK = chunk-size, LVL = level.
declare -A TOOL_KNOBS=(
    [gpzip-cpu]="CHUNK LVL"
    [gpzip-gpu]="CHUNK LVL"
    [gpzip-hybrid]="CHUNK LVL"
    [gzip]="LVL"
    [pigz]="LVL"
    [zstd]="LVL"
    [cozip]=""
)

detect_tools() {
    [[ -x "$GPZIP" ]] && echo "gpzip-cpu"
    [[ -x "$GPZIP" ]] && echo "gpzip-gpu"
    [[ -x "$GPZIP" ]] && echo "gpzip-hybrid"
    command -v gzip >/dev/null && echo "gzip"
    command -v pigz >/dev/null && echo "pigz"
    command -v zstd >/dev/null && echo "zstd"
    [[ -x "$COZIP_RUNNER" ]] && echo "cozip"
}

# ----- single measurement -----
# Echoes "compress_ms decompress_ms out_bytes in_bytes" (or "FAIL").
# Args: tool workload chunk_size_bytes level out_path
run_one() {
    local tool="$1" wkl="$2" chunk="$3" lvl="$4" out="$5"
    local in_dir="$WORK/data/$wkl"
    local in_file="$in_dir/payload.bin"
    local in_size; in_size=$(stat -c %s "$in_file")
    local s e cms dms
    mkdir -p "$(dirname "$out")"

    # Compress.
    case "$tool" in
        gpzip-cpu)
            out="$out.tar.gz"
            s=$(date +%s%N)
            "$GPZIP" --backend cpu --chunk-size "$chunk" -q a "$out" "$in_dir" -l "$lvl" >/dev/null 2>&1
            e=$(date +%s%N) ;;
        gpzip-gpu)
            out="$out.tar.gz"
            s=$(date +%s%N)
            "$GPZIP" --backend gpu --chunk-size "$chunk" -q a "$out" "$in_dir" -l "$lvl" >/dev/null 2>&1
            e=$(date +%s%N) ;;
        gpzip-hybrid)
            out="$out.tar.gz"
            s=$(date +%s%N)
            "$GPZIP" --backend auto --chunk-size "$chunk" -q a "$out" "$in_dir" -l "$lvl" >/dev/null 2>&1
            e=$(date +%s%N) ;;
        gzip)
            out="$out.gz"
            s=$(date +%s%N)
            gzip -"$lvl" -c "$in_file" > "$out"
            e=$(date +%s%N) ;;
        pigz)
            out="$out.gz"
            s=$(date +%s%N)
            pigz -"$lvl" -c "$in_file" > "$out"
            e=$(date +%s%N) ;;
        zstd)
            out="$out.zst"
            s=$(date +%s%N)
            zstd -q -T0 -"$lvl" "$in_file" -o "$out" -f
            e=$(date +%s%N) ;;
        cozip)
            out="$out.czdf"
            s=$(date +%s%N)
            "$COZIP_RUNNER" "$in_file" "$out" >/dev/null 2>&1
            e=$(date +%s%N) ;;
        *) echo "FAIL"; return ;;
    esac
    cms=$(( (e - s) / 1000000 ))

    # Decompress (optional).
    dms=0
    if [[ "$DECOMPRESS" == 1 ]]; then
        local dest="$WORK/dec/$$.${RANDOM}"
        mkdir -p "$dest"
        case "$tool" in
            gpzip-*)
                s=$(date +%s%N)
                "$GPZIP" -q x "$out" -o "$dest" >/dev/null 2>&1
                e=$(date +%s%N) ;;
            gzip|pigz)
                s=$(date +%s%N)
                gunzip -c "$out" > /dev/null
                e=$(date +%s%N) ;;
            zstd)
                s=$(date +%s%N)
                zstd -q -d -c "$out" > /dev/null
                e=$(date +%s%N) ;;
            cozip)
                # No standalone decompressor in our runner. Mark as 0.
                e=$(date +%s%N); s=$e ;;
        esac
        dms=$(( (e - s) / 1000000 ))
        rm -rf "$dest"
    fi

    local out_size; out_size=$(stat -c %s "$out")
    rm -f "$out"
    echo "$cms $dms $out_size $in_size"
}

# ----- main -----

available=$(detect_tools)
if [[ -n "$TOOLS" ]]; then
    selected=$(echo "$TOOLS" | tr ',' '\n')
    available=$(echo "$available" | grep -Fx -f <(echo "$selected") || true)
fi
if [[ -z "$available" ]]; then
    echo "no tools available" >&2; exit 1
fi

workloads=$(echo "$WORKLOADS" | tr ',' ' ')

# chunk_sizes: parse, keep "default" sentinel if not provided.
chunk_list=()
if [[ -n "$CHUNK_SIZES" ]]; then
    for s in $(echo "$CHUNK_SIZES" | tr ',' ' '); do
        chunk_list+=("$(parse_size "$s")")
    done
else
    chunk_list+=("default")
fi

# levels.
level_list=()
if [[ -n "$LEVELS" ]]; then
    for l in $(echo "$LEVELS" | tr ',' ' '); do level_list+=("$l"); done
else
    level_list+=("default")
fi

# Default per-knob values for "default" sentinel.
default_chunk_size=$(( 2 * 1024 * 1024 ))
default_level=5

echo "==> generating $(fmt_bytes "$SIZE") workloads..." >&2
for w in $workloads; do gen_workload "$w"; done

# Build configs: list of "tool|chunk|level" strings.
configs=()
for t in $available; do
    knobs="${TOOL_KNOBS[$t]:-}"
    chunk_iter=()
    level_iter=()

    # If tool doesn't honor a knob, only emit one entry for it.
    if [[ "$knobs" == *CHUNK* ]]; then
        for c in "${chunk_list[@]}"; do chunk_iter+=("$c"); done
    else
        chunk_iter+=("default")
    fi
    if [[ "$knobs" == *LVL* ]]; then
        for l in "${level_list[@]}"; do level_iter+=("$l"); done
    else
        level_iter+=("default")
    fi

    for c in "${chunk_iter[@]}"; do
        for l in "${level_iter[@]}"; do
            configs+=("$t|$c|$l")
        done
    done
done

# ----- collect measurements -----
# Storage: associative array keyed "tool|chunk|level|workload" → "cms_med dms_med out_size"

declare -A RESULTS
declare -A IN_SIZES

for cfg in "${configs[@]}"; do
    IFS='|' read -r tool chunk lvl <<< "$cfg"
    real_chunk=$chunk
    real_lvl=$lvl
    [[ "$chunk" == "default" ]] && real_chunk=$default_chunk_size
    [[ "$lvl" == "default" ]] && real_lvl=$default_level

    for w in $workloads; do
        cms_list=""
        dms_list=""
        out_size=0
        in_size=0
        for ((r=0; r<RUNS; r++)); do
            res=$(run_one "$tool" "$w" "$real_chunk" "$real_lvl" "$WORK/out/${tool}_${chunk}_${lvl}_${w}_${r}" || echo "FAIL")
            if [[ "$res" == "FAIL" ]]; then
                cms_list="FAIL"; break
            fi
            read -r cms dms osz isz <<< "$res"
            cms_list+="$cms"$'\n'
            dms_list+="$dms"$'\n'
            out_size=$osz
            in_size=$isz
        done
        if [[ "$cms_list" == "FAIL" ]]; then
            RESULTS["$tool|$chunk|$lvl|$w"]="FAIL FAIL FAIL"
        else
            cms_med=$(echo -e "$cms_list" | grep -v '^$' | median)
            dms_med=$(echo -e "$dms_list" | grep -v '^$' | median)
            RESULTS["$tool|$chunk|$lvl|$w"]="$cms_med $dms_med $out_size"
        fi
        IN_SIZES["$w"]=$in_size
    done
done

# ----- output -----

config_label() {
    local tool="$1" chunk="$2" lvl="$3"
    local label="$tool"
    [[ "$chunk" != "default" ]] && label+=" ck=$(fmt_bytes "$chunk")"
    [[ "$lvl"   != "default" ]] && label+=" lv=$lvl"
    echo "$label"
}

case "$FORMAT" in
    csv)
        echo "tool,chunk_size,level,workload,compress_ms,decompress_ms,out_bytes,in_bytes,ratio"
        for cfg in "${configs[@]}"; do
            IFS='|' read -r tool chunk lvl <<< "$cfg"
            for w in $workloads; do
                read -r cms dms osz <<< "${RESULTS[$tool|$chunk|$lvl|$w]}"
                isz=${IN_SIZES[$w]}
                if [[ "$cms" == "FAIL" ]]; then
                    printf "%s,%s,%s,%s,FAIL,FAIL,FAIL,%s,FAIL\n" "$tool" "$chunk" "$lvl" "$w" "$isz"
                else
                    ratio=$(awk -v a="$osz" -v b="$isz" 'BEGIN { printf "%.4f", a/b }')
                    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" "$tool" "$chunk" "$lvl" "$w" "$cms" "$dms" "$osz" "$isz" "$ratio"
                fi
            done
        done
        ;;

    json)
        printf '['
        first=1
        for cfg in "${configs[@]}"; do
            IFS='|' read -r tool chunk lvl <<< "$cfg"
            for w in $workloads; do
                read -r cms dms osz <<< "${RESULTS[$tool|$chunk|$lvl|$w]}"
                isz=${IN_SIZES[$w]}
                [[ $first == 0 ]] && printf ','
                first=0
                if [[ "$cms" == "FAIL" ]]; then
                    printf '\n  {"tool":"%s","chunk_size":"%s","level":"%s","workload":"%s","status":"FAIL","in_bytes":%s}' \
                        "$tool" "$chunk" "$lvl" "$w" "$isz"
                else
                    ratio=$(awk -v a="$osz" -v b="$isz" 'BEGIN { printf "%.4f", a/b }')
                    printf '\n  {"tool":"%s","chunk_size":"%s","level":"%s","workload":"%s","compress_ms":%s,"decompress_ms":%s,"out_bytes":%s,"in_bytes":%s,"ratio":%s}' \
                        "$tool" "$chunk" "$lvl" "$w" "$cms" "$dms" "$osz" "$isz" "$ratio"
                fi
            done
        done
        printf '\n]\n'
        ;;

    table|*)
        echo
        echo "  size=$(fmt_bytes "$SIZE")  runs=$RUNS  decompress=$([[ "$DECOMPRESS" == 1 ]] && echo on || echo off)"
        # Column header line.
        if [[ "$DECOMPRESS" == 1 ]]; then
            cell_fmt="  %5dms %5dms %7s %5s   "
            cell_w=33
            cell_hdr="(c.ms / d.ms / out / ratio)"
        else
            cell_fmt="  %5dms %7s %5s   "
            cell_w=22
            cell_hdr="(ms / out / ratio)"
        fi
        max_label=20
        for cfg in "${configs[@]}"; do
            IFS='|' read -r t c l <<< "$cfg"
            label=$(config_label "$t" "$c" "$l")
            (( ${#label} > max_label )) && max_label=${#label}
        done
        printf "\n  %-${max_label}s" "tool"
        for w in $workloads; do
            printf "  %-${cell_w}s" "$w  $cell_hdr"
        done
        printf "\n"
        printf "  %-${max_label}s" "----"
        for w in $workloads; do
            printf "  %-${cell_w}s" "----"
        done
        printf "\n"

        for cfg in "${configs[@]}"; do
            IFS='|' read -r tool chunk lvl <<< "$cfg"
            label=$(config_label "$tool" "$chunk" "$lvl")
            printf "  %-${max_label}s" "$label"
            for w in $workloads; do
                read -r cms dms osz <<< "${RESULTS[$tool|$chunk|$lvl|$w]}"
                isz=${IN_SIZES[$w]}
                if [[ "$cms" == "FAIL" ]]; then
                    printf "  %-${cell_w}s" "FAIL"
                else
                    sz_h=$(fmt_bytes "$osz")
                    ratio=$(awk -v a="$osz" -v b="$isz" 'BEGIN { printf "%.3f", a/b }')
                    if [[ "$DECOMPRESS" == 1 ]]; then
                        printf "$cell_fmt" "$cms" "$dms" "$sz_h" "$ratio"
                    else
                        printf "$cell_fmt" "$cms" "$sz_h" "$ratio"
                    fi
                fi
            done
            printf "\n"
        done
        echo
        ;;
esac
