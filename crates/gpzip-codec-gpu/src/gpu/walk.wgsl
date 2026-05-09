// GPU strict-greedy LZ77 walk.
//
// Reads per-position tokens (one per input byte, packed u32 with
// length<<16 | distance/byte) and writes the walked, non-overlapping
// token stream into `walked[]`, along with the count.
//
// **Strict** greedy — no lazy peek-next-position. The host's
// `lz77::greedy_walk` does a lazy step (drop current match if next
// position has a strictly longer one), which improves compression by
// ~1-3% on text. Parallelising the lazy version on GPU needs special
// handling for the inter-position dependency; for the PoC we accept
// the strict-greedy ratio cost in exchange for a trivial single-pass
// shader.
//
// Single-thread (workgroup_size(1), 1 workgroup dispatched). Wasteful
// in raw GPU terms but the win isn't from parallelising the walk —
// it's from keeping the walked tokens on the GPU so a subsequent
// `huffman_emit_v2` dispatch can consume them without a host round
// trip. For a 128 KiB chunk with avg match length ~4, the walk is
// ~32K iterations of memory-read + branch + memory-write, which on
// modern GPU runs in a few hundred µs.

struct Params {
    n_positions: u32,
}

@group(0) @binding(0) var<storage, read>       per_position: array<u32>;
@group(0) @binding(1) var<storage, read_write> walked:       array<u32>;
@group(0) @binding(2) var<storage, read_write> walked_count: atomic<u32>;
@group(0) @binding(3) var<uniform>             params:       Params;

@compute @workgroup_size(1)
fn walk_serial() {
    var p: u32 = 0u;
    var out_idx: u32 = 0u;
    loop {
        if (p >= params.n_positions) { break; }
        let t = per_position[p];
        walked[out_idx] = t;
        out_idx = out_idx + 1u;
        let len = t >> 16u;
        if (len == 0u) {
            p = p + 1u;
        } else {
            p = p + len;
        }
    }
    atomicStore(&walked_count, out_idx);
}
