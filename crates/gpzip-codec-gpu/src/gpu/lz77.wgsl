// Per-position parallel LZ77 match-find.
//
// For each position p in the input, every thread independently scans backward
// up to WINDOW bytes and reports the longest run of bytes that match the
// suffix starting at p. Output is one token per position:
//   (length, distance) where length >= MIN_MATCH means a back-reference,
//   (0, byte)          means a literal.
//
// This shader does NOT decide which matches the encoder uses. A serial pass
// on the host walks the per-position output and applies a greedy policy
// (take a match if length >= MIN_MATCH, else emit literal; advance by length
// or 1). That keeps the selection logic simple and lets every thread work
// independently — no atomics, no inter-thread coordination.

struct Params {
    input_len: u32,
    window:    u32,  // max look-back
    min_match: u32,  // minimum match length to record
    max_match: u32,  // cap on a single match (DEFLATE uses 258)
}

@group(0) @binding(0) var<storage, read>       input_buf: array<u32>;
@group(0) @binding(1) var<storage, read_write> tokens:    array<vec2<u32>>;
@group(0) @binding(2) var<uniform>             params:    Params;

fn read_byte(idx: u32) -> u32 {
    let word  = input_buf[idx / 4u];
    let shift = (idx % 4u) * 8u;
    return (word >> shift) & 0xffu;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos = gid.x;
    if (pos >= params.input_len) { return; }

    var best_len:  u32 = 0u;
    var best_dist: u32 = 0u;

    // Backward window. Saturating subtraction so window_start stays >= 0.
    var window_start: u32 = 0u;
    if (pos > params.window) {
        window_start = pos - params.window;
    }

    // Brute-force: try every prior position in the window. O(window)
    // per thread, but threads run in parallel so total work is well
    // distributed. Replaced by hash-table approach in A-2d.
    for (var prev: u32 = window_start; prev < pos; prev = prev + 1u) {
        var len: u32 = 0u;
        loop {
            if (len >= params.max_match) { break; }
            if (pos + len >= params.input_len) { break; }
            let a = read_byte(prev + len);
            let b = read_byte(pos + len);
            if (a != b) { break; }
            len = len + 1u;
        }
        if (len > best_len) {
            best_len  = len;
            best_dist = pos - prev;
        }
    }

    if (best_len >= params.min_match) {
        tokens[pos] = vec2<u32>(best_len, best_dist);
    } else {
        tokens[pos] = vec2<u32>(0u, read_byte(pos));
    }
}
