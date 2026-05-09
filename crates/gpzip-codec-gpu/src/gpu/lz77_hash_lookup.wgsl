// Phase 2 of hash-table LZ77.
//
// For each position p, look up the hash table entry, verify the 3-byte
// hash hit isn't a collision, then extend the match forward. Output one
// Token per input byte (literal or back-reference).

struct Params {
    input_len: u32,
    hash_bits: u32,
    window: u32,
    min_match: u32,
    max_match: u32,
}

@group(0) @binding(0) var<storage, read>       input_buf:  array<u32>;
@group(0) @binding(1) var<storage, read>       hash_table: array<u32>;
@group(0) @binding(2) var<storage, read_write> tokens:     array<vec2<u32>>;
@group(0) @binding(3) var<uniform>             params:     Params;

fn read_byte(idx: u32) -> u32 {
    let word  = input_buf[idx / 4u];
    let shift = (idx % 4u) * 8u;
    return (word >> shift) & 0xffu;
}

fn hash3(p: u32) -> u32 {
    let a = read_byte(p);
    let b = read_byte(p + 1u);
    let c = read_byte(p + 2u);
    let x = (a << 16u) | (b << 8u) | c;
    let h = x * 0x9E3779B1u;
    return h >> (32u - params.hash_bits);
}

@compute @workgroup_size(64)
fn lookup(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.input_len) { return; }

    // Tail of input: not enough bytes for a 3-byte hash → emit literal.
    if (p + 2u >= params.input_len) {
        tokens[p] = vec2<u32>(0u, read_byte(p));
        return;
    }

    let h = hash3(p);
    let raw = hash_table[h];
    // 0xFFFFFFFFu is the unused sentinel; raw - 1 underflows there too,
    // so check before subtracting.
    if (raw == 0xFFFFFFFFu) {
        tokens[p] = vec2<u32>(0u, read_byte(p));
        return;
    }
    let prev = raw - 1u;
    if (prev >= p || p - prev > params.window) {
        tokens[p] = vec2<u32>(0u, read_byte(p));
        return;
    }

    // Hash collision check: confirm the 3-byte prefix actually matches.
    if (read_byte(prev) != read_byte(p)) {
        tokens[p] = vec2<u32>(0u, read_byte(p));
        return;
    }
    if (read_byte(prev + 1u) != read_byte(p + 1u)) {
        tokens[p] = vec2<u32>(0u, read_byte(p));
        return;
    }
    if (read_byte(prev + 2u) != read_byte(p + 2u)) {
        tokens[p] = vec2<u32>(0u, read_byte(p));
        return;
    }

    // Extend forward.
    var len: u32 = 3u;
    loop {
        if (len >= params.max_match) { break; }
        if (p + len >= params.input_len) { break; }
        if (read_byte(prev + len) != read_byte(p + len)) { break; }
        len = len + 1u;
    }

    tokens[p] = vec2<u32>(len, p - prev);
}
