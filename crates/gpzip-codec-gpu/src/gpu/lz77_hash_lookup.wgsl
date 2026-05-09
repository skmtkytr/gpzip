// Phase 2 of hash-chain LZ77.
//
// For each position p, walk the linked list at heads[hash(p)] following
// `next_buf` pointers, ordered newest-first. The chain is monotone in
// position (every link points to an earlier position), so once we walk past
// the window boundary we can stop. We bound work with `max_chain` (zlib's
// max_chain_length analogue) — past that, even a longer match isn't worth
// the lookup cost.
//
// Match selection: keep the longest valid match seen across up to
// max_chain candidates. Distance ties broken by closer-is-better (DEFLATE
// shorter distance code = fewer bits).

struct Params {
    input_len: u32,
    hash_bits: u32,
    window: u32,
    min_match: u32,
    max_match: u32,
    max_chain: u32,
}

@group(0) @binding(0) var<storage, read>       input_buf:  array<u32>;
@group(0) @binding(1) var<storage, read>       heads:      array<u32>;
@group(0) @binding(2) var<storage, read>       next_buf:   array<u32>;
@group(0) @binding(3) var<storage, read_write> tokens:     array<vec2<u32>>;
@group(0) @binding(4) var<uniform>             params:     Params;

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

    if (p + 2u >= params.input_len) {
        tokens[p] = vec2<u32>(0u, read_byte(p));
        return;
    }

    let h = hash3(p);
    var cand_p1: u32 = heads[h];
    var depth: u32 = 0u;
    var best_len: u32 = 0u;
    var best_dist: u32 = 0u;

    loop {
        if (cand_p1 == 0u) { break; }
        if (depth >= params.max_chain) { break; }

        let cand = cand_p1 - 1u;
        // Skip our own position and any "later" position (chain may briefly
        // contain entries from concurrent threads with higher p).
        if (cand >= p) {
            cand_p1 = next_buf[cand];
            depth = depth + 1u;
            continue;
        }
        let dist = p - cand;
        // Chain is monotone-decreasing in position once we pass our own p,
        // so once we're outside the window we can stop walking entirely.
        if (dist > params.window) { break; }

        // Hash collision check on first 3 bytes. Cheap, avoids the byte loop
        // on collisions.
        if (read_byte(cand) == read_byte(p)
            && read_byte(cand + 1u) == read_byte(p + 1u)
            && read_byte(cand + 2u) == read_byte(p + 2u)) {
            // Extend match forward.
            var len: u32 = 3u;
            loop {
                if (len >= params.max_match) { break; }
                if (p + len >= params.input_len) { break; }
                if (read_byte(cand + len) != read_byte(p + len)) { break; }
                len = len + 1u;
            }
            // Keep longest. On ties, the chain order means we already have
            // the closest (newest) one as best_dist.
            if (len > best_len) {
                best_len = len;
                best_dist = dist;
            }
        }

        cand_p1 = next_buf[cand];
        depth = depth + 1u;
    }

    if (best_len == 0u) {
        tokens[p] = vec2<u32>(0u, read_byte(p));
    } else {
        tokens[p] = vec2<u32>(best_len, best_dist);
    }
}
