// Phase 2 of segmented-hash LZ77 with two candidates per (hash, segment).
//
// For each segment within window, try both the OLDEST and the NEWEST
// candidate. Newest tends to give shorter distances (smaller Huffman
// codes); oldest is a fallback when newest is > p or fails hash check.
// Keep the longest match seen, with closer-distance tiebreak implicit
// because we prefer newer entries.

struct Params {
    input_len: u32,
    hash_bits: u32,
    window: u32,
    min_match: u32,
    max_match: u32,
    seg_log2: u32,
    num_segs: u32,
}

@group(0) @binding(0) var<storage, read>       input_buf:  array<u32>;
@group(0) @binding(1) var<storage, read>       seg_oldest: array<u32>;
@group(0) @binding(2) var<storage, read>       seg_newest: array<u32>;
// Packed token: length<<16 | distance/byte. See lz77_hash.rs `unpack_tokens`.
@group(0) @binding(3) var<storage, read_write> tokens:     array<u32>;
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

// Verify a candidate `raw` (= p_cand + 1, or sentinel) against position p
// and return (len, dist) if it produces a longer match than `cur_best`.
// Otherwise (0u, 0u). Sentinels: 0 (atomicMax unused) or 0xFFFFFFFF
// (atomicMin unused) both yield no match.
fn try_candidate(raw: u32, p: u32, cur_best: u32) -> vec2<u32> {
    if (raw == 0u || raw == 0xFFFFFFFFu) { return vec2<u32>(0u, 0u); }
    let cand = raw - 1u;
    if (cand >= p) { return vec2<u32>(0u, 0u); }
    let dist = p - cand;
    if (dist > params.window) { return vec2<u32>(0u, 0u); }
    if (read_byte(cand) != read_byte(p)
        || read_byte(cand + 1u) != read_byte(p + 1u)
        || read_byte(cand + 2u) != read_byte(p + 2u)) {
        return vec2<u32>(0u, 0u);
    }
    var len: u32 = 3u;
    loop {
        if (len >= params.max_match) { break; }
        if (p + len >= params.input_len) { break; }
        if (read_byte(cand + len) != read_byte(p + len)) { break; }
        len = len + 1u;
    }
    if (len > cur_best) {
        return vec2<u32>(len, dist);
    }
    return vec2<u32>(0u, 0u);
}

@compute @workgroup_size(64)
fn lookup(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.input_len) { return; }

    if (p + 2u >= params.input_len) {
        tokens[p] = read_byte(p);
        return;
    }

    let h = hash3(p);
    let p_seg = p >> params.seg_log2;
    var best_len: u32 = 0u;
    var best_dist: u32 = 0u;

    var seg_offset: u32 = 0u;
    loop {
        if (seg_offset > p_seg) { break; }
        let seg = p_seg - seg_offset;
        let idx = h * params.num_segs + seg;

        // Try newest first — typically closer in distance.
        let r_new = try_candidate(seg_newest[idx], p, best_len);
        if (r_new.x > 0u) {
            best_len = r_new.x;
            best_dist = r_new.y;
        }
        if (best_len >= params.max_match) { break; }

        // Skip oldest when newest already produced a long match. Saves
        // the second extension loop on rep-style data where newest tends
        // to give length 258 immediately. 16 is a heuristic that keeps
        // the bin ratio gain while avoiding the rep slowdown.
        if (best_len < 16u) {
            let r_old = try_candidate(seg_oldest[idx], p, best_len);
            if (r_old.x > 0u) {
                best_len = r_old.x;
                best_dist = r_old.y;
            }
            if (best_len >= params.max_match) { break; }
        }

        seg_offset = seg_offset + 1u;
    }

    if (best_len == 0u) {
        tokens[p] = read_byte(p);
    } else {
        tokens[p] = (best_len << 16u) | best_dist;
    }
}
