// Phase 2 of segmented-hash LZ77.
//
// For each position p, walk segments p_seg down to 0 (or until distance
// exceeds window). Each segment provides one candidate (the OLDEST
// position in that segment with the same 3-byte hash). Verify each, take
// the longest match found.
//
// Walking is bounded by window/seg_size segments at most — for the default
// SEG_LOG2=12 (4 KiB) and window=32 KiB, that's 8 segments per lookup.

struct Params {
    input_len: u32,
    hash_bits: u32,
    window: u32,
    min_match: u32,
    max_match: u32,
    seg_log2: u32,
    num_segs: u32,
}

@group(0) @binding(0) var<storage, read>       input_buf: array<u32>;
@group(0) @binding(1) var<storage, read>       seg_table: array<u32>;
// Packed token: length in bits 16..31, distance/byte in bits 0..15.
// Length max 258 fits in 9 bits; distance max 32768 fits in 15 bits;
// literal byte fits in 8 bits — all comfortably inside one u16 each.
// Halves the tokens buffer size (and PCIe readback) vs vec2<u32>.
@group(0) @binding(2) var<storage, read_write> tokens:    array<u32>;
@group(0) @binding(3) var<uniform>             params:    Params;

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
        tokens[p] = read_byte(p);
        return;
    }

    let h = hash3(p);
    let p_seg = p >> params.seg_log2;
    var best_len: u32 = 0u;
    var best_dist: u32 = 0u;

    // Walk current segment and earlier ones. p's own segment is included
    // because the bucket may contain a same-segment position with smaller p.
    var seg_offset: u32 = 0u;
    loop {
        if (seg_offset > p_seg) { break; }
        let seg = p_seg - seg_offset;
        let raw = seg_table[h * params.num_segs + seg];
        if (raw == 0u || raw == 0xFFFFFFFFu) {
            seg_offset = seg_offset + 1u;
            continue;
        }
        let cand = raw - 1u;
        // Same-segment bucket may hold a position > p (if a younger thread
        // raced and lost the atomicMin to an even-younger one); skip those.
        if (cand >= p) {
            seg_offset = seg_offset + 1u;
            continue;
        }
        let dist = p - cand;
        if (dist > params.window) { break; }

        // Hash collision check on first 3 bytes.
        if (read_byte(cand) == read_byte(p)
            && read_byte(cand + 1u) == read_byte(p + 1u)
            && read_byte(cand + 2u) == read_byte(p + 2u)) {
            var len: u32 = 3u;
            loop {
                if (len >= params.max_match) { break; }
                if (p + len >= params.input_len) { break; }
                if (read_byte(cand + len) != read_byte(p + len)) { break; }
                len = len + 1u;
            }
            if (len > best_len) {
                best_len = len;
                best_dist = dist;
            }
        }
        seg_offset = seg_offset + 1u;
    }

    if (best_len == 0u) {
        tokens[p] = read_byte(p);
    } else {
        tokens[p] = (best_len << 16u) | best_dist;
    }
}
