// Phase 1 of segmented-hash LZ77 with two candidates per (hash, segment).
//
// Each input position p writes (p+1) into BOTH:
//   seg_oldest[hash(p)][p >> seg_log2]  via atomicMin  → smallest p (earliest in segment)
//   seg_newest[hash(p)][p >> seg_log2]  via atomicMax  → largest  p (latest in segment)
//
// Two candidates per segment lets the lookup pick the closer one (smaller
// distance → shorter Huffman distance code), while still having a
// guaranteed in-segment candidate via atomicMin if the newest happens to
// be > p (race) or filtered out by the cand<p check.
//
// Sentinels: 0xFFFFFFFF for atomicMin (oldest), 0 for atomicMax (newest).
// p+1 is stored so the lookup can distinguish "unused" from p=0.

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
@group(0) @binding(1) var<storage, read_write> seg_oldest: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> seg_newest: array<atomic<u32>>;
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
fn build(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p + 2u >= params.input_len) { return; }
    let h = hash3(p);
    let seg = p >> params.seg_log2;
    let idx = h * params.num_segs + seg;
    atomicMin(&seg_oldest[idx], p + 1u);
    atomicMax(&seg_newest[idx], p + 1u);
}
