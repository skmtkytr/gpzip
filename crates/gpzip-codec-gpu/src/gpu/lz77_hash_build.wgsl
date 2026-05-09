// Phase 1 of segmented-hash LZ77.
//
// Each input position p writes (p+1) into seg_table[hash(p)][p >> seg_log2]
// via atomicMin — keeps the OLDEST position per (hash, segment) bucket.
//
// Why segmentation: a parallel hash chain on GPU loses position ordering
// (workgroups race at the head), so chain walks cant reliably find a
// *close* prior position. Segmenting by `p >> seg_log2` gives every
// position p a guaranteed candidate in each prior segment within window —
// distance bounded by (segments_walked + 1) * seg_size. That's enough for
// match-find quality even though we get only one candidate per segment.
//
// p+1 is stored (not p) so 0 (the reset value) reads as "unused".

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
@group(0) @binding(1) var<storage, read_write> seg_table: array<atomic<u32>>;
@group(0) @binding(2) var<uniform>             params:    Params;

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
    // Store p+1 so the all-0xFF reset reads as "unused".
    atomicMin(&seg_table[h * params.num_segs + seg], p + 1u);
}
