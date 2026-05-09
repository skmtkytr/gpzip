// Phase 1 of hash-table LZ77.
//
// Each thread writes its position into one of K sub-slots in its 3-byte
// hash bucket. The sub-slot is selected by `p % K`, so positions sharing a
// hash bucket spread across all K sub-slots and only collide when their
// hashes match AND their `p % K` values match. atomicMin per sub-slot
// keeps the oldest writer in each — gives the host K candidate prior
// positions to choose from in lookup.
//
// p+1 is stored (not p) so the all-ones initial value reads as "unused".

struct Params {
    input_len: u32,
    hash_bits: u32,
    window: u32,
    min_match: u32,
    max_match: u32,
    chain_k: u32,
}

@group(0) @binding(0) var<storage, read>       input_buf:  array<u32>;
@group(0) @binding(1) var<storage, read_write> hash_table: array<atomic<u32>>;
@group(0) @binding(2) var<uniform>             params:     Params;

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
    let sub = p % params.chain_k;
    atomicMin(&hash_table[h * params.chain_k + sub], p + 1u);
}
