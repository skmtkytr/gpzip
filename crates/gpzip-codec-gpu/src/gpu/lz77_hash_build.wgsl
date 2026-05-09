// Phase 1 of hash-chain LZ77 (replaces the previous K-way atomicMin bucket).
//
// One head pointer per hash bucket + a `next` array sized to the input. Each
// position p performs `atomicExchange(heads[h], p+1)` to install itself at
// the chain head; the previous head value is recorded in `next[p]`. The
// chain at any bucket is a linked list of all positions with that 3-byte
// hash, ordered newest-first.
//
// p+1 is stored (not p) so head=0 means "empty bucket" and next[p]=0 means
// "end of chain". This frees us from any sentinel like 0xffffffff.

struct Params {
    input_len: u32,
    hash_bits: u32,
    window: u32,
    min_match: u32,
    max_match: u32,
    max_chain: u32,
}

@group(0) @binding(0) var<storage, read>       input_buf:  array<u32>;
@group(0) @binding(1) var<storage, read_write> heads:      array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> next_buf:   array<u32>;
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
    // Insert at head of chain. atomicExchange returns the previous head,
    // which becomes our `next` pointer. Race-safe: each thread's swap is
    // serialized at the head, and only this thread writes its own next[p].
    let prev = atomicExchange(&heads[h], p + 1u);
    next_buf[p] = prev;
}
