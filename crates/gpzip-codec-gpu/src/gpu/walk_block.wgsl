// Block-parallel strict-greedy LZ77 walk.
//
// 3-pass design that turns the inherently-serial walk into mostly
// parallel work + one short serial chain through blocks:
//
// Pass 1 (block_summary, parallel × N threads):
//   For each block b and each possible entry e ∈ [0, W) within the
//   block, pre-compute "if a walk entered the block at relative
//   position e, where would it exit?" + how many tokens it'd emit.
//   Threads are independent: workgroup b, thread e walks from
//   block_start + e until it crosses block_end.
//
// Pass 2 (chain_blocks, serial × N/W iters, single thread):
//   Trace from cur = 0 through blocks: block_idx = cur/W, look up
//   exit_table[block][cur%W], advance cur, record
//   actual_entry[block] (= 1 + entry, with 0 meaning "unvisited").
//   Also computes prefix-sum of count_table[block][actual_entry] to
//   give each visited block its output base offset.
//
// Pass 3 (block_emit, parallel × N/W workgroups, 1 thread each):
//   Each visited block walks from its actual_entry, writing tokens
//   into walked[token_offsets[block]..]. Other blocks early-exit.
//
// Output is the walked, non-overlapping token stream — same format
// and contract as the C-1 serial walk shader, just (hopefully) much
// faster. Result must match `host_strict_greedy` byte-for-byte; tests
// in walk.rs assert this.

struct Params {
    n_positions: u32,
    n_blocks: u32,
    block_size: u32,  // W; matched to workgroup_size in pass 1
}

@group(0) @binding(0) var<storage, read>       per_position:  array<u32>;
@group(0) @binding(1) var<storage, read_write> exit_table:    array<u32>;  // B * W
@group(0) @binding(2) var<storage, read_write> count_table:   array<u32>;  // B * W
@group(0) @binding(3) var<storage, read_write> actual_entry:  array<u32>;  // B (0 = unvisited)
@group(0) @binding(4) var<storage, read_write> token_offsets: array<u32>;  // B
@group(0) @binding(5) var<storage, read_write> walked:        array<u32>;
@group(0) @binding(6) var<storage, read_write> walked_count:  atomic<u32>;
@group(0) @binding(7) var<uniform>             params:        Params;

// Pass 1: per-block, per-entry pre-computed exits + counts.
// Workgroup size MUST match block_size (W=128).
@compute @workgroup_size(128)
fn block_summary(@builtin(workgroup_id) wid: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>) {
    let b = wid.x;
    let e = lid.x;
    let block_start = b * params.block_size;
    let block_end = min(block_start + params.block_size, params.n_positions);

    let idx = b * params.block_size + e;
    if (block_start + e >= params.n_positions) {
        // Beyond input — sentinel exit so the chain terminates cleanly.
        exit_table[idx] = params.block_size;
        count_table[idx] = 0u;
        return;
    }

    var cur: u32 = block_start + e;
    var count: u32 = 0u;
    loop {
        if (cur >= block_end) { break; }
        let t = per_position[cur];
        count = count + 1u;
        let len = t >> 16u;
        if (len == 0u) {
            cur = cur + 1u;
        } else {
            cur = cur + len;
        }
    }

    // exit_table holds the exit position relative to block_start.
    // It can equal or exceed block_size if the last token's back-ref
    // jumped past the block boundary.
    exit_table[idx] = cur - block_start;
    count_table[idx] = count;
}

// Pass 2: serial walk through blocks, ~N/W iterations.
// Single thread. Computes actual_entry[b] (1-based; 0 = unvisited)
// and the per-block exclusive prefix of token counts.
@compute @workgroup_size(1)
fn chain_blocks() {
    var cur: u32 = 0u;
    var token_off: u32 = 0u;

    var b: u32 = 0u;
    loop {
        if (b >= params.n_blocks) { break; }
        let block_start = b * params.block_size;

        if (cur >= params.n_positions) {
            actual_entry[b] = 0u;
            token_offsets[b] = token_off;
            b = b + 1u;
            continue;
        }
        if (cur < block_start) {
            // Walk has skipped past this block (a previous block's
            // back-ref jumped over it).
            actual_entry[b] = 0u;
            token_offsets[b] = token_off;
            b = b + 1u;
            continue;
        }
        if (cur >= block_start + params.block_size) {
            // Shouldn't happen given monotone progression and the
            // skip-past check above, but guard for safety.
            actual_entry[b] = 0u;
            token_offsets[b] = token_off;
            b = b + 1u;
            continue;
        }

        let e = cur - block_start;
        actual_entry[b] = e + 1u;
        token_offsets[b] = token_off;
        let idx = b * params.block_size + e;
        token_off = token_off + count_table[idx];
        cur = block_start + exit_table[idx];

        b = b + 1u;
    }
    atomicStore(&walked_count, token_off);
}

// Pass 3: each visited block walks from its actual_entry, emits
// tokens to walked[token_offsets[b]..]. One workgroup per block,
// one thread per workgroup (the rest of the entries are unused
// because only one entry is the real one for this block).
@compute @workgroup_size(1)
fn block_emit(@builtin(workgroup_id) wid: vec3<u32>) {
    let b = wid.x;
    if (b >= params.n_blocks) { return; }
    let entry_plus1 = actual_entry[b];
    if (entry_plus1 == 0u) { return; }
    let e = entry_plus1 - 1u;
    let block_start = b * params.block_size;
    let block_end = min(block_start + params.block_size, params.n_positions);
    let out_base = token_offsets[b];

    var cur: u32 = block_start + e;
    var out_idx: u32 = 0u;
    loop {
        if (cur >= block_end) { break; }
        let t = per_position[cur];
        walked[out_base + out_idx] = t;
        out_idx = out_idx + 1u;
        let len = t >> 16u;
        if (len == 0u) {
            cur = cur + 1u;
        } else {
            cur = cur + len;
        }
    }
}
