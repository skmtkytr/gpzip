// GPU-side DEFLATE bitstream emission.
//
// Each "atom" is a single (bit_offset, value, n_bits) write into the
// output bitstream. The host pre-computes the atom list — laying out the
// block header, the per-token Huffman codes + extra bits, and the EOB —
// so the GPU only has to do the parallel atomicOr placement.
//
// One thread per atom. Each atom touches 1 or 2 u32 words in the output
// buffer (since n_bits ≤ 24 and bit_offset is arbitrary). Concurrent
// writes to the same word are merged via atomicOr — safe because every
// bit position is written by exactly one atom.

struct Atom {
    bit_offset: u32,
    value: u32,    // bits 0..n_bits-1 hold the data, higher bits are zero
    n_bits: u32,   // ≤ 24
}

struct Params {
    n_atoms: u32,
}

@group(0) @binding(0) var<storage, read>       atoms:  array<Atom>;
@group(0) @binding(1) var<storage, read_write> output: array<atomic<u32>>;
@group(0) @binding(2) var<uniform>             params: Params;

@compute @workgroup_size(256)
fn emit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_atoms) { return; }
    let a = atoms[i];
    let bit_offset = a.bit_offset;
    let value      = a.value;
    let n_bits     = a.n_bits;

    let word_idx    = bit_offset >> 5u;
    let bit_in_word = bit_offset & 31u;

    // Bits that fit in the first word.
    let space_in_word = 32u - bit_in_word;
    let low_bits = min(n_bits, space_in_word);
    let low_mask = (1u << low_bits) - 1u;
    let low_val  = (value & low_mask) << bit_in_word;
    atomicOr(&output[word_idx], low_val);

    // Spillover into the next word, if any.
    if (n_bits > low_bits) {
        let high_bits = n_bits - low_bits;
        let high_val  = value >> low_bits;
        atomicOr(&output[word_idx + 1u], high_val);
    }
}
