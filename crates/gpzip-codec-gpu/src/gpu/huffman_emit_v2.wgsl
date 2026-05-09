// D-2: full-GPU fixed-Huffman DEFLATE emission.
//
// Three compute passes share these bindings (some passes use only a
// subset; the shared layout keeps the host code simple — one bind group
// reused across passes).
//
//   tokens[i]            packed (length<<16 | dist_or_byte)
//   lit_lens[s]          Huffman code length for lit/len symbol s (288 entries)
//   lit_codes_pre[s]     pre-reversed Huffman code (LSB-first packing)
//   dist_lens[s]         Huffman code length for distance symbol s (30 entries)
//   dist_codes_pre[s]    pre-reversed distance code
//   len_lut[len]         packed (sym-257)|(extra<<8)|(base<<16) for length 3..258
//   dist_lut_lo[d]       packed sym|(extra<<8)|(base<<16) for distance 1..256
//   dist_lut_hi[(d-1)>>7] same packing for distance 257..32768
//
//   per_token_offset[i]  exclusive bit offset within block body (output of pass 1)
//   workgroup_totals[w]  inclusive sum of bits in workgroup w (output of pass 1)
//   workgroup_bases[w]   exclusive prefix of workgroup_totals (output of pass 2)
//   output[]             u32 array, the atomicOr-built bitstream (output of pass 3)
//
// Block framing:
//   - Pass 3 writes the 3-bit block header (BFINAL=1, BTYPE=01) at bit 0
//     by thread 0 of workgroup 0.
//   - Pass 3 writes the EOB code right after the last token's bits.
//   - Token bit offsets are biased by 3 (the header) — done in the emit
//     pass, not in the scan, so the scan stays clean.

struct Params {
    n_tokens: u32,
    n_workgroups: u32,
    // Bit offset where the per-token emissions start in `output[]`. The
    // host pre-writes the block header (BFINAL/BTYPE for fixed, plus the
    // dynamic-Huffman header for BTYPE=10) into output[0..ceil(header_bit_count/8)]
    // before the dispatch. The emit shader places token bits starting at
    // this offset and the EOB right after them.
    header_bit_count: u32,
}

@group(0) @binding(0)  var<storage, read>       tokens:           array<u32>;
@group(0) @binding(1)  var<storage, read>       lit_lens:         array<u32>;
@group(0) @binding(2)  var<storage, read>       lit_codes_pre:    array<u32>;
@group(0) @binding(3)  var<storage, read>       dist_lens:        array<u32>;
@group(0) @binding(4)  var<storage, read>       dist_codes_pre:   array<u32>;
@group(0) @binding(5)  var<storage, read>       len_lut:          array<u32>;
@group(0) @binding(6)  var<storage, read>       dist_lut_lo:      array<u32>;
@group(0) @binding(7)  var<storage, read>       dist_lut_hi:      array<u32>;
@group(0) @binding(8)  var<storage, read_write> per_token_offset: array<u32>;
@group(0) @binding(9)  var<storage, read_write> workgroup_totals: array<u32>;
@group(0) @binding(10) var<storage, read_write> workgroup_bases:  array<u32>;
@group(0) @binding(11) var<storage, read_write> output:           array<atomic<u32>>;
@group(0) @binding(12) var<uniform>             params:           Params;

const WG_SIZE: u32 = 256u;

fn token_bits(tok: u32) -> u32 {
    let len = tok >> 16u;
    let dist_or_byte = tok & 0xFFFFu;
    if (len == 0u) {
        return lit_lens[dist_or_byte];
    }
    let lp = len_lut[len];
    let len_sym = (lp & 0xFFu) + 257u;
    let len_extra = (lp >> 8u) & 0xFFu;
    let lcb = lit_lens[len_sym];

    var dp: u32;
    if (dist_or_byte <= 256u) {
        dp = dist_lut_lo[dist_or_byte];
    } else {
        dp = dist_lut_hi[(dist_or_byte - 1u) >> 7u];
    }
    let dist_sym = dp & 0xFFu;
    let dist_extra = (dp >> 8u) & 0xFFu;
    let dcb = dist_lens[dist_sym];

    return lcb + len_extra + dcb + dist_extra;
}

// ============================================================
// Pass 1: per-token bit length + workgroup-local scan
// Output: per_token_offset[] (exclusive prefix within workgroup),
//         workgroup_totals[w] (inclusive total of workgroup w)
// ============================================================

var<workgroup> shared_bits: array<u32, WG_SIZE>;

@compute @workgroup_size(256)
fn compute_and_local_scan(@builtin(global_invocation_id) gid: vec3<u32>,
                          @builtin(local_invocation_id) lid: vec3<u32>,
                          @builtin(workgroup_id) wid: vec3<u32>) {
    let i = gid.x;
    let l = lid.x;

    var bits: u32 = 0u;
    if (i < params.n_tokens) {
        bits = token_bits(tokens[i]);
    }
    shared_bits[l] = bits;
    workgroupBarrier();

    // Hillis–Steele inclusive scan with two barriers per step (single-buffer).
    var step: u32 = 1u;
    loop {
        if (step >= WG_SIZE) { break; }
        var addend: u32 = 0u;
        if (l >= step) { addend = shared_bits[l - step]; }
        workgroupBarrier();
        shared_bits[l] = shared_bits[l] + addend;
        workgroupBarrier();
        step = step * 2u;
    }

    let inclusive = shared_bits[l];
    let exclusive = inclusive - bits;

    if (i < params.n_tokens) {
        per_token_offset[i] = exclusive;
    }
    if (l == WG_SIZE - 1u) {
        workgroup_totals[wid.x] = inclusive;
    }
}

// ============================================================
// Pass 2: scan workgroup_totals into workgroup_bases (exclusive)
// Single workgroup of WG_SIZE threads; assumes n_workgroups <= WG_SIZE
// (i.e., n_tokens <= 65 536, well above our 32 KiB chunk's max token
// count which is 32K positions).
// ============================================================

var<workgroup> shared_totals: array<u32, WG_SIZE>;

@compute @workgroup_size(256)
fn scan_totals(@builtin(local_invocation_id) lid: vec3<u32>) {
    let l = lid.x;

    var v: u32 = 0u;
    if (l < params.n_workgroups) {
        v = workgroup_totals[l];
    }
    shared_totals[l] = v;
    workgroupBarrier();

    var step: u32 = 1u;
    loop {
        if (step >= WG_SIZE) { break; }
        var addend: u32 = 0u;
        if (l >= step) { addend = shared_totals[l - step]; }
        workgroupBarrier();
        shared_totals[l] = shared_totals[l] + addend;
        workgroupBarrier();
        step = step * 2u;
    }

    let inclusive = shared_totals[l];
    let exclusive = inclusive - v;
    if (l < params.n_workgroups) {
        workgroup_bases[l] = exclusive;
    }
}

// ============================================================
// Pass 3: emit
// Each token thread looks up its global bit offset (= per_token_offset
// + workgroup_bases[wid] + 3 for block header) and atomicOrs its
// 1-4 bit-packed emissions.
// Thread 0 of workgroup 0 also emits the 3-bit block header.
// One designated thread emits the EOB after the last token.
// ============================================================

fn write_bits(bit_offset: u32, value: u32, n_bits: u32) {
    if (n_bits == 0u) { return; }
    let word_idx = bit_offset >> 5u;
    let bit_in_word = bit_offset & 31u;
    let space_in_word = 32u - bit_in_word;
    let low_bits = min(n_bits, space_in_word);
    let low_mask = (1u << low_bits) - 1u;
    let low_val = (value & low_mask) << bit_in_word;
    atomicOr(&output[word_idx], low_val);
    if (n_bits > low_bits) {
        let high_bits = n_bits - low_bits;
        let high_val = value >> low_bits;
        atomicOr(&output[word_idx + 1u], high_val);
    }
}

@compute @workgroup_size(256)
fn emit(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = gid.x;
    let hdr = params.header_bit_count;

    if (i >= params.n_tokens) {
        // The first out-of-range thread (i == n_tokens) emits the EOB.
        // The workgroup containing the LAST token is `n_workgroups - 1`
        // — not `n_tokens / WG_SIZE`, which is one too high when
        // n_tokens is a multiple of WG_SIZE and would read OOB on the
        // workgroup_bases / workgroup_totals arrays. The bug went
        // unnoticed for fixed Huffman because the fixed EOB code is
        // all-zero (atomicOr with 0 is idempotent), but dynamic Huffman
        // has non-zero EOB bits and an OOB read returned 0, placing the
        // EOB at offset = hdr instead of past the body.
        if (i == params.n_tokens) {
            let last_wg = params.n_workgroups - 1u;
            let body_offset = workgroup_bases[last_wg];
            let bits_in_last_wg = workgroup_totals[last_wg];
            let global_after_tokens = hdr + body_offset + bits_in_last_wg;
            let eob_code = lit_codes_pre[256u];
            let eob_bits = lit_lens[256u];
            write_bits(global_after_tokens, eob_code, eob_bits);
        }
        return;
    }

    let tok = tokens[i];
    let local_off = per_token_offset[i];
    let wg_base = workgroup_bases[wid.x];
    var bit_off = hdr + wg_base + local_off;

    let len = tok >> 16u;
    let dist_or_byte = tok & 0xFFFFu;

    if (len == 0u) {
        let sym = dist_or_byte;
        let code = lit_codes_pre[sym];
        let bits = lit_lens[sym];
        write_bits(bit_off, code, bits);
    } else {
        let lp = len_lut[len];
        let len_sym = (lp & 0xFFu) + 257u;
        let len_extra = (lp >> 8u) & 0xFFu;
        let len_base = (lp >> 16u) & 0xFFFFu;
        let lcode = lit_codes_pre[len_sym];
        let lbits = lit_lens[len_sym];
        write_bits(bit_off, lcode, lbits);
        bit_off = bit_off + lbits;
        if (len_extra > 0u) {
            let extra_val = len - len_base;
            write_bits(bit_off, extra_val, len_extra);
            bit_off = bit_off + len_extra;
        }
        var dp: u32;
        if (dist_or_byte <= 256u) {
            dp = dist_lut_lo[dist_or_byte];
        } else {
            dp = dist_lut_hi[(dist_or_byte - 1u) >> 7u];
        }
        let dist_sym = dp & 0xFFu;
        let dist_extra = (dp >> 8u) & 0xFFu;
        let dist_base = (dp >> 16u) & 0xFFFFu;
        let dcode = dist_codes_pre[dist_sym];
        let dbits = dist_lens[dist_sym];
        write_bits(bit_off, dcode, dbits);
        bit_off = bit_off + dbits;
        if (dist_extra > 0u) {
            let extra_val = dist_or_byte - dist_base;
            write_bits(bit_off, extra_val, dist_extra);
        }
    }
}
