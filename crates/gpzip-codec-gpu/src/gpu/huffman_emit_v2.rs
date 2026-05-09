//! D-2: full-GPU fixed-Huffman DEFLATE emission.
//!
//! Where D-1 had the host build a pre-laid-out atom list and the GPU
//! only do the parallel `atomicOr` placement, D-2 keeps everything
//! GPU-side after the token upload:
//!
//! 1. **compute_and_local_scan** — per-token compute of bit length
//!    using uploaded Huffman / length-symbol / distance-symbol LUTs,
//!    then a workgroup-local Hillis–Steele scan into per-token
//!    exclusive-prefix offsets. Each workgroup also writes its
//!    inclusive total to `workgroup_totals[wg_id]`.
//!
//! 2. **scan_totals** — single-workgroup scan of the (≤ WG_SIZE)
//!    workgroup_totals into `workgroup_bases[wg_id]` (exclusive
//!    prefix), bridging the per-workgroup scans into a global one.
//!
//! 3. **emit** — per-token thread reads its global bit offset
//!    (= `workgroup_bases[wid] + per_token_offset[i] + 3` for the
//!    block header) and `atomicOr`s its 1–4 emissions. The leading
//!    thread of workgroup 0 emits the block header; the first thread
//!    past the last token emits EOB.
//!
//! Status: PoC, fixed-Huffman only. Production swap requires D-3
//! (dynamic Huffman header on GPU). Output bytes are bit-identical to
//! `deflate::encode_fixed_block` and the D-1 emit shader on the same
//! token stream — verified by round-trip and direct `assert_eq!`
//! against D-1 output in the bench.

#![allow(dead_code)]

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::context::GpuContext;
use super::lz77::Token;

const SHADER: &str = include_str!("huffman_emit_v2.wgsl");
const WG_SIZE: u32 = 256;

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Params {
    n_tokens: u32,
    n_workgroups: u32,
    _pad: [u32; 2],
}

pub struct HuffmanEmitV2Pipeline {
    ctx: Arc<GpuContext>,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_compute: wgpu::ComputePipeline,
    pipeline_scan: wgpu::ComputePipeline,
    pipeline_emit: wgpu::ComputePipeline,
    // Pre-uploaded fixed-Huffman LUT buffers — built once at pipeline
    // creation and reused across every emit call.
    buf_lit_lens: wgpu::Buffer,
    buf_lit_codes_pre: wgpu::Buffer,
    buf_dist_lens: wgpu::Buffer,
    buf_dist_codes_pre: wgpu::Buffer,
    buf_len_lut: wgpu::Buffer,
    buf_dist_lut_lo: wgpu::Buffer,
    buf_dist_lut_hi: wgpu::Buffer,
}

impl HuffmanEmitV2Pipeline {
    pub fn new(ctx: Arc<GpuContext>) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpzip-huffman-emit-v2-shader"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });

        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpzip-huffman-emit-v2-bgl"),
                entries: &[
                    storage_entry(0, true),   // tokens
                    storage_entry(1, true),   // lit_lens
                    storage_entry(2, true),   // lit_codes_pre
                    storage_entry(3, true),   // dist_lens
                    storage_entry(4, true),   // dist_codes_pre
                    storage_entry(5, true),   // len_lut
                    storage_entry(6, true),   // dist_lut_lo
                    storage_entry(7, true),   // dist_lut_hi
                    storage_entry(8, false),  // per_token_offset
                    storage_entry(9, false),  // workgroup_totals
                    storage_entry(10, false), // workgroup_bases
                    storage_entry(11, false), // output (atomic)
                    uniform_entry(12),
                ],
            });
        let pl_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpzip-huffman-emit-v2-pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
        let make_pipeline = |entry: &str, label: &str| {
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pl_layout),
                    module: &module,
                    entry_point: entry,
                })
        };
        let pipeline_compute = make_pipeline("compute_and_local_scan", "v2-compute-scan");
        let pipeline_scan = make_pipeline("scan_totals", "v2-scan-totals");
        let pipeline_emit = make_pipeline("emit", "v2-emit");

        // Upload the Huffman/symbol LUTs once. They never change for
        // fixed Huffman.
        let lit_lens = build_fixed_lit_lens();
        let lit_codes_pre = build_fixed_lit_codes_pre();
        let dist_lens = build_fixed_dist_lens();
        let dist_codes_pre = build_fixed_dist_codes_pre();
        let len_lut = build_len_lut();
        let dist_lut_lo = build_dist_lut_lo();
        let dist_lut_hi = build_dist_lut_hi();

        let mk_storage = |name: &str, data: &[u32]| {
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(name),
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        Self {
            buf_lit_lens: mk_storage("v2-lit-lens", &lit_lens),
            buf_lit_codes_pre: mk_storage("v2-lit-codes-pre", &lit_codes_pre),
            buf_dist_lens: mk_storage("v2-dist-lens", &dist_lens),
            buf_dist_codes_pre: mk_storage("v2-dist-codes-pre", &dist_codes_pre),
            buf_len_lut: mk_storage("v2-len-lut", &len_lut),
            buf_dist_lut_lo: mk_storage("v2-dist-lut-lo", &dist_lut_lo),
            buf_dist_lut_hi: mk_storage("v2-dist-lut-hi", &dist_lut_hi),
            ctx,
            bind_group_layout: bgl,
            pipeline_compute,
            pipeline_scan,
            pipeline_emit,
        }
    }

    /// Encode `tokens` as a fixed-Huffman DEFLATE block on the GPU,
    /// computing bit lengths and bit offsets entirely on the GPU. Output
    /// is bit-identical to `encode_fixed_block` (host) and to D-1's
    /// `emit_fixed_block`.
    pub fn emit_fixed_block_v2(&self, tokens: &[Token]) -> Vec<u8> {
        // Empty token stream: just header + EOB. Build directly on host
        // — not worth a GPU dispatch for 12 bits.
        if tokens.is_empty() {
            return empty_fixed_block();
        }

        // Pack tokens to u32 (length<<16 | dist_or_byte).
        let packed: Vec<u32> = tokens
            .iter()
            .map(|t| (t.length << 16) | (t.distance & 0xFFFF))
            .collect();
        let n_tokens = packed.len() as u32;
        let n_workgroups = n_tokens.div_ceil(WG_SIZE);
        assert!(
            n_workgroups <= WG_SIZE,
            "v2 single-pass scan_totals only handles up to WG_SIZE workgroups; \
             n_tokens={} requires {} workgroups (max {})",
            n_tokens,
            n_workgroups,
            WG_SIZE
        );

        // Output upper bound: 3 (header) + n_tokens * 32 (worst-case
        // match emission) + 9 (EOB) bits, padded to u32 word boundary
        // plus one slack word for the spillover branch in `write_bits`.
        let max_bits = 3 + (n_tokens as u64) * 32 + 9;
        let max_words = (max_bits as usize).div_ceil(32) + 1;
        let max_bytes = max_words * 4;

        let buf_tokens = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("v2-tokens"),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buf_per_token_offset = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("v2-per-token-offset"),
            size: (n_tokens as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let buf_workgroup_totals = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("v2-workgroup-totals"),
            size: (n_workgroups as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let buf_workgroup_bases = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("v2-workgroup-bases"),
            size: (n_workgroups as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let buf_output = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("v2-output"),
            size: max_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_staging = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("v2-staging"),
            size: max_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = Params {
            n_tokens,
            n_workgroups,
            _pad: [0; 2],
        };
        let buf_params = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("v2-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("v2-bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    bge(0, &buf_tokens),
                    bge(1, &self.buf_lit_lens),
                    bge(2, &self.buf_lit_codes_pre),
                    bge(3, &self.buf_dist_lens),
                    bge(4, &self.buf_dist_codes_pre),
                    bge(5, &self.buf_len_lut),
                    bge(6, &self.buf_dist_lut_lo),
                    bge(7, &self.buf_dist_lut_hi),
                    bge(8, &buf_per_token_offset),
                    bge(9, &buf_workgroup_totals),
                    bge(10, &buf_workgroup_bases),
                    bge(11, &buf_output),
                    bge(12, &buf_params),
                ],
            });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("v2-enc"),
            });
        encoder.clear_buffer(&buf_output, 0, None);
        // Pass 1: compute + local scan
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("v2-compute-scan-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_compute);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n_workgroups, 1, 1);
        }
        // Pass 2: scan workgroup totals (single workgroup, ≤ WG_SIZE entries)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("v2-scan-totals-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_scan);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        // Pass 3: emit (one extra thread for EOB)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("v2-emit-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_emit);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((n_tokens + 1).div_ceil(WG_SIZE), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&buf_output, 0, &buf_staging, 0, max_bytes as u64);
        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        let slice = buf_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("v2 buffer map failed");
        let view = slice.get_mapped_range();
        let raw = view.to_vec();
        drop(view);
        buf_staging.unmap();

        // Compute total bits on host (cheap; same arithmetic the GPU
        // did, no readback dance). Header + per-token + EOB.
        let total_bits = total_bits_for_fixed(tokens);
        let n_bytes = (total_bits as usize).div_ceil(8);
        raw[..n_bytes].to_vec()
    }
}

// ============================================================
// LUT builders + host-side fixed Huffman tables
// ============================================================

fn fixed_litlen(symbol: u32) -> (u32, u32) {
    match symbol {
        0..=143 => (0b0011_0000 + symbol, 8),
        144..=255 => (0b1_1001_0000 + (symbol - 144), 9),
        256..=279 => (symbol - 256, 7),
        280..=287 => (0b1100_0000 + (symbol - 280), 8),
        _ => unreachable!(),
    }
}

fn fixed_dist(symbol: u32) -> (u32, u32) {
    (symbol, 5)
}

#[inline]
fn rev_bits(code: u32, bits: u32) -> u32 {
    if bits == 0 {
        0
    } else {
        code.reverse_bits() >> (32 - bits)
    }
}

fn build_fixed_lit_lens() -> Vec<u32> {
    (0..288u32).map(|s| fixed_litlen(s).1).collect()
}
fn build_fixed_lit_codes_pre() -> Vec<u32> {
    (0..288u32)
        .map(|s| {
            let (c, b) = fixed_litlen(s);
            rev_bits(c, b)
        })
        .collect()
}
fn build_fixed_dist_lens() -> Vec<u32> {
    (0..30u32).map(|s| fixed_dist(s).1).collect()
}
fn build_fixed_dist_codes_pre() -> Vec<u32> {
    (0..30u32)
        .map(|s| {
            let (c, b) = fixed_dist(s);
            rev_bits(c, b)
        })
        .collect()
}

/// Length 3..=258 → packed (sym-257) | (extra<<8) | (base<<16). Length
/// indices 0..3 are unused (length must be ≥ 3) and stay zero.
fn build_len_lut() -> Vec<u32> {
    const ROWS: &[(u32, u32, u32)] = &[
        (3, 0, 257),
        (4, 0, 258),
        (5, 0, 259),
        (6, 0, 260),
        (7, 0, 261),
        (8, 0, 262),
        (9, 0, 263),
        (10, 0, 264),
        (11, 1, 265),
        (13, 1, 266),
        (15, 1, 267),
        (17, 1, 268),
        (19, 2, 269),
        (23, 2, 270),
        (27, 2, 271),
        (31, 2, 272),
        (35, 3, 273),
        (43, 3, 274),
        (51, 3, 275),
        (59, 3, 276),
        (67, 4, 277),
        (83, 4, 278),
        (99, 4, 279),
        (115, 4, 280),
        (131, 5, 281),
        (163, 5, 282),
        (195, 5, 283),
        (227, 5, 284),
        (258, 0, 285),
    ];
    let mut lut = vec![0u32; 259];
    for (idx, &(base, extra, sym)) in ROWS.iter().enumerate() {
        let next = if idx + 1 < ROWS.len() {
            ROWS[idx + 1].0
        } else {
            259
        };
        for len in base..next {
            lut[len as usize] = (sym - 257) | (extra << 8) | (base << 16);
        }
    }
    lut
}

const DIST_ROWS: &[(u32, u32, u32)] = &[
    (1, 0, 0),
    (2, 0, 1),
    (3, 0, 2),
    (4, 0, 3),
    (5, 1, 4),
    (7, 1, 5),
    (9, 2, 6),
    (13, 2, 7),
    (17, 3, 8),
    (25, 3, 9),
    (33, 4, 10),
    (49, 4, 11),
    (65, 5, 12),
    (97, 5, 13),
    (129, 6, 14),
    (193, 6, 15),
    (257, 7, 16),
    (385, 7, 17),
    (513, 8, 18),
    (769, 8, 19),
    (1025, 9, 20),
    (1537, 9, 21),
    (2049, 10, 22),
    (3073, 10, 23),
    (4097, 11, 24),
    (6145, 11, 25),
    (8193, 12, 26),
    (12289, 12, 27),
    (16385, 13, 28),
    (24577, 13, 29),
];

/// Distance 1..=256 → packed sym | (extra<<8) | (base<<16). Index 0 is
/// unused (distance must be ≥ 1).
fn build_dist_lut_lo() -> Vec<u32> {
    let mut lut = vec![0u32; 257];
    for (idx, &(base, extra, sym)) in DIST_ROWS.iter().enumerate() {
        if base > 256 {
            break;
        }
        let next = if idx + 1 < DIST_ROWS.len() {
            DIST_ROWS[idx + 1].0
        } else {
            32769
        };
        let end = next.min(257);
        for d in base..end {
            lut[d as usize] = sym | (extra << 8) | (base << 16);
        }
    }
    lut
}

/// Distance > 256: indexed by (d-1)>>7, gives the same packed
/// representation. Index space is 256 entries (covers up to d = 32768).
fn build_dist_lut_hi() -> Vec<u32> {
    let mut lut = vec![0u32; 256];
    for (idx, &(base, extra, sym)) in DIST_ROWS.iter().enumerate() {
        if base < 257 {
            continue;
        }
        let next = if idx + 1 < DIST_ROWS.len() {
            DIST_ROWS[idx + 1].0
        } else {
            32769
        };
        for d in base..next {
            let h_idx = ((d - 1) >> 7) as usize;
            if h_idx < lut.len() {
                lut[h_idx] = sym | (extra << 8) | (base << 16);
            }
        }
    }
    lut
}

/// Compute total DEFLATE bits for a fixed-Huffman block over `tokens`.
/// Mirrors the GPU's per-token compute + the host's header/EOB framing.
fn total_bits_for_fixed(tokens: &[Token]) -> u64 {
    let mut bits: u64 = 3; // BFINAL + BTYPE
    for t in tokens {
        if t.is_literal() {
            let (_, b) = fixed_litlen(t.distance);
            bits += b as u64;
        } else {
            let (lcode, lextra, _lval) = host_length_code(t.length);
            let (_, lb) = fixed_litlen(lcode);
            bits += (lb + lextra) as u64;
            let (_dcode, dextra, _dval) = host_distance_code(t.distance);
            bits += (5 + dextra) as u64;
        }
    }
    let (_, eob_bits) = fixed_litlen(256);
    bits + eob_bits as u64
}

fn host_length_code(length: u32) -> (u32, u32, u32) {
    debug_assert!((3..=258).contains(&length));
    let mut idx = 0;
    for (i, row) in LEN_ROWS.iter().enumerate() {
        if row.0 <= length {
            idx = i;
        } else {
            break;
        }
    }
    let (base, extra, code) = LEN_ROWS[idx];
    (code, extra, length - base)
}

fn host_distance_code(distance: u32) -> (u32, u32, u32) {
    debug_assert!((1..=32768).contains(&distance));
    let mut idx = 0;
    for (i, row) in DIST_ROWS.iter().enumerate() {
        if row.0 <= distance {
            idx = i;
        } else {
            break;
        }
    }
    let (base, extra, code) = DIST_ROWS[idx];
    (code, extra, distance - base)
}

const LEN_ROWS: &[(u32, u32, u32)] = &[
    (3, 0, 257),
    (4, 0, 258),
    (5, 0, 259),
    (6, 0, 260),
    (7, 0, 261),
    (8, 0, 262),
    (9, 0, 263),
    (10, 0, 264),
    (11, 1, 265),
    (13, 1, 266),
    (15, 1, 267),
    (17, 1, 268),
    (19, 2, 269),
    (23, 2, 270),
    (27, 2, 271),
    (31, 2, 272),
    (35, 3, 273),
    (43, 3, 274),
    (51, 3, 275),
    (59, 3, 276),
    (67, 4, 277),
    (83, 4, 278),
    (99, 4, 279),
    (115, 4, 280),
    (131, 5, 281),
    (163, 5, 282),
    (195, 5, 283),
    (227, 5, 284),
    (258, 0, 285),
];

/// Fixed-Huffman empty block: header (BFINAL=1, BTYPE=01, 0b011 LSB)
/// then EOB (symbol 256, fixed code = 7-bit `0000000` = 0). Total 10
/// bits → 2 bytes. Used as a fast path when there are no tokens.
fn empty_fixed_block() -> Vec<u8> {
    let mut out = vec![0u8; 2];
    // Bit 0..2: 0b011
    out[0] = 0b011;
    // EOB code is 0 in 7 bits, written LSB-first → adds 7 zero bits.
    // Already zero; nothing to do.
    out
}

// ============================================================
// Bind-group helpers
// ============================================================

fn bge<'a>(binding: u32, buf: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::lz77::Token;
    use super::*;
    use flate2::read::DeflateDecoder;
    use std::io::Read;

    fn try_pipeline() -> Option<HuffmanEmitV2Pipeline> {
        let ctx = GpuContext::try_init().ok()?;
        Some(HuffmanEmitV2Pipeline::new(Arc::new(ctx)))
    }

    fn reconstruct(tokens: &[Token]) -> Vec<u8> {
        let mut out = Vec::new();
        for t in tokens {
            if t.is_literal() {
                out.push(t.distance as u8);
            } else {
                let start = out.len() - t.distance as usize;
                for i in 0..t.length as usize {
                    let b = out[start + i];
                    out.push(b);
                }
            }
        }
        out
    }

    fn round_trip(tokens: &[Token]) {
        let Some(p) = try_pipeline() else {
            eprintln!("no GPU — skipping");
            return;
        };
        let deflate_bytes = p.emit_fixed_block_v2(tokens);
        let mut decoded = Vec::new();
        DeflateDecoder::new(&deflate_bytes[..])
            .read_to_end(&mut decoded)
            .expect("flate2 should decode v2 GPU-emitted DEFLATE");
        let expected = reconstruct(tokens);
        assert_eq!(
            decoded,
            expected,
            "v2 GPU round-trip mismatch on {} tokens",
            tokens.len()
        );
    }

    #[test]
    fn v2_literals_only() {
        let tokens: Vec<Token> = (0..64u8).map(Token::literal).collect();
        round_trip(&tokens);
    }

    #[test]
    fn v2_single_back_ref() {
        let mut tokens: Vec<Token> = b"hello, ".iter().map(|&b| Token::literal(b)).collect();
        tokens.push(Token::back_ref(5, 7));
        round_trip(&tokens);
    }

    #[test]
    fn v2_many_back_refs_with_extra_bits() {
        let mut tokens: Vec<Token> = Vec::new();
        for i in 0..512u32 {
            tokens.push(Token::literal((i & 0xff) as u8));
        }
        for &(len, dist) in &[(3, 1), (11, 5), (35, 17), (131, 257)] {
            tokens.push(Token::back_ref(len, dist));
        }
        round_trip(&tokens);
    }

    #[test]
    fn v2_long_random_literal_stream() {
        let tokens: Vec<Token> = (0..2048u32)
            .map(|i| Token::literal((i.wrapping_mul(2654435761) as u32 & 0xff) as u8))
            .collect();
        round_trip(&tokens);
    }

    #[test]
    fn v2_empty_token_stream() {
        let Some(p) = try_pipeline() else { return };
        let deflate_bytes = p.emit_fixed_block_v2(&[]);
        let mut decoded = Vec::new();
        DeflateDecoder::new(&deflate_bytes[..])
            .read_to_end(&mut decoded)
            .expect("v2 empty fixed block should decode");
        assert!(decoded.is_empty());
    }

    /// Cross multi-workgroup boundary (>256 tokens) so the inter-workgroup
    /// scan is exercised, not just the local one.
    #[test]
    fn v2_multi_workgroup() {
        let tokens: Vec<Token> = (0..4096u32)
            .map(|i| Token::literal((i & 0xff) as u8))
            .collect();
        round_trip(&tokens);
    }
}
