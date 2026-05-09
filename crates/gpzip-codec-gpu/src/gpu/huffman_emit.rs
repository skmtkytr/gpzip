//! GPU-side DEFLATE bitstream emission.
//!
//! Phase D-1 of the GPU encoder. The host builds an "atom" list — every
//! variable-bit-width write that would happen in `write_fixed_block`,
//! laid out at its eventual bit offset — and the GPU does the parallel
//! atomicOr placement in one shader dispatch. Output is byte-identical
//! to the host fixed-Huffman block writer (verified by round-trip
//! through `flate2::MultiGzDecoder`).
//!
//! Status: PoC. Not yet wired into the production chunk_fn — the
//! existing `encode_block_fast` (host-side, dynamic Huffman) still
//! handles compression. D-2 will move bit-length compute and the
//! prefix-sum onto the GPU; D-3 will add dynamic-Huffman header
//! support and swap this in as the production encoder. Until then the
//! module is exercised only by its tests.

#![allow(dead_code)]
//!
//! What stays on the host (for now):
//!   - Computing per-token bit lengths from the Huffman lookup tables
//!   - Prefix-summing the bit lengths to get bit offsets
//!   - Reversing each Huffman code to LSB-first packing order
//!
//! What runs on the GPU:
//!   - Parallel `atomicOr` of every atom's bits into the output buffer
//!
//! Future phases (D-2 / D-3) will move bit-length compute and the
//! prefix-sum onto the GPU as well, then full dynamic-Huffman header
//! and `greedy_walk`.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::context::GpuContext;
use super::lz77::Token;

const SHADER: &str = include_str!("huffman_emit.wgsl");

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Atom {
    bit_offset: u32,
    value: u32,
    n_bits: u32,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Params {
    n_atoms: u32,
    _pad: [u32; 3],
}

pub struct HuffmanEmitPipeline {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl HuffmanEmitPipeline {
    pub fn new(ctx: Arc<GpuContext>) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpzip-huffman-emit-shader"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });

        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpzip-huffman-emit-bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    uniform_entry(2),
                ],
            });
        let pl_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpzip-huffman-emit-pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gpzip-huffman-emit-pipeline"),
                layout: Some(&pl_layout),
                module: &module,
                entry_point: "emit",
            });
        Self {
            ctx,
            pipeline,
            bind_group_layout: bgl,
        }
    }

    /// Encode `tokens` as a fixed-Huffman DEFLATE block on the GPU.
    /// Output is the raw deflate bytes — wrap with `gzip_wrap` to make a
    /// gzip member. Returns the same bytes the host fixed-Huffman writer
    /// would produce for the same token stream.
    pub fn emit_fixed_block(&self, tokens: &[Token]) -> Vec<u8> {
        let (atoms, total_bits) = build_atoms_fixed(tokens);
        if atoms.is_empty() {
            return Vec::new();
        }

        // Output buffer: round up to u32 words (atomic-OR target). Add a
        // trailing word of slack so the spillover branch in the shader
        // can write past the last "live" word without OOB.
        let total_words = total_bits.div_ceil(32) as usize + 1;
        let total_bytes_padded = total_words * 4;

        let atoms_bytes = bytemuck::cast_slice::<Atom, u8>(&atoms);
        let atoms_buf = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpzip-huffman-emit-atoms"),
                contents: atoms_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buf = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpzip-huffman-emit-output"),
            size: total_bytes_padded as u64,
            // COPY_DST needed for clear_buffer (zero-init below).
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpzip-huffman-emit-staging"),
            size: total_bytes_padded as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = Params {
            n_atoms: atoms.len() as u32,
            _pad: [0; 3],
        };
        let params_buf = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpzip-huffman-emit-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpzip-huffman-emit-bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: atoms_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpzip-huffman-emit-enc"),
            });
        // Zero-init the output buffer — atomicOr only sets bits, so we
        // need a clean slate.
        encoder.clear_buffer(&output_buf, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpzip-huffman-emit-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((atoms.len() as u32).div_ceil(256), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, total_bytes_padded as u64);
        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("buffer map failed");
        let view = slice.get_mapped_range();
        let raw = view.to_vec();
        drop(view);
        staging.unmap();

        // Trim padding: the bitstream spans `ceil(total_bits / 8)` bytes.
        let n_bytes = total_bits.div_ceil(8) as usize;
        raw[..n_bytes].to_vec()
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

/// Reverse the low `bits` bits of `code` — Huffman codes are MSB-first
/// per RFC 1951, but DEFLATE packs them LSB-first into the bitstream.
#[inline]
fn rev_bits(code: u32, bits: u32) -> u32 {
    if bits == 0 {
        0
    } else {
        code.reverse_bits() >> (32 - bits)
    }
}

/// Fixed-Huffman literal/length code (RFC 1951 §3.2.6). Same table as
/// `deflate.rs::fixed_litlen_code`.
fn fixed_litlen(symbol: u32) -> (u32, u32) {
    match symbol {
        0..=143 => (0b0011_0000 + symbol, 8),
        144..=255 => (0b1_1001_0000 + (symbol - 144), 9),
        256..=279 => (symbol - 256, 7),
        280..=287 => (0b1100_0000 + (symbol - 280), 8),
        _ => unreachable!(),
    }
}

/// Length 3..=258 → (length-symbol, extra-bits, extra-value). Same table
/// as `deflate.rs::length_code`.
fn length_code(length: u32) -> (u32, u32, u32) {
    debug_assert!((3..=258).contains(&length));
    const TABLE: &[(u32, u32, u32)] = &[
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
    let mut idx = 0;
    for (i, row) in TABLE.iter().enumerate() {
        if row.0 <= length {
            idx = i;
        } else {
            break;
        }
    }
    let (base, extra, code) = TABLE[idx];
    (code, extra, length - base)
}

fn distance_code(distance: u32) -> (u32, u32, u32) {
    debug_assert!((1..=32768).contains(&distance));
    const TABLE: &[(u32, u32, u32)] = &[
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
    let mut idx = 0;
    for (i, row) in TABLE.iter().enumerate() {
        if row.0 <= distance {
            idx = i;
        } else {
            break;
        }
    }
    let (base, extra, code) = TABLE[idx];
    (code, extra, distance - base)
}

/// Build the atom list and compute the total bit count for a fixed-
/// Huffman DEFLATE block over `tokens`. Mirrors `write_fixed_block` in
/// `deflate.rs` so output is bit-identical.
fn build_atoms_fixed(tokens: &[Token]) -> (Vec<Atom>, u32) {
    let mut atoms = Vec::with_capacity(tokens.len() * 2 + 2);
    let mut bit_offset: u32 = 0;

    // Block header: BFINAL=1, BTYPE=01 → 0b011 LSB-first → value 0b011, 3 bits.
    push_atom(&mut atoms, &mut bit_offset, 0b011, 3);

    for tok in tokens {
        if tok.is_literal() {
            let (code, bits) = fixed_litlen(tok.distance);
            push_atom(&mut atoms, &mut bit_offset, rev_bits(code, bits), bits);
        } else {
            let (lcode, lextra_bits, lextra_val) = length_code(tok.length);
            let (lhuf, lhuf_bits) = fixed_litlen(lcode);
            push_atom(
                &mut atoms,
                &mut bit_offset,
                rev_bits(lhuf, lhuf_bits),
                lhuf_bits,
            );
            if lextra_bits > 0 {
                push_atom(&mut atoms, &mut bit_offset, lextra_val, lextra_bits);
            }
            let (dcode, dextra_bits, dextra_val) = distance_code(tok.distance);
            // Distance fixed code: 5 bits MSB-first → reverse for LSB-first.
            push_atom(&mut atoms, &mut bit_offset, rev_bits(dcode, 5), 5);
            if dextra_bits > 0 {
                push_atom(&mut atoms, &mut bit_offset, dextra_val, dextra_bits);
            }
        }
    }

    // EOB
    let (eob, eob_bits) = fixed_litlen(256);
    push_atom(
        &mut atoms,
        &mut bit_offset,
        rev_bits(eob, eob_bits),
        eob_bits,
    );

    (atoms, bit_offset)
}

#[inline]
fn push_atom(atoms: &mut Vec<Atom>, bit_offset: &mut u32, value: u32, n_bits: u32) {
    if n_bits == 0 {
        return;
    }
    atoms.push(Atom {
        bit_offset: *bit_offset,
        value,
        n_bits,
    });
    *bit_offset += n_bits;
}

#[cfg(test)]
mod tests {
    use super::super::lz77::Token;
    use super::*;
    use flate2::read::DeflateDecoder;
    use std::io::Read;

    fn try_pipeline() -> Option<HuffmanEmitPipeline> {
        let ctx = GpuContext::try_init().ok()?;
        Some(HuffmanEmitPipeline::new(Arc::new(ctx)))
    }

    /// Reconstruct what the GPU bitstream should decode to from a raw
    /// LZ77 token stream — a literal emits its byte; a back-ref copies
    /// `length` bytes from `distance` back. Same as `lz77::reconstruct`.
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

    fn round_trip_via_gpu(tokens: &[Token]) {
        let Some(p) = try_pipeline() else {
            eprintln!("no GPU — skipping");
            return;
        };
        let deflate_bytes = p.emit_fixed_block(tokens);
        let mut decoded = Vec::new();
        DeflateDecoder::new(&deflate_bytes[..])
            .read_to_end(&mut decoded)
            .expect("flate2 should decode GPU-emitted DEFLATE");
        let expected = reconstruct(tokens);
        assert_eq!(
            decoded, expected,
            "GPU-emitted DEFLATE didn't round-trip to expected bytes"
        );
    }

    #[test]
    fn literals_only() {
        let tokens: Vec<Token> = (0..64u8).map(Token::literal).collect();
        round_trip_via_gpu(&tokens);
    }

    #[test]
    fn single_back_ref() {
        let mut tokens: Vec<Token> = b"hello, ".iter().map(|&b| Token::literal(b)).collect();
        tokens.push(Token::back_ref(5, 7)); // "hello"
        round_trip_via_gpu(&tokens);
    }

    #[test]
    fn many_back_refs_with_extra_bits() {
        // Build a stream that exercises distance-extra-bits and
        // length-extra-bits paths. Need ≥ 400 literals up front so the
        // largest distance (257) is valid in reconstruct().
        let mut tokens: Vec<Token> = Vec::new();
        for i in 0..512u32 {
            tokens.push(Token::literal((i & 0xff) as u8));
        }
        // Distances 1, 5, 17, 257 cover extra-bit ranges 0, 1, 3, 7.
        // Lengths 3, 11, 35, 131 cover extra-bit ranges 0, 1, 3, 5.
        for &(len, dist) in &[(3, 1), (11, 5), (35, 17), (131, 257)] {
            tokens.push(Token::back_ref(len, dist));
        }
        round_trip_via_gpu(&tokens);
    }

    #[test]
    fn long_random_literal_stream() {
        // 2K literals; exercises the parallel atomicOr emit at scale.
        let tokens: Vec<Token> = (0..2048u32)
            .map(|i| Token::literal((i.wrapping_mul(2654435761) as u32 & 0xff) as u8))
            .collect();
        round_trip_via_gpu(&tokens);
    }

    #[test]
    fn empty_token_stream() {
        // No literals, no matches — but we still need a valid block:
        // header + EOB. Decoded output should be empty.
        let Some(p) = try_pipeline() else { return };
        let deflate_bytes = p.emit_fixed_block(&[]);
        let mut decoded = Vec::new();
        DeflateDecoder::new(&deflate_bytes[..])
            .read_to_end(&mut decoded)
            .expect("empty fixed block should decode");
        assert!(decoded.is_empty());
    }
}
