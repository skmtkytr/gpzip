//! Parallel LZ77 match-finding on the GPU. See `lz77.wgsl` for the shader.
//!
//! Output is *per-position*: one token per byte of input. A separate serial
//! pass (`greedy_walk`) selects which matches the final encoder actually
//! uses, and `reconstruct` is the round-trip verifier.
//!
//! Helpers are scaffolding for A-3 (token emit) — exercised today only by
//! tests, so dead-code lints are allowed here.
#![allow(dead_code)]

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::context::GpuContext;

const SHADER_SRC: &str = include_str!("lz77.wgsl");

/// Default look-back window. DEFLATE allows up to 32 KiB; we start smaller
/// since the brute-force shader is O(window) per position.
pub const DEFAULT_WINDOW: u32 = 4096;

/// Minimum match length worth recording. DEFLATE uses 3.
pub const MIN_MATCH: u32 = 3;

/// Cap on a single match. DEFLATE uses 258.
pub const MAX_MATCH: u32 = 258;

/// One LZ77 token. `length == 0` means literal (and `distance` holds the
/// byte). `length >= MIN_MATCH` means a back-reference of that length at
/// `distance` bytes back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct Token {
    pub length: u32,
    pub distance: u32,
}

impl Token {
    pub fn literal(byte: u8) -> Self {
        Self {
            length: 0,
            distance: byte as u32,
        }
    }
    pub fn back_ref(length: u32, distance: u32) -> Self {
        debug_assert!((MIN_MATCH..=MAX_MATCH).contains(&length));
        debug_assert!(distance >= 1);
        Self { length, distance }
    }
    pub fn is_literal(&self) -> bool {
        self.length == 0
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Params {
    input_len: u32,
    window: u32,
    min_match: u32,
    max_match: u32,
}

pub struct Lz77Pipeline {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl Lz77Pipeline {
    pub fn new(ctx: Arc<GpuContext>) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpzip-lz77-shader"),
                source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
            });

        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpzip-lz77-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pl_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpzip-lz77-pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gpzip-lz77-pipeline"),
                layout: Some(&pl_layout),
                module: &module,
                entry_point: "main",
            });

        Self {
            ctx,
            pipeline,
            bind_group_layout: bgl,
        }
    }

    /// Returns one token per input byte. Use `greedy_walk` to convert this
    /// per-position view into a non-overlapping LZ77 token stream.
    pub fn match_find(&self, input: &[u8], window: u32) -> Vec<Token> {
        let n = input.len() as u32;
        if n == 0 {
            return Vec::new();
        }

        let padded = input.len().next_multiple_of(4);
        let mut input_padded = Vec::with_capacity(padded);
        input_padded.extend_from_slice(input);
        input_padded.resize(padded, 0);

        let input_buffer = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpzip-lz77-input"),
                contents: &input_padded,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let token_bytes = (n as u64) * (std::mem::size_of::<Token>() as u64);
        let tokens_buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpzip-lz77-tokens"),
            size: token_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpzip-lz77-staging"),
            size: token_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = Params {
            input_len: n,
            window,
            min_match: MIN_MATCH,
            max_match: MAX_MATCH,
        };
        let params_buffer = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpzip-lz77-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpzip-lz77-bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: tokens_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpzip-lz77-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpzip-lz77-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&tokens_buffer, 0, &staging_buffer, 0, token_bytes);
        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("buffer map failed");

        let view = slice.get_mapped_range();
        let tokens: Vec<Token> = bytemuck::cast_slice::<u8, Token>(&view).to_vec();
        drop(view);
        staging_buffer.unmap();

        tokens
    }
}

/// Walk the per-position output, applying a *lazy* match policy: at each
/// position, if the next position's match is strictly longer, emit the
/// current byte as a literal and re-evaluate from the next position. Same
/// idea as zlib's lazy matching — a few percent ratio improvement on
/// text-heavy input for trivial extra work.
///
/// Needs the input slice to recover the literal byte at positions where
/// the shader picked a back-reference but lazy demotes to literal.
pub fn greedy_walk(per_position: &[Token], input: &[u8]) -> Vec<Token> {
    debug_assert_eq!(per_position.len(), input.len());
    let mut out = Vec::new();
    let mut p = 0;
    while p < per_position.len() {
        let t = per_position[p];
        if t.is_literal() {
            out.push(t);
            p += 1;
            continue;
        }
        // Lazy: peek at next position. If its match is strictly longer,
        // emit the current byte as literal and let the next position win.
        if let Some(next) = per_position.get(p + 1) {
            if !next.is_literal() && next.length > t.length {
                out.push(Token::literal(input[p]));
                p += 1;
                continue;
            }
        }
        out.push(t);
        p += t.length as usize;
    }
    out
}

/// Reconstruct the original byte stream from a non-overlapping token list.
/// Used to verify GPU output: round-trip should match input bytes exactly.
pub fn reconstruct(tokens: &[Token]) -> Vec<u8> {
    let mut out = Vec::new();
    for t in tokens {
        if t.is_literal() {
            out.push(t.distance as u8);
        } else {
            let start = out.len() - t.distance as usize;
            // Use a manual loop because matches can self-overlap (e.g.
            // length=5 dist=1 means "repeat the last byte 5 times").
            for i in 0..t.length as usize {
                let b = out[start + i];
                out.push(b);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_pipeline() -> Option<Lz77Pipeline> {
        let ctx = GpuContext::try_init().ok()?;
        Some(Lz77Pipeline::new(Arc::new(ctx)))
    }

    #[test]
    fn empty_input_yields_no_tokens() {
        let Some(p) = try_pipeline() else {
            return;
        };
        assert!(p.match_find(&[], 4096).is_empty());
    }

    #[test]
    fn pure_literals_when_no_matches() {
        let Some(pipeline) = try_pipeline() else {
            return;
        };
        // Bytes 0..32 are all distinct, so no matches possible.
        let input: Vec<u8> = (0..32).collect();
        let raw = pipeline.match_find(&input, 4096);
        for (i, t) in raw.iter().enumerate() {
            assert!(t.is_literal(), "pos {i}: {:?}", t);
            assert_eq!(t.distance as u8, input[i]);
        }
        let walked = greedy_walk(&raw, &input);
        let restored = reconstruct(&walked);
        assert_eq!(restored, input);
    }

    #[test]
    fn finds_obvious_repeat() {
        let Some(pipeline) = try_pipeline() else {
            return;
        };
        // "abcabcabc..." — every position from 3 onward should match.
        let mut input = Vec::new();
        for _ in 0..16 {
            input.extend_from_slice(b"abc");
        }
        let raw = pipeline.match_find(&input, 4096);
        // Position 3 should at least match "abc..." back to position 0
        let t = raw[3];
        assert!(!t.is_literal(), "position 3 should be a match: {:?}", t);
        assert!(t.length >= 3);
        assert_eq!(t.distance, 3);

        let walked = greedy_walk(&raw, &input);
        let restored = reconstruct(&walked);
        assert_eq!(restored, input);
    }

    #[test]
    fn round_trips_random_data() {
        let Some(pipeline) = try_pipeline() else {
            return;
        };
        // Pseudo-random but deterministic.
        let input: Vec<u8> = (0..2048u32)
            .map(|i| ((i.wrapping_mul(2654435761)) & 0xff) as u8)
            .collect();
        let raw = pipeline.match_find(&input, 4096);
        let walked = greedy_walk(&raw, &input);
        let restored = reconstruct(&walked);
        assert_eq!(restored, input);
    }

    #[test]
    fn round_trips_self_overlapping_match() {
        let Some(pipeline) = try_pipeline() else {
            return;
        };
        // "aaaaaa..." — best match is length=N-1, distance=1, which requires
        // the decoder to handle self-overlap.
        let input = vec![b'a'; 64];
        let raw = pipeline.match_find(&input, 4096);
        let walked = greedy_walk(&raw, &input);
        let restored = reconstruct(&walked);
        assert_eq!(restored, input);
    }
}
