//! Hash-chain LZ77 on the GPU. Two-pass design:
//!
//! 1. **Build pass** — each position p does
//!    `prev = atomicExchange(&heads[hash(p)], p+1)` to install itself at the
//!    chain head, then writes `next[p] = prev`. The chain at any bucket is
//!    a singly-linked list of all positions sharing the 3-byte hash,
//!    ordered newest-first.
//!
//! 2. **Lookup pass** — walk the chain at heads[hash(p)] following next[],
//!    bounded by `max_chain` candidates. For each candidate verify the
//!    3-byte hit and extend the match forward; keep the longest. The chain
//!    is monotone in position (newer entries point at older ones), so once
//!    we walk past `window` bytes we can stop early.
//!
//! Output token format matches `lz77.rs`: one (length, distance) per input
//! byte. The host's `greedy_walk` and the rest of the gzip pipeline are
//! shared between brute-force and hash variants.
//!
//! This replaces an earlier K-way `atomicMin` bucket design that gave
//! lock-free O(1) build but pinned each bucket to its oldest occupants —
//! catastrophic for repetitive data because every later position would
//! resolve to a distance > window and lose all matches. See the review
//! notes for the measurement (rep workload ratio went from 0.547 to ~0.003
//! with this change).

use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::context::GpuContext;
use super::lz77::Token;

const BUILD_SHADER: &str = include_str!("lz77_hash_build.wgsl");
const LOOKUP_SHADER: &str = include_str!("lz77_hash_lookup.wgsl");

/// 2^HASH_BITS buckets. 16 → 64K buckets, heads buffer = 256 KiB.
pub const HASH_BITS: u32 = 16;
const HASH_BUCKETS: usize = 1 << HASH_BITS;

pub const MIN_MATCH: u32 = 3;
pub const MAX_MATCH: u32 = 258;
pub const DEFAULT_WINDOW: u32 = 32 * 1024;

/// Cap on chain walk in lookup. zlib uses 32 at level 5, 128 at level 7.
/// Since GPU threads run thousands in parallel, a longer chain costs more
/// per-thread but doesn't hurt occupancy. Set high (1024) because the GPU
/// chain isn't ordered by position — workgroups race at the head, so the
/// chain head is biased toward whichever workgroup ran last. To reach
/// genuine prior-position candidates within a 512 KiB chunk, the walk
/// often needs to traverse several hundred entries.
pub const MAX_CHAIN: u32 = 1024;

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Params {
    input_len: u32,
    hash_bits: u32,
    window: u32,
    min_match: u32,
    max_match: u32,
    max_chain: u32,
}

pub struct Lz77HashPipeline {
    ctx: Arc<GpuContext>,
    build_pipeline: wgpu::ComputePipeline,
    lookup_pipeline: wgpu::ComputePipeline,
    build_layout: wgpu::BindGroupLayout,
    lookup_layout: wgpu::BindGroupLayout,
    /// Reusable buffer sets — see `BufferSet`. The pool is a stack: most
    /// recently released entry is reused next, so caches stay warm.
    pool: Mutex<Vec<BufferSet>>,
    /// Pre-staged buffer of all-`0xff` bytes, used to reset the hash table
    /// between chunks via `copy_buffer_to_buffer` instead of a fresh upload.
    reset_blob: wgpu::Buffer,
}

/// One set of GPU buffers sized for a single chunk. The pool reuses these
/// across `match_find` calls so we stop paying buffer-creation overhead on
/// every chunk (was ~5-10 ms / chunk per profiling).
struct BufferSet {
    input: wgpu::Buffer,
    /// Chain heads, one u32 per hash bucket. Reset to zero between chunks.
    heads: wgpu::Buffer,
    /// next[p] = previous chain head when this position inserted itself.
    /// One u32 per input byte; reads only happen at positions that were
    /// written (no stale-data concern), so no reset between chunks.
    next: wgpu::Buffer,
    tokens: wgpu::Buffer,
    staging: wgpu::Buffer,
    params: wgpu::Buffer,
    /// Number of input bytes the buffers were allocated for. A set can
    /// service requests up to this size; smaller chunks just leave the tail
    /// of the buffer unused. Larger chunks need a fresh, bigger set.
    capacity_bytes: usize,
}

impl BufferSet {
    fn new(ctx: &GpuContext, capacity_bytes: usize) -> Self {
        let padded = capacity_bytes.next_multiple_of(4);
        let input = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpzip-lz77-pool-input"),
            size: padded as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let heads = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpzip-lz77-pool-heads"),
            size: (HASH_BUCKETS * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let next = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpzip-lz77-pool-next"),
            // One u32 per input byte position.
            size: (padded as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let token_bytes = (capacity_bytes as u64) * (std::mem::size_of::<Token>() as u64);
        let tokens = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpzip-lz77-pool-tokens"),
            size: token_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpzip-lz77-pool-staging"),
            size: token_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpzip-lz77-pool-params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            input,
            heads,
            next,
            tokens,
            staging,
            params,
            capacity_bytes,
        }
    }
}

impl Lz77HashPipeline {
    pub fn new(ctx: Arc<GpuContext>) -> Self {
        let build_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpzip-lz77-hash-build"),
                source: wgpu::ShaderSource::Wgsl(BUILD_SHADER.into()),
            });
        let lookup_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpzip-lz77-hash-lookup"),
                source: wgpu::ShaderSource::Wgsl(LOOKUP_SHADER.into()),
            });

        // Build: input(read) | heads(atomic rw) | next(rw, non-atomic) | params
        let build_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpzip-lz77-hash-build-bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    storage_entry(2, false),
                    uniform_entry(3),
                ],
            });
        // Lookup: input(read) | heads(read) | next(read) | tokens(rw) | params
        let lookup_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpzip-lz77-hash-lookup-bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, false),
                    uniform_entry(4),
                ],
            });

        let build_pl_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpzip-lz77-hash-build-pl"),
                bind_group_layouts: &[&build_layout],
                push_constant_ranges: &[],
            });
        let lookup_pl_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpzip-lz77-hash-lookup-pl"),
                bind_group_layouts: &[&lookup_layout],
                push_constant_ranges: &[],
            });

        let build_pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gpzip-lz77-hash-build-pipeline"),
                layout: Some(&build_pl_layout),
                module: &build_module,
                entry_point: "build",
            });
        let lookup_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("gpzip-lz77-hash-lookup-pipeline"),
                    layout: Some(&lookup_pl_layout),
                    module: &lookup_module,
                    entry_point: "lookup",
                });

        // Pre-stage one buffer of zeros for fast `heads` reset between
        // chunks. `next` doesn't need reset (only positions that wrote
        // their next[] entry will ever be read by the chain walk).
        let reset_blob = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpzip-lz77-hash-reset-blob"),
                contents: &vec![0u8; HASH_BUCKETS * 4],
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        Self {
            ctx,
            build_pipeline,
            lookup_pipeline,
            build_layout,
            lookup_layout,
            pool: Mutex::new(Vec::new()),
            reset_blob,
        }
    }

    fn acquire(&self, capacity_bytes: usize) -> BufferSet {
        let mut pool = self.pool.lock().unwrap();
        if let Some(idx) = pool.iter().position(|s| s.capacity_bytes >= capacity_bytes) {
            return pool.swap_remove(idx);
        }
        drop(pool);
        BufferSet::new(&self.ctx, capacity_bytes)
    }

    fn release(&self, set: BufferSet) {
        // Cap the pool so we don't accumulate unbounded memory if many
        // different chunk sizes have flowed through.
        let mut pool = self.pool.lock().unwrap();
        if pool.len() < 8 {
            pool.push(set);
        }
    }

    /// Per-position LZ77. Same output shape as the brute-force pipeline.
    /// Reuses GPU buffers across calls via the internal pool.
    ///
    /// Today's production path goes through `match_find_batch` via the
    /// `BatchedLz77` worker; this single-shot entry point is kept for
    /// tests and as a fallback for callers that don't have batching set up.
    #[allow(dead_code)]
    pub fn match_find(&self, input: &[u8], window: u32) -> Vec<Token> {
        let n = input.len() as u32;
        if n == 0 {
            return Vec::new();
        }

        let set = self.acquire(input.len());
        let token_bytes = (n as u64) * (std::mem::size_of::<Token>() as u64);

        // Upload input — write_buffer reuses the buffer object. We pad to
        // u32 because the WGSL side reads as `array<u32>`.
        let padded = input.len().next_multiple_of(4);
        let pad_extra = padded - input.len();
        if pad_extra == 0 {
            self.ctx.queue.write_buffer(&set.input, 0, input);
        } else {
            let mut buf = Vec::with_capacity(padded);
            buf.extend_from_slice(input);
            buf.resize(padded, 0);
            self.ctx.queue.write_buffer(&set.input, 0, &buf);
        }

        // Update params (cheap, 24 bytes).
        let params = Params {
            input_len: n,
            hash_bits: HASH_BITS,
            window,
            min_match: MIN_MATCH,
            max_match: MAX_MATCH,
            max_chain: MAX_CHAIN,
        };
        self.ctx
            .queue
            .write_buffer(&set.params, 0, bytemuck::bytes_of(&params));

        // Bind groups must be rebuilt because they reference buffer handles
        // and we use a fresh BufferSet each call. Bind-group creation is
        // cheap (~tens of µs) so this isn't worth pooling separately.
        let build_bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpzip-lz77-hash-build-bg"),
                layout: &self.build_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: set.input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: set.heads.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: set.next.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: set.params.as_entire_binding(),
                    },
                ],
            });
        let lookup_bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpzip-lz77-hash-lookup-bg"),
                layout: &self.lookup_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: set.input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: set.heads.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: set.next.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: set.tokens.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: set.params.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpzip-lz77-hash-enc"),
            });

        // Reset heads to zero via GPU-side copy. 256 KiB.
        encoder.copy_buffer_to_buffer(
            &self.reset_blob,
            0,
            &set.heads,
            0,
            (HASH_BUCKETS * 4) as u64,
        );

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpzip-lz77-hash-build-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.build_pipeline);
            pass.set_bind_group(0, &build_bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpzip-lz77-hash-lookup-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.lookup_pipeline);
            pass.set_bind_group(0, &lookup_bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&set.tokens, 0, &set.staging, 0, token_bytes);
        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        let slice = set.staging.slice(0..token_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("buffer map failed");

        let view = slice.get_mapped_range();
        let tokens: Vec<Token> = bytemuck::cast_slice::<u8, Token>(&view).to_vec();
        drop(view);
        set.staging.unmap();

        self.release(set);
        tokens
    }

    /// Process several chunks in a single command-buffer submission. Acquires
    /// one BufferSet per input, builds all the hash-reset, build, lookup, and
    /// copy commands into one encoder, submits once, polls once, then maps
    /// every staging buffer in turn. Cuts the per-chunk submit + poll cost
    /// down to per-batch.
    pub fn match_find_batch(&self, inputs: &[&[u8]], window: u32) -> Vec<Vec<Token>> {
        if inputs.is_empty() {
            return Vec::new();
        }

        // Acquire all buffer sets up front so the encoder can reference them.
        let sets: Vec<BufferSet> = inputs
            .iter()
            .map(|i| self.acquire(i.len().max(4)))
            .collect();

        // Bind groups likewise need to be alive for the whole encoder; collect
        // them into Vecs so they live until submit.
        let mut build_bgs = Vec::with_capacity(inputs.len());
        let mut lookup_bgs = Vec::with_capacity(inputs.len());
        let mut token_bytes_each = Vec::with_capacity(inputs.len());

        // Upload inputs and update params buffers.
        for (input, set) in inputs.iter().zip(&sets) {
            let n = input.len() as u32;
            let padded = input.len().next_multiple_of(4);
            let pad_extra = padded - input.len();
            if pad_extra == 0 {
                self.ctx.queue.write_buffer(&set.input, 0, input);
            } else {
                let mut buf = Vec::with_capacity(padded);
                buf.extend_from_slice(input);
                buf.resize(padded, 0);
                self.ctx.queue.write_buffer(&set.input, 0, &buf);
            }

            let params = Params {
                input_len: n,
                hash_bits: HASH_BITS,
                window,
                min_match: MIN_MATCH,
                max_match: MAX_MATCH,
                max_chain: MAX_CHAIN,
            };
            self.ctx
                .queue
                .write_buffer(&set.params, 0, bytemuck::bytes_of(&params));

            build_bgs.push(
                self.ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("gpzip-lz77-batch-build-bg"),
                        layout: &self.build_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: set.input.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: set.heads.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: set.next.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: set.params.as_entire_binding(),
                            },
                        ],
                    }),
            );
            lookup_bgs.push(
                self.ctx
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("gpzip-lz77-batch-lookup-bg"),
                        layout: &self.lookup_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: set.input.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: set.heads.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: set.next.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: set.tokens.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: set.params.as_entire_binding(),
                            },
                        ],
                    }),
            );
            token_bytes_each.push((n as u64) * (std::mem::size_of::<Token>() as u64));
        }

        // One encoder for all chunks.
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpzip-lz77-batch-enc"),
            });
        for (idx, set) in sets.iter().enumerate() {
            let n = inputs[idx].len() as u32;
            encoder.copy_buffer_to_buffer(
                &self.reset_blob,
                0,
                &set.heads,
                0,
                (HASH_BUCKETS * 4) as u64,
            );
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gpzip-lz77-batch-build-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.build_pipeline);
                pass.set_bind_group(0, &build_bgs[idx], &[]);
                pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gpzip-lz77-batch-lookup-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.lookup_pipeline);
                pass.set_bind_group(0, &lookup_bgs[idx], &[]);
                pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
            }
            encoder.copy_buffer_to_buffer(&set.tokens, 0, &set.staging, 0, token_bytes_each[idx]);
        }
        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        // Map every staging buffer; a single device.poll(Wait) covers all of
        // them because Wait blocks until *all* pending submissions complete.
        let receivers: Vec<_> = sets
            .iter()
            .zip(&token_bytes_each)
            .map(|(set, &nbytes)| {
                let slice = set.staging.slice(0..nbytes);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |res| {
                    let _ = tx.send(res);
                });
                rx
            })
            .collect();
        self.ctx.device.poll(wgpu::Maintain::Wait);

        let mut out = Vec::with_capacity(inputs.len());
        for ((set, &nbytes), rx) in sets.iter().zip(&token_bytes_each).zip(receivers) {
            rx.recv().unwrap().expect("buffer map failed");
            let view = set.staging.slice(0..nbytes).get_mapped_range();
            let tokens: Vec<Token> = bytemuck::cast_slice::<u8, Token>(&view).to_vec();
            drop(view);
            set.staging.unmap();
            out.push(tokens);
        }

        for set in sets {
            self.release(set);
        }
        out
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
    use super::super::lz77::{greedy_walk, reconstruct};
    use super::*;

    fn try_pipeline() -> Option<Lz77HashPipeline> {
        let ctx = GpuContext::try_init().ok()?;
        Some(Lz77HashPipeline::new(Arc::new(ctx)))
    }

    fn round_trip(p: &Lz77HashPipeline, input: &[u8]) {
        let raw = p.match_find(input, DEFAULT_WINDOW);
        let walked = greedy_walk(&raw, input);
        let restored = reconstruct(&walked);
        assert_eq!(
            restored,
            input,
            "round trip mismatch on input len {}",
            input.len()
        );
    }

    #[test]
    fn empty_input() {
        let Some(p) = try_pipeline() else { return };
        assert!(p.match_find(&[], DEFAULT_WINDOW).is_empty());
    }

    #[test]
    fn small_no_match() {
        let Some(p) = try_pipeline() else { return };
        round_trip(&p, &(0..32u8).collect::<Vec<_>>());
    }

    #[test]
    fn obvious_repeat() {
        let Some(p) = try_pipeline() else { return };
        let input: Vec<u8> = b"abcabcabcabcabcabcabcabc".repeat(8).to_vec();
        round_trip(&p, &input);
    }

    #[test]
    fn self_overlap() {
        let Some(p) = try_pipeline() else { return };
        round_trip(&p, &[b'q'; 64]);
    }

    #[test]
    fn pseudo_random() {
        let Some(p) = try_pipeline() else { return };
        let input: Vec<u8> = (0..2048u32)
            .map(|i| (i.wrapping_mul(2654435761)) as u8)
            .collect();
        round_trip(&p, &input);
    }

    #[test]
    fn larger_text() {
        let Some(p) = try_pipeline() else { return };
        let mut input = Vec::new();
        for i in 0..512 {
            input.extend_from_slice(format!("line {i}: the quick brown fox\n").as_bytes());
        }
        round_trip(&p, &input);
    }
}
