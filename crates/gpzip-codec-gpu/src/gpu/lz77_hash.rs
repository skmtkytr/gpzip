//! Hash-table LZ77 on the GPU. Two-pass design:
//!
//! 1. **Build pass** — every input position writes its 3-byte hash into a
//!    hash table slot. `atomicMin` keeps the oldest position seen for each
//!    bucket; later writers can't displace an earlier one. This concedes
//!    "best match" quality (ideal LZ77 wants the *closest* prior position)
//!    in exchange for a lock-free O(1) build.
//!
//! 2. **Lookup pass** — every position reads its slot, verifies the 3-byte
//!    hit isn't a hash collision, then extends the match forward.
//!
//! Output token format matches `lz77.rs`: one (length, distance) per input
//! byte. The host's `greedy_walk` and the rest of the gzip pipeline are
//! shared between brute-force and hash variants.

use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::context::GpuContext;
use super::lz77::Token;

const BUILD_SHADER: &str = include_str!("lz77_hash_build.wgsl");
const LOOKUP_SHADER: &str = include_str!("lz77_hash_lookup.wgsl");

/// 2^HASH_BITS buckets. 16 → 64K buckets.
pub const HASH_BITS: u32 = 16;
const HASH_BUCKETS: usize = 1 << HASH_BITS;

/// K sub-slots per bucket. Each input position p writes to sub-slot
/// `p % CHAIN_K`, so positions sharing a hash spread across sub-slots and
/// only collide when both hash AND `p % K` match. Lookup reads all K
/// sub-slots and picks the closest prior position. K=4 keeps the table at
/// 1 MiB and gives a decent shot at finding a recent match.
pub const CHAIN_K: u32 = 4;
const HASH_SIZE: usize = HASH_BUCKETS * CHAIN_K as usize;

pub const MIN_MATCH: u32 = 3;
pub const MAX_MATCH: u32 = 258;
pub const DEFAULT_WINDOW: u32 = 32 * 1024;

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Params {
    input_len: u32,
    hash_bits: u32,
    window: u32,
    min_match: u32,
    max_match: u32,
    chain_k: u32,
    _pad: [u32; 2],
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
    hash_table: wgpu::Buffer,
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
        let hash_table = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpzip-lz77-pool-hash"),
            size: (HASH_SIZE * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
            hash_table,
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

        // Build: input(read) | hash_table(atomic read_write) | params(uniform)
        let build_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpzip-lz77-hash-build-bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    uniform_entry(2),
                ],
            });
        // Lookup: input(read) | hash_table(read) | tokens(read_write) | params
        let lookup_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpzip-lz77-hash-lookup-bgl"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, false),
                    uniform_entry(3),
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

        // Pre-stage one buffer of all-0xff bytes for fast hash-table reset.
        let reset_blob = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpzip-lz77-hash-reset-blob"),
                contents: &vec![0xffu8; HASH_SIZE * 4],
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

        // Update params (cheap, 32 bytes).
        let params = Params {
            input_len: n,
            hash_bits: HASH_BITS,
            window,
            min_match: MIN_MATCH,
            max_match: MAX_MATCH,
            chain_k: CHAIN_K,
            _pad: [0; 2],
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
                        resource: set.hash_table.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
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
                        resource: set.hash_table.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: set.tokens.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
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

        // Reset hash table to all-0xff via cheap GPU-side copy from the
        // pre-staged blob. Avoids re-uploading 1 MiB of constant data.
        encoder.copy_buffer_to_buffer(
            &self.reset_blob,
            0,
            &set.hash_table,
            0,
            (HASH_SIZE * 4) as u64,
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
