//! Segmented-hash LZ77 on the GPU. Two-pass design:
//!
//! 1. **Build pass** — each position p does
//!    `atomicMin(&seg_table[hash(p)][p >> seg_log2], p+1)`, keeping the
//!    *oldest* position per (hash, segment) bucket.
//!
//! 2. **Lookup pass** — walk segments from p's own segment back to 0 (or
//!    until distance exceeds window). Each segment provides at most one
//!    candidate; verify, extend, keep the longest match.
//!
//! ## Why segmentation
//!
//! Earlier attempts at a parallel hash chain (atomicExchange linked list)
//! and a K-way atomicMin bucket both failed for repetitive data. The chain
//! variant has no position ordering — workgroups race at the head, so the
//! chain is dominated by whichever workgroup ran last; lookup walks find
//! mostly "future" positions that get filtered out. The K-way bucket
//! variant pinned each bucket to its oldest K occupants, leaving every
//! later position outside the 32 KiB window.
//!
//! Segmenting by `p >> seg_log2` sidesteps both problems: every segment
//! within window distance is guaranteed to have at most one candidate, and
//! that candidate is bounded in distance by `(seg_offset + 1) * seg_size`.
//! For SEG_LOG2=12 (4 KiB segments) and window=32 KiB, each lookup walks
//! up to 8 segments, getting up to 8 candidates spread across the window.
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

pub const MIN_MATCH: u32 = 3;
pub const MAX_MATCH: u32 = 258;
pub const DEFAULT_WINDOW: u32 = 32 * 1024;

/// Unpack the GPU's packed-u32 token stream into the host-side `Token`
/// struct. Encoded layout: `length << 16 | (distance | byte)`. For
/// literals length=0 and the low 16 bits hold the byte value (0..255).
#[inline]
fn unpack_tokens(packed: &[u32]) -> Vec<Token> {
    packed
        .iter()
        .map(|&w| Token {
            length: w >> 16,
            distance: w & 0xFFFF,
        })
        .collect()
}

/// log2 of segment size in bytes. 12 → 4 KiB segments. Smaller segments
/// give finer-grained candidates (closer back-refs in expectation) at the
/// cost of a larger seg_table. With window=32 KiB, lookup walks up to
/// `window >> SEG_LOG2` = 8 segments.
pub const SEG_LOG2: u32 = 12;
const SEG_SIZE: usize = 1 << SEG_LOG2;

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Params {
    input_len: u32,
    hash_bits: u32,
    window: u32,
    min_match: u32,
    max_match: u32,
    seg_log2: u32,
    num_segs: u32,
    _pad: u32,
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
    /// Per-(hash, segment) oldest position. Sized HASH_BUCKETS × num_segs
    /// (where num_segs = ceil(capacity_bytes / SEG_SIZE)). Reset to all
    /// 0xFF between chunks so the atomicMin write always replaces the
    /// sentinel.
    seg_oldest: wgpu::Buffer,
    seg_newest: wgpu::Buffer,
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
        let num_segs = capacity_bytes.div_ceil(SEG_SIZE).max(1);
        let seg_table_bytes = HASH_BUCKETS * num_segs * 4;
        // seg_table_bytes is consumed only by the create_buffer call below;
        // the per-chunk reset size is recomputed in match_find* from the
        // actual input length.
        let input = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpzip-lz77-pool-input"),
            size: padded as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Two seg tables: atomicMin keeps the OLDEST p per (hash, seg)
        // bucket; atomicMax keeps the NEWEST. Lookup tries both to get a
        // closer-distance candidate (better Huffman code) when available.
        let seg_oldest = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpzip-lz77-pool-seg-oldest"),
            size: seg_table_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let seg_newest = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpzip-lz77-pool-seg-newest"),
            size: seg_table_bytes as u64,
            // COPY_DST needed for clear_buffer (zero fill).
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Tokens are packed as one u32 per input position (length<<16 |
        // distance/byte). Halves the buffer + readback bytes vs the older
        // vec2<u32> layout — host-side `Token { length: u32, distance: u32 }`
        // is reconstructed by `unpack_tokens` after readback.
        let token_bytes = (capacity_bytes as u64) * (std::mem::size_of::<u32>() as u64);
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
            seg_oldest,
            seg_newest,
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

        // Build: input(read) | seg_oldest(atomic rw) | seg_newest(atomic rw) | params
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
        // Lookup: input(read) | seg_oldest(read) | seg_newest(read) | tokens(rw) | params
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

        // Pre-stage all-0xFF for seg_table reset between chunks. Sized to
        // cover seg_tables up to 512 KiB chunks (worst case 64K * 128 * 4
        // = 32 MiB on GPU memory). RTX 4090 has 24 GiB so this is rounding
        // error; smaller GPUs still have plenty of headroom and we never
        // upload it from host after init.
        const RESET_BLOB_BYTES: usize = HASH_BUCKETS * 128 * 4;
        let reset_blob = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpzip-lz77-hash-reset-blob"),
                contents: &vec![0xffu8; RESET_BLOB_BYTES],
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
        let token_bytes = (n as u64) * (std::mem::size_of::<u32>() as u64);

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
        let num_segs = (input.len().div_ceil(SEG_SIZE)).max(1) as u32;
        let params = Params {
            input_len: n,
            hash_bits: HASH_BITS,
            window,
            min_match: MIN_MATCH,
            max_match: MAX_MATCH,
            seg_log2: SEG_LOG2,
            num_segs,
            _pad: 0,
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
                        resource: set.seg_oldest.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: set.seg_newest.as_entire_binding(),
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
                        resource: set.seg_oldest.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: set.seg_newest.as_entire_binding(),
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

        // Reset both seg tables. seg_oldest needs all-0xFF (atomicMin
        // sentinel); seg_newest needs all-0x00 (atomicMax sentinel).
        // Only reset the bytes used by THIS chunk's table (smaller chunks
        // need less).
        let reset_bytes = (HASH_BUCKETS as u64) * (num_segs as u64) * 4;
        encoder.copy_buffer_to_buffer(&self.reset_blob, 0, &set.seg_oldest, 0, reset_bytes);
        encoder.clear_buffer(&set.seg_newest, 0, Some(reset_bytes));

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
        let tokens = unpack_tokens(bytemuck::cast_slice::<u8, u32>(&view));
        drop(view);
        set.staging.unmap();

        self.release(set);
        tokens
    }

    /// Synchronous wrapper around `submit_batch_async` + `collect_async`.
    /// Kept for direct callers (tests) that don't pipeline; the production
    /// path through `BatchedLz77` uses the async pair so submission of
    /// batch N+1 can overlap with read-back of batch N (`WaitForSubmissionIndex`
    /// is per-submission, so the GPU queue can deepen beyond one batch).
    #[allow(dead_code)]
    pub fn match_find_batch(&self, inputs: &[&[u8]], window: u32) -> Vec<Vec<Token>> {
        if inputs.is_empty() {
            return Vec::new();
        }
        let async_batch = self.submit_batch_async(inputs, window);
        self.collect_async(async_batch)
    }

    /// Submit a batch to the GPU and return a handle the caller can hand
    /// to `collect_async` later. Doesn't block on completion — the only
    /// host work is the input upload + encoder build + submit, all
    /// synchronous but fast (~50–100 µs per batch in profile).
    pub fn submit_batch_async(&self, inputs: &[&[u8]], window: u32) -> AsyncBatch {
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

            let num_segs = (input.len().div_ceil(SEG_SIZE)).max(1) as u32;
            let params = Params {
                input_len: n,
                hash_bits: HASH_BITS,
                window,
                min_match: MIN_MATCH,
                max_match: MAX_MATCH,
                seg_log2: SEG_LOG2,
                num_segs,
                _pad: 0,
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
                                resource: set.seg_oldest.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: set.seg_newest.as_entire_binding(),
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
                                resource: set.seg_oldest.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: set.seg_newest.as_entire_binding(),
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
            token_bytes_each.push((n as u64) * (std::mem::size_of::<u32>() as u64));
        }

        // One encoder for all chunks.
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpzip-lz77-batch-enc"),
            });
        for (idx, set) in sets.iter().enumerate() {
            let input_bytes = inputs[idx].len();
            let n = input_bytes as u32;
            let num_segs = input_bytes.div_ceil(SEG_SIZE).max(1) as u64;
            let reset_bytes = (HASH_BUCKETS as u64) * num_segs * 4;
            encoder.copy_buffer_to_buffer(&self.reset_blob, 0, &set.seg_oldest, 0, reset_bytes);
            encoder.clear_buffer(&set.seg_newest, 0, Some(reset_bytes));
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
        let submission_index = self.ctx.queue.submit(std::iter::once(encoder.finish()));

        // Register map_async on every staging buffer; callbacks fire when
        // the corresponding poll() processes them in `collect_async`.
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

        AsyncBatch {
            sets,
            receivers,
            token_bytes: token_bytes_each,
            submission_index,
        }
    }

    /// Block until the given submission completes, then read each staging
    /// buffer and release the BufferSets back to the pool. Uses
    /// `WaitForSubmissionIndex` rather than `Wait` so the wait scope is
    /// just this batch — later submissions in the queue keep running on
    /// the GPU concurrently with this readback.
    pub fn collect_async(&self, batch: AsyncBatch) -> Vec<Vec<Token>> {
        let AsyncBatch {
            sets,
            receivers,
            token_bytes,
            submission_index,
        } = batch;

        self.ctx
            .device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(submission_index));

        let mut out = Vec::with_capacity(sets.len());
        for ((set, &nbytes), rx) in sets.iter().zip(&token_bytes).zip(receivers) {
            rx.recv().unwrap().expect("buffer map failed");
            let view = set.staging.slice(0..nbytes).get_mapped_range();
            let tokens = unpack_tokens(bytemuck::cast_slice::<u8, u32>(&view));
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

/// In-flight batch handle. Holds the GPU buffers alive until `collect_async`
/// reads them back, plus a SubmissionIndex so the wait can be scoped to
/// just this batch (lets the worker submit the next batch concurrently).
pub struct AsyncBatch {
    sets: Vec<BufferSet>,
    receivers: Vec<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    token_bytes: Vec<u64>,
    submission_index: wgpu::SubmissionIndex,
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
