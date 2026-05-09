//! wgpu-based GPU codec backend.
//!
//! Cross-vendor (Vulkan / Metal / DX12) via wgpu. Initial scope: Deflate
//! compression only. Decompression remains on CPU because GPU Deflate decode
//! is dominated by sequential huffman/LZ77 dependencies that don't parallelize
//! cleanly. Inspired by cozip's Chunk-Member Profile.
//!
//! Without the `enabled` feature, this crate compiles to a stub backend that
//! reports `Capability::None` for every algorithm, so the workspace builds on
//! machines without a GPU runtime.

#[cfg(feature = "enabled")]
mod gpu;

#[cfg(not(feature = "enabled"))]
mod stub;

#[cfg(feature = "enabled")]
pub use gpu::GpuBackend;

#[cfg(not(feature = "enabled"))]
pub use stub::StubBackend as GpuBackend;
