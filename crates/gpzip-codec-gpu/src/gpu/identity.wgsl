// Identity copy shader. Bring-up scaffolding for the GPU codec — proves
// that buffer upload, dispatch, and readback all work end-to-end on the
// host's GPU. Future shaders (LZ77 match find, Huffman) plug into the
// same wgpu plumbing.

@group(0) @binding(0) var<storage, read> input_buf: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_buf: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&input_buf)) {
        output_buf[idx] = input_buf[idx];
    }
}
