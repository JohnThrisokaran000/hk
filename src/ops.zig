//! Small f32 helpers behind the C ABI: activations, norms, softmax and the rotary layout permutations between Hugging
//! Face and GGUF. They are not the hot path (the engine has its own vectorized versions in `engine/ops.zig`); they are
//! kept as plain reference implementations with the semantics the bindings were written against.

const std = @import("std");
const engine_ops = @import("engine/ops.zig");
const abi_ops = @import("engine/abi_ops.zig");

/// Convert BF16 (represented as u16 raw bits) to standard IEEE 754 f32 with 0 compute overhead
pub inline fn bf16ToF32(val: u16) f32 {
    const u: u32 = @as(u32, val) << 16;
    return @bitCast(u);
}

/// Native SiLU (Swish) activation: f(x) = x / (1.0 + exp(-x))
pub fn siluF32(x: []const f32, out: []f32) void {
    const len = @min(x.len, out.len);
    for (0..len) |i| {
        const v = x[i];
        out[i] = v / (1.0 + @exp(-v));
    }
}

/// Fast RMSNorm: y = (x / sqrt(mean(x^2) + eps)) * weight (SIMD accelerated)
pub fn rmsNormF32(
    x: []const f32,
    weight: []const f32,
    eps: f32,
    out: []f32,
) void {
    const len = @min(@min(x.len, weight.len), out.len);
    if (len == 0) return;
    engine_ops.rmsNorm(out[0..len], x[0..len], weight[0..len], eps);
}

/// Native Fused SwiGLU Forward Pass:
/// y = W_down * (SiLU(W_gate * x + b_gate) * (W_up * x + b_up)) + b_down
/// Computes without any external Python or PyTorch runtime.
pub fn forwardSwiGLUF32(
    x: []const f32, // [in_f]
    w_gate: []const f32, // [inter_f, in_f]
    b_gate: ?[]const f32, // [inter_f]
    w_up: []const f32, // [inter_f, in_f]
    b_up: ?[]const f32, // [inter_f]
    w_down: []const f32, // [out_f, inter_f]
    b_down: ?[]const f32, // [out_f]
    inter_buffer: []f32, // scratch buffer of at least 2 * inter_f
    out: []f32, // [out_f]
    in_f: usize,
    inter_f: usize,
    out_f: usize,
) !void {
    if (inter_buffer.len < inter_f * 2) return error.BufferTooSmall;
    if (out.len < out_f) return error.BufferTooSmall;

    const gate_buf = inter_buffer[0..inter_f];
    const up_buf = inter_buffer[inter_f .. inter_f * 2];

    // 1. gate = W_gate * x + b_gate
    try abi_ops.gemv(.f32, std.mem.sliceAsBytes(w_gate), x, b_gate, gate_buf, inter_f, in_f);

    // 2. up = W_up * x + b_up
    try abi_ops.gemv(.f32, std.mem.sliceAsBytes(w_up), x, b_up, up_buf, inter_f, in_f);

    // 3. act = SiLU(gate) * up
    for (0..inter_f) |i| {
        const g = gate_buf[i];
        const silu_g = g / (1.0 + @exp(-g));
        gate_buf[i] = silu_g * up_buf[i]; // in-place combined activation in gate_buf
    }

    // 4. out = W_down * act + b_down
    try abi_ops.gemv(.f32, std.mem.sliceAsBytes(w_down), gate_buf, b_down, out, out_f, inter_f);
}

/// Numerically stable Softmax: p_i = exp(z_i - max) / sum(exp(z_j - max))
pub fn softmaxF32(logits: []const f32, out_probs: []f32) void {
    const len = @min(logits.len, out_probs.len);
    if (len == 0) return;

    var max_val: f32 = logits[0];
    for (1..len) |i| {
        if (logits[i] > max_val) max_val = logits[i];
    }

    var sum: f32 = 0.0;
    for (0..len) |i| {
        const diff = logits[i] - max_val;
        // e^(-20) is ~2e-9, below fp32 precision significance in probability distributions
        if (diff < -20.0) {
            out_probs[i] = 0.0;
        } else {
            const e = @exp(diff);
            out_probs[i] = e;
            sum += e;
        }
    }

    if (sum > 0.0) {
        const inv_sum = 1.0 / sum;
        for (0..len) |i| {
            out_probs[i] *= inv_sum;
        }
    }
}

/// Permutes weights from Hugging Face half-split RoPE layout to GGUF/llama.cpp interleaved layout.
pub fn ropePermuteHFToGGUF(in: []const f32, out: []f32, n_heads: usize, head_dim: usize) void {
    const head_size = n_heads * head_dim;
    if (head_size == 0) return;
    const n_batches = in.len / head_size;
    const half = head_dim / 2;

    for (0..n_batches) |b| {
        const batch_in = in[b * head_size .. (b + 1) * head_size];
        const batch_out = out[b * head_size .. (b + 1) * head_size];

        for (0..n_heads) |h| {
            const src_h = batch_in[h * head_dim .. (h + 1) * head_dim];
            const dst_h = batch_out[h * head_dim .. (h + 1) * head_dim];
            for (0..half) |i| {
                dst_h[2 * i] = src_h[i];
                dst_h[2 * i + 1] = src_h[half + i];
            }
        }
    }
}

/// Unpermutes weights from GGUF/llama.cpp interleaved layout to Hugging Face half-split layout.
pub fn ropeUnpermuteGGUFToHF(in: []const f32, out: []f32, n_heads: usize, head_dim: usize) void {
    const head_size = n_heads * head_dim;
    if (head_size == 0) return;
    const n_batches = in.len / head_size;
    const half = head_dim / 2;

    for (0..n_batches) |b| {
        const batch_in = in[b * head_size .. (b + 1) * head_size];
        const batch_out = out[b * head_size .. (b + 1) * head_size];

        for (0..n_heads) |h| {
            const src_h = batch_in[h * head_dim .. (h + 1) * head_dim];
            const dst_h = batch_out[h * head_dim .. (h + 1) * head_dim];
            for (0..half) |i| {
                dst_h[i] = src_h[2 * i];
                dst_h[half + i] = src_h[2 * i + 1];
            }
        }
    }
}

/// Applies an additive offset (e.g. +1.0 or -1.0 for Gemma/T5 LayerNorm/RMSNorm)
pub fn layerNormOffsetF32(data: []f32, offset: f32) void {
    for (0..data.len) |i| {
        data[i] += offset;
    }
}

