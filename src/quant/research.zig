//! Research formats with no GGML counterpart: NF4, the dual-mode DQ8 and ternary DQT block formats, FP8 E4M3, and
//! raw 4-bit streams (IQ4_NL codebook, MXFP4, NVFP4) with an explicit scale. They are stored and used from Python and the
//! C ABI; the engine does not run them. These moved here unchanged from the removed legacy `quantization.zig`.

const std = @import("std");

/// 16-point NormalFloat-4 (NF4) quantile lookup table
/// Normalized to [-1.0, 1.0] from standard normal distribution N(0, 1) quantiles
pub const NF4_TABLE: [16]f32 = .{
    -1.0000000,
    -0.6961928,
    -0.5250731,
    -0.3949175,
    -0.2844414,
    -0.1847734,
    -0.0910500,
    0.0000000,
    0.0795803,
    0.1609302,
    0.2461123,
    0.3379152,
    0.4407098,
    0.5626170,
    0.7229568,
    1.0000000,
};

/// Finds the closest NF4 code (0..15) for a normalized value in [-1.0, 1.0]
pub fn findClosestNF4(val: f32) u4 {
    var best_idx: u4 = 0;
    var min_dist: f32 = std.math.inf(f32);

    inline for (0..16) |i| {
        const dist = @abs(val - NF4_TABLE[i]);
        if (dist < min_dist) {
            min_dist = dist;
            best_idx = @intCast(i);
        }
    }
    return best_idx;
}

/// Quantizes a block of floats to NF4 with optional residual tracking.
/// packed_out receives ceil(block.len / 2) bytes (2 4-bit codes per byte).
pub fn quantizeBlockNF4(
    block: []const f32,
    packed_out: []u8,
    residual_out: ?[]f32,
) f32 {
    if (block.len == 0) return 0.0;

    var max_abs: f32 = 0.0;
    for (block) |x| {
        const a = @abs(x);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0) max_abs = 1e-8;

    const inv_scale = 1.0 / max_abs;

    for (0..block.len) |i| {
        const norm = std.math.clamp(block[i] * inv_scale, -1.0, 1.0);
        const code = findClosestNF4(norm);

        const byte_idx = i / 2;
        if (i % 2 == 0) {
            packed_out[byte_idx] = @as(u8, code);
        } else {
            packed_out[byte_idx] |= (@as(u8, code) << 4);
        }

        if (residual_out) |res| {
            const deq_val = NF4_TABLE[code] * max_abs;
            res[i] = block[i] - deq_val;
        }
    }

    return max_abs;
}

/// Dequantizes packed 4-bit NF4 bytes to 32-bit floats using block scale.
pub fn dequantizeBlockNF4(
    packed_in: []const u8,
    scale: f32,
    count: usize,
    out: []f32,
) void {
    var i: usize = 0;
    // Process 4 elements at a time with vectorization where possible
    while (i + 4 <= count) : (i += 4) {
        const b0 = packed_in[i / 2];
        const b1 = packed_in[(i / 2) + 1];
        const c0: u4 = @truncate(b0 & 0x0F);
        const c1: u4 = @truncate((b0 >> 4) & 0x0F);
        const c2: u4 = @truncate(b1 & 0x0F);
        const c3: u4 = @truncate((b1 >> 4) & 0x0F);

        const v: @Vector(4, f32) = .{
            NF4_TABLE[c0],
            NF4_TABLE[c1],
            NF4_TABLE[c2],
            NF4_TABLE[c3],
        };
        const s_vec: @Vector(4, f32) = @splat(scale);
        const res_vec = v * s_vec;
        out[i..][0..4].* = res_vec;
    }

    while (i < count) : (i += 1) {
        const byte_idx = i / 2;
        const code: u4 = if (i % 2 == 0)
            @truncate(packed_in[byte_idx] & 0x0F)
        else
            @truncate((packed_in[byte_idx] >> 4) & 0x0F);

        out[i] = NF4_TABLE[code] * scale;
    }
}

/// Quantizes a block of floats to symmetric 8-bit integers (DQ8) with optional residual delta.
pub fn quantizeBlockDQ8(
    block: []const f32,
    out_i8: []i8,
    residual_out: ?[]f32,
) f32 {
    if (block.len == 0) return 0.0;

    var max_abs: f32 = 0.0;
    for (block) |x| {
        const a = @abs(x);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0) max_abs = 1e-8;

    const scale = max_abs / 127.0;
    const inv_scale = 127.0 / max_abs;

    for (0..block.len) |i| {
        const q_float = std.math.clamp(block[i] * inv_scale, -127.0, 127.0);
        const q_int: i8 = @intFromFloat(@round(q_float));
        out_i8[i] = q_int;

        if (residual_out) |res| {
            const deq = @as(f32, @floatFromInt(q_int)) * scale;
            res[i] = block[i] - deq;
        }
    }

    return scale;
}

/// Dequantizes signed 8-bit integers (DQ8) back to 32-bit floats.
pub fn dequantizeBlockDQ8(
    in_i8: []const i8,
    scale: f32,
    count: usize,
    out: []f32,
) void {
    var i: usize = 0;
    while (i + 4 <= count) : (i += 4) {
        const q: @Vector(4, f32) = .{
            @floatFromInt(in_i8[i + 0]),
            @floatFromInt(in_i8[i + 1]),
            @floatFromInt(in_i8[i + 2]),
            @floatFromInt(in_i8[i + 3]),
        };
        const s: @Vector(4, f32) = @splat(scale);
        out[i..][0..4].* = q * s;
    }
    while (i < count) : (i += 1) {
        out[i] = @as(f32, @floatFromInt(in_i8[i])) * scale;
    }
}

/// BitNet b1.58 Ternary Quantization: weights in {-1, 0, +1}
/// 2 bits per weight:
///   00 -> 0
///   01 -> +1
///   10 -> -1
/// Scale = average absolute value: alpha = (1/N) * sum(|W|)
pub fn quantizeBlockDQT(
    block: []const f32,
    packed_out: []u8,
    residual_out: ?[]f32,
) f32 {
    if (block.len == 0) return 0.0;

    var sum_abs: f64 = 0.0;
    for (block) |x| {
        sum_abs += @abs(x);
    }
    const scale: f32 = @floatCast(sum_abs / @as(f64, @floatFromInt(block.len)));
    const eff_scale = if (scale == 0.0) 1e-8 else scale;
    const thresh = eff_scale * 0.5;

    @memset(packed_out, 0);

    for (0..block.len) |i| {
        var code: u2 = 0; // 00 -> 0
        var val: f32 = 0.0;

        if (block[i] > thresh) {
            code = 1; // 01 -> +1
            val = eff_scale;
        } else if (block[i] < -thresh) {
            code = 2; // 10 -> -1
            val = -eff_scale;
        }

        const byte_idx = i / 4;
        const bit_shift: u3 = @intCast((i % 4) * 2);
        packed_out[byte_idx] |= (@as(u8, code) << bit_shift);

        if (residual_out) |res| {
            res[i] = block[i] - val;
        }
    }

    return eff_scale;
}

/// Dequantizes packed 2-bit ternary weights back to 32-bit floats using block scale.
pub fn dequantizeBlockDQT(
    packed_in: []const u8,
    scale: f32,
    count: usize,
    out: []f32,
) void {
    for (0..count) |i| {
        const byte_idx = i / 4;
        const bit_shift: u3 = @intCast((i % 4) * 2);
        const code: u2 = @truncate((packed_in[byte_idx] >> bit_shift) & 0x03);

        const val: f32 = switch (code) {
            1 => scale,
            2 => -scale,
            else => 0.0,
        };
        out[i] = val;
    }
}

fn computeFP8_E4M3_LUT() [256]f32 {
    @setEvalBranchQuota(100000);
    var table: [256]f32 = undefined;
    for (0..256) |idx| {
        const b: u8 = @intCast(idx);
        const sign: f32 = if ((b & 0x80) != 0) -1.0 else 1.0;
        const exp = (b >> 3) & 0x0F;
        const mant = b & 0x07;

        if (exp == 0) {
            // Subnormal: (-1)^sign * 2^(-6) * (mant / 8)
            // 2^(-6) = 0.015625
            table[idx] = sign * 0.015625 * (@as(f32, @floatFromInt(mant)) / 8.0);
        } else if (exp == 15 and mant == 7) {
            // NaN in E4M3
            table[idx] = std.math.nan(f32);
        } else {
            // Normalized: (-1)^sign * 2^(exp - 7) * (1 + mant / 8)
            const exp_i: i32 = @as(i32, @intCast(exp)) - 7;
            const factor: f32 = if (exp_i >= 0)
                @floatFromInt(@as(u32, 1) << @intCast(exp_i))
            else
                1.0 / @as(f32, @floatFromInt(@as(u32, 1) << @intCast(-exp_i)));
            table[idx] = sign * factor * (1.0 + @as(f32, @floatFromInt(mant)) / 8.0);
        }
    }
    return table;
}

pub const FP8_E4M3_TABLE: [256]f32 = computeFP8_E4M3_LUT();

/// Dequantizes FP8 E4M3 to FP32 using comptime-generated lookup table (sign: 1, exp: 4, mantissa: 3, bias: 7)
pub fn dequantizeFP8_E4M3(in_bytes: []const u8, count: usize, out: []f32) void {
    const limit = @min(@min(in_bytes.len, count), out.len);
    var i: usize = 0;
    while (i + 8 <= limit) : (i += 8) {
        out[i + 0] = FP8_E4M3_TABLE[in_bytes[i + 0]];
        out[i + 1] = FP8_E4M3_TABLE[in_bytes[i + 1]];
        out[i + 2] = FP8_E4M3_TABLE[in_bytes[i + 2]];
        out[i + 3] = FP8_E4M3_TABLE[in_bytes[i + 3]];
        out[i + 4] = FP8_E4M3_TABLE[in_bytes[i + 4]];
        out[i + 5] = FP8_E4M3_TABLE[in_bytes[i + 5]];
        out[i + 6] = FP8_E4M3_TABLE[in_bytes[i + 6]];
        out[i + 7] = FP8_E4M3_TABLE[in_bytes[i + 7]];
    }
    while (i < limit) : (i += 1) {
        out[i] = FP8_E4M3_TABLE[in_bytes[i]];
    }
}

// ---------------------------------------------------------------------------
// Standard Baseline Quantizations: Q4_0 and Q8_0 (32 weights per block)
// ---------------------------------------------------------------------------

pub const IQ4NL_TABLE: [16]f32 = .{
    -1.0000000, -0.7903226, -0.6129032, -0.4677419,
    -0.3387097, -0.2258065, -0.1290323, -0.0483871,
     0.0483871,  0.1290323,  0.2258065,  0.3387097,
     0.4677419,  0.6129032,  0.7903226,  1.0000000,
};

pub fn findClosestIQ4NL(val: f32) u4 {
    var best: u4 = 0;
    var min_d = std.math.inf(f32);
    inline for (0..16) |i| {
        const dist = @abs(val - IQ4NL_TABLE[i]);
        if (dist < min_d) {
            min_d = dist;
            best = @intCast(i);
        }
    }
    return best;
}

pub fn quantizeBlockIQ4_NL(block: []const f32, packed_out: []u8) f32 {
    if (block.len == 0) return 0.0;
    var max_abs: f32 = 0.0;
    for (block) |x| {
        const a = @abs(x);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0) max_abs = 1e-8;
    const inv_scale = 1.0 / max_abs;

    for (0..block.len) |i| {
        const norm = std.math.clamp(block[i] * inv_scale, -1.0, 1.0);
        const code = findClosestIQ4NL(norm);
        const byte_idx = i / 2;
        if (i % 2 == 0) {
            packed_out[byte_idx] = @as(u8, code);
        } else {
            packed_out[byte_idx] |= (@as(u8, code) << 4);
        }
    }
    return max_abs;
}

pub fn dequantizeBlockIQ4_NL(packed_in: []const u8, scale: f32, count: usize, out: []f32) void {
    for (0..count) |i| {
        const byte_idx = i / 2;
        const code: u4 = if (i % 2 == 0)
            @truncate(packed_in[byte_idx] & 0x0F)
        else
            @truncate((packed_in[byte_idx] >> 4) & 0x0F);

        out[i] = IQ4NL_TABLE[code] * scale;
    }
}

// ---------------------------------------------------------------------------
// Microscaling Formats: MXFP4 (OCP) and NVFP4 (Blackwell)
// ---------------------------------------------------------------------------

pub const FP4_E2M1_TABLE: [16]f32 = .{
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
};

/// OCP Microscaling MXFP4: 32 elements per block with E8M0 scale factor
pub fn dequantizeBlockMXFP4(packed_fp4: []const u8, e8m0_scale: u8, count: usize, out: []f32) void {
    const scale_exp = @as(f32, @floatFromInt(e8m0_scale)) - 127.0;
    const scale = std.math.pow(f32, 2.0, scale_exp);

    for (0..count) |i| {
        const byte_idx = i / 2;
        const code: u4 = if (i % 2 == 0)
            @truncate(packed_fp4[byte_idx] & 0x0F)
        else
            @truncate((packed_fp4[byte_idx] >> 4) & 0x0F);

        out[i] = FP4_E2M1_TABLE[code] * scale;
    }
}

/// NVIDIA Blackwell NVFP4: 16 elements per block with FP8 E4M3 scale factor
pub fn dequantizeBlockNVFP4(packed_fp4: []const u8, fp8_scale: u8, count: usize, out: []f32) void {
    var scale_arr: [1]f32 = undefined;
    const scale_bytes = [_]u8{fp8_scale};
    dequantizeFP8_E4M3(&scale_bytes, 1, &scale_arr);
    const scale = scale_arr[0];

    for (0..count) |i| {
        const byte_idx = i / 2;
        const code: u4 = if (i % 2 == 0)
            @truncate(packed_fp4[byte_idx] & 0x0F)
        else
            @truncate((packed_fp4[byte_idx] >> 4) & 0x0F);

        out[i] = FP4_E2M1_TABLE[code] * scale;
    }
}


// -- research products (no engine counterpart) -----------------------------------------------------------------------
// Moved unchanged from the removed legacy `tensor_ops.zig`.

const nf4 = @import("../nf4.zig");
const format = @import("../format.zig");

const FusedGemvNF4Ctx = struct {
    packed_W: []const u8,
    scales: []const f32,
    x: []const f32,
    bias: ?[]const f32,
    y: []f32,
    K: usize,
    block_size: usize,
    blocks_per_row: usize,
};

/// Fused NF4 Dequantize-and-GEMV: computes y = Dequant(W_nf4) * x + bias
/// Does not materialize dequantized float32 weight matrix in RAM!
pub fn fusedGemvNF4(
    packed_W: []const u8,
    scales: []const f32,
    x: []const f32,
    bias: ?[]const f32,
    y: []f32,
    M: usize,
    K: usize,
    block_size: usize,
) void {
    const safe_M = @min(M, y.len);
    const blocks_per_row = (K + block_size - 1) / block_size;
    fusedGemvNF4Worker(packed_W, scales, x, bias, y, 0, safe_M, K, block_size, blocks_per_row);
}

fn fusedGemvNF4Task(ctx_ptr: *anyopaque, start_r: usize, end_r: usize) void {
    const ctx: *const FusedGemvNF4Ctx = @ptrCast(@alignCast(ctx_ptr));
    fusedGemvNF4Worker(ctx.packed_W, ctx.scales, ctx.x, ctx.bias, ctx.y, start_r, end_r, ctx.K, ctx.block_size, ctx.blocks_per_row);
}

fn fusedGemvNF4Worker(
    packed_W: []const u8,
    scales: []const f32,
    x: []const f32,
    bias: ?[]const f32,
    y: []f32,
    start_r: usize,
    end_r: usize,
    K: usize,
    block_size: usize,
    blocks_per_row: usize,
) void {
    const k_bytes = (K + 1) / 2;
    for (start_r..end_r) |r| {
        const row_packed = packed_W[r * k_bytes .. (r + 1) * k_bytes];
        var dot: f32 = 0.0;
        var global_block_idx = r * blocks_per_row;

        var k: usize = 0;
        while (k < K) {
            const block_scale = if (global_block_idx < scales.len) scales[global_block_idx] else 1.0;
            global_block_idx += 1;

            const cur_block_len = @min(block_size, K - k);
            for (0..cur_block_len) |bi| {
                const elem_idx = k + bi;
                const byte_val = row_packed[elem_idx / 2];
                const code: usize = if (elem_idx % 2 == 0)
                    (byte_val & 0x0F)
                else
                    ((byte_val >> 4) & 0x0F);

                const w_val = nf4.NF4_TABLE[code] * block_scale;
                dot += w_val * x[elem_idx];
            }
            k += cur_block_len;
        }

        if (bias) |b| {
            if (r < b.len) dot += b[r];
        }
        y[r] = dot;
    }
}

/// Fused DQ8 Dequantize-and-GEMV: computes y = (W_i8 * scale) * x + bias
pub fn fusedGemvDQ8(
    W_i8: []const i8,
    scales: []const f32,
    x: []const f32,
    bias: ?[]const f32,
    y: []f32,
    M: usize,
    K: usize,
    block_size: usize,
) void {
    const safe_M = @min(M, y.len);
    const blocks_per_row = (K + block_size - 1) / block_size;
    fusedGemvDQ8Worker(W_i8, scales, x, bias, y, 0, safe_M, K, block_size, blocks_per_row);
}

fn fusedGemvDQ8Worker(
    W_i8: []const i8,
    scales: []const f32,
    x: []const f32,
    bias: ?[]const f32,
    y: []f32,
    start_r: usize,
    end_r: usize,
    K: usize,
    block_size: usize,
    blocks_per_row: usize,
) void {
    for (start_r..end_r) |r| {
        const row_i8 = W_i8[r * K .. (r + 1) * K];
        var dot: f32 = 0.0;
        var global_block_idx = r * blocks_per_row;

        var k: usize = 0;
        while (k < K) {
            const block_scale = if (global_block_idx < scales.len) scales[global_block_idx] else 1.0;
            global_block_idx += 1;

            const cur_block_len = @min(block_size, K - k);
            var bi: usize = 0;
            while (bi < cur_block_len) : (bi += 1) {
                const elem_idx = k + bi;
                const w_val = @as(f32, @floatFromInt(row_i8[elem_idx])) * block_scale;
                dot += w_val * x[elem_idx];
            }
            k += cur_block_len;
        }

        if (bias) |b| {
            if (r < b.len) dot += b[r];
        }
        y[r] = dot;
    }
}

/// Fast INT8 dot product (optimized for Intel VNNI and ARM NEON pipelines, 32 elements/iter)
pub fn dotProductInt8(a: []const i8, b: []const i8) i32 {
    const len = @min(a.len, b.len);
    var i: usize = 0;

    var acc0: @Vector(8, i32) = @splat(0);
    var acc1: @Vector(8, i32) = @splat(0);
    var acc2: @Vector(8, i32) = @splat(0);
    var acc3: @Vector(8, i32) = @splat(0);

    while (i + 32 <= len) : (i += 32) {
        const v0_a: @Vector(8, i8) = a[i + 0 ..][0..8].*;
        const v0_b: @Vector(8, i8) = b[i + 0 ..][0..8].*;
        acc0 += @as(@Vector(8, i32), v0_a) * @as(@Vector(8, i32), v0_b);

        const v1_a: @Vector(8, i8) = a[i + 8 ..][0..8].*;
        const v1_b: @Vector(8, i8) = b[i + 8 ..][0..8].*;
        acc1 += @as(@Vector(8, i32), v1_a) * @as(@Vector(8, i32), v1_b);

        const v2_a: @Vector(8, i8) = a[i + 16 ..][0..8].*;
        const v2_b: @Vector(8, i8) = b[i + 16 ..][0..8].*;
        acc2 += @as(@Vector(8, i32), v2_a) * @as(@Vector(8, i32), v2_b);

        const v3_a: @Vector(8, i8) = a[i + 24 ..][0..8].*;
        const v3_b: @Vector(8, i8) = b[i + 24 ..][0..8].*;
        acc3 += @as(@Vector(8, i32), v3_a) * @as(@Vector(8, i32), v3_b);
    }

    var sum = @reduce(.Add, (acc0 + acc1) + (acc2 + acc3));

    while (i + 8 <= len) : (i += 8) {
        const va: @Vector(8, i8) = a[i..][0..8].*;
        const vb: @Vector(8, i8) = b[i..][0..8].*;
        sum += @reduce(.Add, @as(@Vector(8, i32), va) * @as(@Vector(8, i32), vb));
    }

    while (i < len) : (i += 1) {
        sum += @as(i32, a[i]) * @as(i32, b[i]);
    }
    return sum;
}

/// Fast Matrix-Vector Multiplication for Raw INT8 Weights with f32 activation (SIMD accelerated):
/// y = (W_int8 * x) * scale + bias
pub fn gemvInt8Scaled(
    W_i8: []const i8,
    x: []const f32,
    scale_w: f32,
    bias: ?[]const f32,
    y: []f32,
    M: usize,
    K: usize,
) void {
    const safe_M = @min(M, y.len);
    if (scale_w == 0.0) {
        for (0..safe_M) |r| {
            y[r] = if (bias) |b| (if (r < b.len) b[r] else 0.0) else 0.0;
        }
        return;
    }
    const is_unit_scale = (scale_w == 1.0);
    for (0..safe_M) |r| {
        const row_start = r * K;
        const row_end = @min(row_start + K, W_i8.len);
        if (row_start >= W_i8.len) {
            y[r] = if (bias) |b| (if (r < b.len) b[r] else 0.0) else 0.0;
            continue;
        }
        const row = W_i8[row_start..row_end];
        var row_sum = dotProductInt8F32(row, x);
        if (!is_unit_scale) {
            row_sum *= scale_w;
        }
        if (bias) |b| {
            if (r < b.len) row_sum += b[r];
        }
        y[r] = row_sum;
    }
}

/// Fast SIMD dot product of INT8 weights with float32 activations (32 elements/iter)
pub fn dotProductInt8F32(a: []const i8, b: []const f32) f32 {
    const len = @min(a.len, b.len);
    var i: usize = 0;

    var acc0: @Vector(8, f32) = @splat(0.0);
    var acc1: @Vector(8, f32) = @splat(0.0);
    var acc2: @Vector(8, f32) = @splat(0.0);
    var acc3: @Vector(8, f32) = @splat(0.0);

    while (i + 32 <= len) : (i += 32) {
        const v0_i8: @Vector(8, i8) = a[i + 0 ..][0..8].*;
        const vb0: @Vector(8, f32) = b[i + 0 ..][0..8].*;
        acc0 += @as(@Vector(8, f32), @floatFromInt(v0_i8)) * vb0;

        const v1_i8: @Vector(8, i8) = a[i + 8 ..][0..8].*;
        const vb1: @Vector(8, f32) = b[i + 8 ..][0..8].*;
        acc1 += @as(@Vector(8, f32), @floatFromInt(v1_i8)) * vb1;

        const v2_i8: @Vector(8, i8) = a[i + 16 ..][0..8].*;
        const vb2: @Vector(8, f32) = b[i + 16 ..][0..8].*;
        acc2 += @as(@Vector(8, f32), @floatFromInt(v2_i8)) * vb2;

        const v3_i8: @Vector(8, i8) = a[i + 24 ..][0..8].*;
        const vb3: @Vector(8, f32) = b[i + 24 ..][0..8].*;
        acc3 += @as(@Vector(8, f32), @floatFromInt(v3_i8)) * vb3;
    }

    var sum = @reduce(.Add, (acc0 + acc1) + (acc2 + acc3));

    while (i + 8 <= len) : (i += 8) {
        const v_i8: @Vector(8, i8) = a[i..][0..8].*;
        const vb: @Vector(8, f32) = b[i..][0..8].*;
        sum += @reduce(.Add, @as(@Vector(8, f32), @floatFromInt(v_i8)) * vb);
    }

    while (i < len) : (i += 1) {
        sum += @as(f32, @floatFromInt(a[i])) * b[i];
    }
    return sum;
}
