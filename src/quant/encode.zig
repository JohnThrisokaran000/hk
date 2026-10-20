//! Encoders (quantizers) for the GGML block formats the engine decodes.
//!
//! The engine's decoders live in `dequant.zig` and are tested against llama.cpp; every encoder here is checked by
//! decoding its output with them. Block layouts are the engine's (`blocks.zig`): there is one definition of each
//! format in the repository. Encoders are straightforward min/max quantizers, not llama.cpp's slower search-based
//! ones, so their output is valid and decodes correctly but is not byte-identical to llama.cpp's.

const std = @import("std");
const blocks = @import("blocks.zig");

pub const BlockQ4_0 = blocks.BlockQ4_0;
pub const BlockQ8_0 = blocks.BlockQ8_0;
pub const BlockQ4_K = blocks.BlockQ4K;
pub const BlockQ5_K = blocks.BlockQ5K;
pub const BlockQ3_K = blocks.BlockQ3K;
pub const BlockQ6_K = blocks.BlockQ6K;
pub const BlockQ2_K = blocks.BlockQ2K;
pub const BlockQ8_K = blocks.BlockQ8K;

pub const QK4_0: usize = 32;

pub const QK8_0: usize = 32;

pub const QK_K: usize = 256;

pub fn quantizeBlockQ4_0(weights: []const f32, block: *BlockQ4_0) void {
    var max_abs: f32 = 0.0;
    var max_val: f32 = 0.0;
    for (weights) |w| {
        const a = @abs(w);
        if (a > max_abs) {
            max_abs = a;
            max_val = w;
        }
    }
    if (max_abs == 0.0) {
        block.d = 0;
        @memset(&block.qs, 0);
        return;
    }

    const d: f32 = max_val / -8.0;
    const d_f16: f16 = @floatCast(d);
    block.d = d_f16;
    const d_eff: f32 = @floatCast(d_f16);
    const id: f32 = if (d_eff != 0.0) 1.0 / d_eff else 0.0;

    for (0..16) |i| {
        const w0 = if (i < weights.len) weights[i] else 0.0;
        const w1 = if (i + 16 < weights.len) weights[i + 16] else 0.0;
        const q0_f = std.math.clamp(std.math.trunc(w0 * id + 8.5), 0.0, 15.0);
        const q1_f = std.math.clamp(std.math.trunc(w1 * id + 8.5), 0.0, 15.0);
        const q0: u8 = @intFromFloat(q0_f);
        const q1: u8 = @intFromFloat(q1_f);
        block.qs[i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
    }
}

pub fn quantizeBlockQ8_0(weights: []const f32, block: *BlockQ8_0) void {
    var max_abs: f32 = 0.0;
    for (weights) |w| {
        const a = @abs(w);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0) {
        block.d = 0;
        @memset(&block.qs, 0);
        return;
    }

    const d_f32 = max_abs / 127.0;
    const d_f16: f16 = @floatCast(d_f32);
    block.d = d_f16;
    const d_eff: f32 = @floatCast(d_f16);
    const id: f32 = if (d_eff != 0.0) 1.0 / d_eff else 0.0;

    for (0..@min(weights.len, 32)) |i| {
        const q = std.math.clamp(@round(weights[i] * id), -128.0, 127.0);
        block.qs[i] = @intFromFloat(q);
    }
    if (weights.len < 32) {
        @memset(block.qs[weights.len..32], 0);
    }
}

pub fn quantizeSuperBlockQ8_K(weights: []const f32, block: *BlockQ8_K) void {
    var max_abs: f32 = 0.0;
    for (weights) |w| {
        const a = @abs(w);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs == 0.0) max_abs = 1e-8;

    const scale = max_abs / 127.0;
    block.d = scale;
    const inv_scale = 127.0 / max_abs;

    for (0..weights.len) |i| {
        const q = std.math.clamp(weights[i] * inv_scale, -127.0, 127.0);
        block.qs[i] = @intFromFloat(@round(q));
    }

    // Compute sub-block sums for fast GEMV dot-products
    for (0..16) |sb| {
        var sum: i32 = 0;
        for (0..16) |j| {
            sum += block.qs[sb * 16 + j];
        }
        block.bsums[sb] = @intCast(std.math.clamp(sum, -32768, 32767));
    }
}

/// Helper to unpack 6-bit scales and mins from 12-byte packed buffer for Q4_K and Q5_K
pub fn getScaleMinK4(scales: *const [12]u8, sc: *[8]u8, min: *[8]u8) void {
    for (0..4) |j| {
        const d_byte = scales[j];
        const m_byte = scales[j + 4];
        const md_byte = scales[j + 8];
        sc[j] = d_byte & 0x3F;
        sc[j + 4] = (md_byte & 0x0F) | ((d_byte >> 2) & 0x30);
        min[j] = m_byte & 0x3F;
        min[j + 4] = ((md_byte >> 4) & 0x0F) | ((m_byte >> 2) & 0x30);
    }
}

/// Helper to pack 6-bit scales and mins into 12-byte buffer for Q4_K and Q5_K
pub fn setScaleMinK4(sc: *const [8]u8, min: *const [8]u8, scales: *[12]u8) void {
    for (0..4) |j| {
        const sc0 = sc[j] & 0x3F;
        const sc1 = sc[j + 4] & 0x3F;
        const m0 = min[j] & 0x3F;
        const m1 = min[j + 4] & 0x3F;
        scales[j] = sc0 | ((sc1 & 0x30) << 2);
        scales[j + 4] = m0 | ((m1 & 0x30) << 2);
        scales[j + 8] = (sc1 & 0x0F) | ((m1 & 0x0F) << 4);
    }
}

pub fn quantizeSuperBlockQ5_K(weights: []const f32, block: *BlockQ5_K) void {
    var min_all: f32 = std.math.inf(f32);
    var max_all: f32 = -std.math.inf(f32);
    for (weights) |w| {
        if (w < min_all) min_all = w;
        if (w > max_all) max_all = w;
    }
    if (max_all <= min_all) {
        block.d = 0;
        block.dmin = 0;
        @memset(&block.scales, 0);
        @memset(&block.qh, 0);
        @memset(&block.qs, 0);
        return;
    }

    var sb_min: [8]f32 = undefined;
    var sb_max: [8]f32 = undefined;
    for (0..8) |sb| {
        var smin: f32 = std.math.inf(f32);
        var smax: f32 = -std.math.inf(f32);
        const start = sb * 32;
        const end = @min(start + 32, weights.len);
        for (weights[start..end]) |w| {
            if (w < smin) smin = w;
            if (w > smax) smax = w;
        }
        sb_min[sb] = if (smin == std.math.inf(f32)) 0.0 else smin;
        sb_max[sb] = if (smax == -std.math.inf(f32)) 0.0 else smax;
    }

    var max_range: f32 = 0.0;
    for (0..8) |sb| {
        const r = sb_max[sb] - sb_min[sb];
        if (r > max_range) max_range = r;
    }
    const d_val = max_range / (31.0 * 63.0);
    const d_eff = if (d_val == 0) 1e-8 else d_val;
    block.d = @floatCast(d_eff);

    var max_min: f32 = 0.0;
    for (0..8) |sb| {
        const am = @abs(sb_min[sb]);
        if (am > max_min) max_min = am;
    }
    const dmin_val = max_min / 63.0;
    const dmin_eff = if (dmin_val == 0) 1e-8 else dmin_val;
    block.dmin = @floatCast(dmin_eff);

    var sc: [8]u8 = @as([8]u8, @splat(0));
    var min: [8]u8 = @as([8]u8, @splat(0));
    for (0..8) |sb| {
        const r = sb_max[sb] - sb_min[sb];
        const sc_f = std.math.clamp(@round(r / (31.0 * d_eff)), 0.0, 63.0);
        sc[sb] = @intFromFloat(sc_f);
        const min_f = std.math.clamp(@round(-sb_min[sb] / dmin_eff), 0.0, 63.0);
        min[sb] = @intFromFloat(min_f);
    }
    setScaleMinK4(&sc, &min, &block.scales);

    @memset(&block.qh, 0);
    @memset(&block.qs, 0);

    for (0..8) |sb| {
        const start = sb * 32;
        const end = @min(start + 32, weights.len);
        const sub_d = @as(f32, @floatCast(block.d)) * @as(f32, @floatFromInt(sc[sb]));
        const sub_m = @as(f32, @floatCast(block.dmin)) * @as(f32, @floatFromInt(min[sb]));
        const inv_d = if (sub_d != 0) 1.0 / sub_d else 0.0;

        const pair_idx = sb / 2;
        const is_high = (sb % 2) == 1;
        const qs_offset = pair_idx * 32;

        for (start..end) |i| {
            const idx_in_sb = i - start;
            const q_float = std.math.clamp(@round((weights[i] + sub_m) * inv_d), 0.0, 31.0);
            const q_val: u8 = @intFromFloat(q_float);
            const ql = q_val & 0x0F;
            const qh = (q_val >> 4) & 0x01;

            if (is_high) {
                block.qs[qs_offset + idx_in_sb] |= (ql << 4);
            } else {
                block.qs[qs_offset + idx_in_sb] = ql;
            }
            block.qh[idx_in_sb] |= (qh << @intCast(sb));
        }
    }
}

/// Q3_K: 256 weights as sixteen sub-blocks of 16, each with a signed 6-bit scale (stored with a +32 bias) relative to
/// the block's f16 super-scale `d`, so `y = d * (scale - 32) * v` with v in -4..3. A weight's 3 bits are two bits of `qs`
/// plus an inverted sign bit in `hmask` (set means non-negative). The element -> byte/bit mapping and the packing of the
/// sixteen 6-bit scales into 12 bytes are the inverse of `dequant.q3_k`, which is the reference.
pub fn quantizeSuperBlockQ3_K(weights: []const f32, block: *BlockQ3_K) void {
    std.debug.assert(weights.len == QK_K);
    var step: [16]f32 = undefined;
    var max_abs_step: f32 = 0;
    for (0..16) |j| {
        const w = weights[j * 16 ..][0..16];
        // Map the element of largest magnitude to -4, the end with the extra level.
        var extreme: f32 = 0;
        for (w) |x| {
            if (@abs(x) > @abs(extreme)) extreme = x;
        }
        step[j] = extreme / -4.0;
        max_abs_step = @max(max_abs_step, @abs(step[j]));
    }
    block.d = @floatCast(if (max_abs_step > 0) max_abs_step / 32.0 else 0);
    const d: f32 = @floatCast(block.d);

    var sc: [16]i32 = undefined;
    for (0..16) |j| {
        sc[j] = if (d > 0) @intFromFloat(std.math.clamp(@round(step[j] / d), -32.0, 31.0)) else 0;
    }

    // Pack the biased 6-bit scales: low nibbles of scales k / k+8 share a byte (k < 4) and of k+4 / k+12 likewise;
    // the top two bits of four scales share each of the last four bytes.
    var u: [16]u8 = undefined;
    for (0..16) |j| u[j] = @intCast(sc[j] + 32);
    for (0..4) |k| {
        block.scales[k] = (u[k] & 0x0F) | ((u[k + 8] & 0x0F) << 4);
        block.scales[4 + k] = (u[4 + k] & 0x0F) | ((u[12 + k] & 0x0F) << 4);
        block.scales[8 + k] = (u[k] >> 4) | ((u[4 + k] >> 4) << 2) | ((u[8 + k] >> 4) << 4) | ((u[12 + k] >> 4) << 6);
    }

    @memset(&block.qs, 0);
    @memset(&block.hmask, 0);
    for (0..QK_K) |e| {
        const half = e / 128;
        const r = e % 128;
        const s = r / 32;
        const sub = (r % 32) / 16;
        const l = r % 16;
        const j = half * 8 + s * 2 + sub;
        const eff = d * @as(f32, @floatFromInt(sc[j]));
        const v: i32 = if (eff == 0) 0 else @intFromFloat(std.math.clamp(@round(weights[e] / eff), -4.0, 3.0));
        const lo: u8 = @intCast(if (v >= 0) v else v + 4);
        block.qs[half * 32 + sub * 16 + l] |= lo << @intCast(2 * s);
        if (v >= 0) block.hmask[sub * 16 + l] |= @as(u8, 1) << @intCast(half * 4 + s);
    }
}

pub fn quantizeSuperBlockQ6_K(weights: []const f32, block: *BlockQ6_K) void {
    var max_all: f32 = 0.0;
    for (weights) |w| {
        const a = @abs(w);
        if (a > max_all) max_all = a;
    }
    if (max_all == 0.0) {
        block.d = 0;
        @memset(&block.ql, 0);
        @memset(&block.qh, 0);
        @memset(&block.scales, 0);
        return;
    }

    var sb_max: [16]f32 = @as([16]f32, @splat(0.0));
    for (0..16) |sb| {
        const start = sb * 16;
        const end = @min(start + 16, weights.len);
        var smax: f32 = 0.0;
        for (weights[start..end]) |w| {
            const a = @abs(w);
            if (a > smax) smax = a;
        }
        sb_max[sb] = smax;
    }

    const d_val = max_all / (31.0 * 127.0);
    const d_eff = if (d_val == 0) 1e-8 else d_val;
    block.d = @floatCast(d_eff);

    for (0..16) |sb| {
        const sc_f = std.math.clamp(@round(sb_max[sb] / (31.0 * d_eff)), 1.0, 127.0);
        block.scales[sb] = @intFromFloat(sc_f);
    }

    @memset(&block.ql, 0);
    @memset(&block.qh, 0);

    for (0..16) |sb| {
        const start = sb * 16;
        const end = @min(start + 16, weights.len);
        const sub_scale = @as(f32, @floatCast(block.d)) * @as(f32, @floatFromInt(block.scales[sb]));
        const inv_scale = if (sub_scale != 0) 1.0 / sub_scale else 0.0;

        const h: usize = start / 128;
        const rem = start % 128;
        const chunk32 = rem / 32;
        const is_high = chunk32 >= 2;
        const ql_chunk: usize = chunk32 % 2;
        const offset_in_32: usize = rem % 32;
        const ql_base: usize = h * 64 + ql_chunk * 32 + offset_in_32;
        const qh_base: usize = h * 32 + offset_in_32;
        const qh_shift: u3 = @intCast(chunk32 * 2);

        for (start..end) |i| {
            const j = i - start;
            const q_f = std.math.clamp(@round(weights[i] * inv_scale), -32.0, 31.0);
            const val6: i8 = @intFromFloat(q_f);
            const code: u8 = @intCast(val6 + 32);

            const low_nibble = code & 0x0F;
            const high_bits = (code >> 4) & 0x03;

            if (is_high) {
                block.ql[ql_base + j] |= (low_nibble << 4);
            } else {
                block.ql[ql_base + j] = low_nibble;
            }

            block.qh[qh_base + j] |= (high_bits << qh_shift);
        }
    }
}

pub fn quantizeSuperBlockQ2_K(weights: []const f32, block: *BlockQ2_K) void {
    var min_all: f32 = std.math.inf(f32);
    var max_all: f32 = -std.math.inf(f32);
    for (weights) |w| {
        if (w < min_all) min_all = w;
        if (w > max_all) max_all = w;
    }
    if (max_all <= min_all) {
        block.d = 0;
        block.dmin = 0;
        @memset(&block.scales, 0);
        @memset(&block.qs, 0);
        return;
    }

    var sb_min: [16]f32 = undefined;
    var sb_max: [16]f32 = undefined;
    for (0..16) |sb| {
        var smin: f32 = std.math.inf(f32);
        var smax: f32 = -std.math.inf(f32);
        const start = sb * 16;
        const end = @min(start + 16, weights.len);
        for (weights[start..end]) |w| {
            if (w < smin) smin = w;
            if (w > smax) smax = w;
        }
        sb_min[sb] = if (smin == std.math.inf(f32)) 0.0 else smin;
        sb_max[sb] = if (smax == -std.math.inf(f32)) 0.0 else smax;
    }

    var max_range: f32 = 0.0;
    for (0..16) |sb| {
        const r = sb_max[sb] - sb_min[sb];
        if (r > max_range) max_range = r;
    }
    const d_val = max_range / (3.0 * 15.0);
    const d_eff = if (d_val == 0) 1e-8 else d_val;
    block.d = @floatCast(d_eff);

    var max_min: f32 = 0.0;
    for (0..16) |sb| {
        const am = @abs(sb_min[sb]);
        if (am > max_min) max_min = am;
    }
    const dmin_val = max_min / 15.0;
    const dmin_eff = if (dmin_val == 0) 1e-8 else dmin_val;
    block.dmin = @floatCast(dmin_eff);

    @memset(&block.scales, 0);
    @memset(&block.qs, 0);

    for (0..16) |sb| {
        const r = sb_max[sb] - sb_min[sb];
        const sc_f = std.math.clamp(@round(r / (3.0 * d_eff)), 0.0, 15.0);
        const sc_val: u8 = @intFromFloat(sc_f);
        const min_f = std.math.clamp(@round(-sb_min[sb] / dmin_eff), 0.0, 15.0);
        const min_val: u8 = @intFromFloat(min_f);
        block.scales[sb] = (sc_val & 0x0F) | ((min_val & 0x0F) << 4);

        const sub_scale = @as(f32, @floatCast(block.d)) * @as(f32, @floatFromInt(sc_val));
        const sub_min = @as(f32, @floatCast(block.dmin)) * @as(f32, @floatFromInt(min_val));
        const inv_scale = if (sub_scale != 0) 1.0 / sub_scale else 0.0;

        const start = sb * 16;
        const end = @min(start + 16, weights.len);
        const h: usize = start / 128;
        const rem = start % 128;
        const chunk32 = rem / 32;
        const shift: u3 = @intCast(chunk32 * 2);
        const offset_in_32: usize = rem % 32;
        const qs_base: usize = h * 32 + offset_in_32;

        for (start..end) |i| {
            const j = i - start;
            const q_f = std.math.clamp(@round((weights[i] + sub_min) * inv_scale), 0.0, 3.0);
            const code: u2 = @intFromFloat(q_f);

            block.qs[qs_base + j] |= (@as(u8, code) << shift);
        }
    }
}


/// Q4_K: 256 weights as eight sub-blocks of 32. Each sub-block has its own 6-bit scale and 6-bit minimum, stored
/// relative to the block's two f16 super-scales `d` and `dmin`, so `y = d * sc * q - dmin * m` with q in 0..15. The
/// 12 scale bytes use the layout shared with Q5_K (`setScaleMinK4`), and each byte of `qs` holds sub-block `2g` in its
/// low nibble and `2g + 1` in its high nibble, for the 64-weight group `g`. This is GGML's Q4_K, not the private layout
/// the legacy quantizer wrote.
pub fn quantizeSuperBlockQ4_K(weights: []const f32, block: *BlockQ4_K) void {
    std.debug.assert(weights.len == QK_K);
    var step: [8]f32 = undefined;
    var offset: [8]f32 = undefined;
    var max_step: f32 = 0;
    var max_offset: f32 = 0;
    for (0..8) |j| {
        const w = weights[j * 32 ..][0..32];
        var lo = w[0];
        var hi = w[0];
        for (w) |x| {
            lo = @min(lo, x);
            hi = @max(hi, x);
        }
        // The offset is non-negative, so a sub-block whose values are all positive starts at zero.
        if (lo > 0) lo = 0;
        step[j] = (hi - lo) / 15.0;
        offset[j] = -lo;
        max_step = @max(max_step, step[j]);
        max_offset = @max(max_offset, offset[j]);
    }
    block.d = @floatCast(if (max_step > 0) max_step / 63.0 else 0);
    block.dmin = @floatCast(if (max_offset > 0) max_offset / 63.0 else 0);
    // Choose the codes against the f16 scales the decoder will actually read.
    const d: f32 = @floatCast(block.d);
    const dmin: f32 = @floatCast(block.dmin);
    var sc: [8]u8 = undefined;
    var mn: [8]u8 = undefined;
    for (0..8) |j| {
        sc[j] = if (d > 0) @intFromFloat(@round(@min(step[j] / d, 63.0))) else 0;
        mn[j] = if (dmin > 0) @intFromFloat(@round(@min(offset[j] / dmin, 63.0))) else 0;
    }
    setScaleMinK4(&sc, &mn, &block.scales);
    for (0..4) |g| {
        for (0..32) |l| {
            const lo_code = q4kCode(weights[g * 64 + l], d * @as(f32, @floatFromInt(sc[2 * g])), dmin * @as(f32, @floatFromInt(mn[2 * g])));
            const hi_code = q4kCode(weights[g * 64 + 32 + l], d * @as(f32, @floatFromInt(sc[2 * g + 1])), dmin * @as(f32, @floatFromInt(mn[2 * g + 1])));
            block.qs[g * 32 + l] = lo_code | (hi_code << 4);
        }
    }
}

/// The 4-bit code whose decoded value `step * q - offset` is closest to `x`.
inline fn q4kCode(x: f32, step: f32, offset: f32) u8 {
    if (step <= 0) return 0;
    return @intFromFloat(std.math.clamp(@round((x + offset) / step), 0.0, 15.0));
}


// -- tests: every encoder's output must decode, with the engine's decoders, to something close to the input --------------

const dequant = @import("dequant.zig");

/// Mean absolute error of encoding `n_blocks` blocks of random normal weights and decoding them with the engine.
fn roundtripError(comptime elems: usize, comptime encodeFn: anytype, comptime decodeFn: anytype, comptime B: type, seed: u64) !f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();
    var total: f64 = 0;
    var count: usize = 0;
    for (0..64) |_| {
        var w: [elems]f32 = undefined;
        for (&w) |*x| x.* = rand.floatNorm(f32);
        var blk: B = std.mem.zeroes(B);
        encodeFn(&w, &blk);
        var out: [elems]f32 = undefined;
        decodeFn(std.mem.asBytes(&blk), &out);
        for (w, out) |a, b| {
            total += @abs(a - b);
            count += 1;
        }
    }
    return @floatCast(total / @as(f64, @floatFromInt(count)));
}

test "Q4_K encoder writes GGML layout: the engine decoder recovers the weights" {
    // 4 bits over a ~6 sigma range: about 0.04 mean absolute error is expected; a wrong layout gives about 0.8.
    const err = try roundtripError(256, quantizeSuperBlockQ4_K, dequant.q4_k, BlockQ4_K, 1);
    try std.testing.expect(err < 0.07);
}

test "Q4_K handles constant, all positive and all zero sub-blocks" {
    var w: [256]f32 = undefined;
    for (0..8) |j| for (0..32) |i| {
        w[j * 32 + i] = switch (j % 4) {
            0 => 0.0, // all zero
            1 => 3.0, // constant positive
            2 => -2.5, // constant negative
            else => @as(f32, @floatFromInt(i)) * 0.1 + 0.5, // positive ramp
        };
    };
    var blk: BlockQ4_K = std.mem.zeroes(BlockQ4_K);
    quantizeSuperBlockQ4_K(&w, &blk);
    var out: [256]f32 = undefined;
    dequant.q4_k(std.mem.asBytes(&blk), &out);
    for (w, out) |a, b| try std.testing.expect(std.math.isFinite(b) and @abs(a - b) < 0.4);
}

test "the other K and legacy encoders decode accurately with the engine decoders" {
    // Bounds are loose enough for the format's bit width and tight enough that a layout error (about 0.8) fails.
    try std.testing.expect(try roundtripError(32, quantizeBlockQ4_0, dequant.q4_0, BlockQ4_0, 2) < 0.09);
    try std.testing.expect(try roundtripError(32, quantizeBlockQ8_0, dequant.q8_0, BlockQ8_0, 3) < 0.01);
    try std.testing.expect(try roundtripError(256, quantizeSuperBlockQ5_K, dequant.q5_k, BlockQ5_K, 4) < 0.05);
    try std.testing.expect(try roundtripError(256, quantizeSuperBlockQ3_K, dequant.q3_k, BlockQ3_K, 5) < 0.2);
    try std.testing.expect(try roundtripError(256, quantizeSuperBlockQ6_K, dequant.q6_k, BlockQ6_K, 6) < 0.02);
    try std.testing.expect(try roundtripError(256, quantizeSuperBlockQ2_K, dequant.q2_k, BlockQ2_K, 7) < 0.4);
    try std.testing.expect(try roundtripError(256, quantizeSuperBlockQ8_K, dequant.q8_k, BlockQ8_K, 8) < 0.01);
}

/// Worst per sub-block mean absolute error divided by that sub-block's amplitude, for input whose sub-blocks differ
/// in magnitude by up to 8x. A scale stored for or applied to the wrong sub-block shows up as an error of order 1 or
/// more here, which a whole-block average over identically distributed data hides.
fn subBlockError(comptime sub: usize, comptime encodeFn: anytype, comptime decodeFn: anytype, comptime B: type, seed: u64) !f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();
    var worst: f32 = 0;
    for (0..32) |_| {
        var w: [256]f32 = undefined;
        for (0..256 / sub) |j| {
            const amp: f32 = @floatFromInt(@as(u32, 1) << @intCast((j * 7) % 4)); // 1, 2, 4, 8 in a scrambled order
            for (0..sub) |i| w[j * sub + i] = amp * rand.floatNorm(f32);
        }
        var blk: B = std.mem.zeroes(B);
        encodeFn(&w, &blk);
        var out: [256]f32 = undefined;
        decodeFn(std.mem.asBytes(&blk), &out);
        for (0..256 / sub) |j| {
            const amp: f32 = @floatFromInt(@as(u32, 1) << @intCast((j * 7) % 4));
            var e: f32 = 0;
            for (0..sub) |i| e += @abs(w[j * sub + i] - out[j * sub + i]);
            worst = @max(worst, e / sub / amp);
        }
    }
    return worst;
}

test "sub-blocks of very different magnitude round-trip independently (scales land on the right sub-block)" {
    // Measured with the correct encoders: Q4_K 0.11, Q5_K 0.06, Q3_K 0.28, Q6_K 0.04, Q2_K 0.63. A scale applied to
    // the wrong sub-block gives errors of order the 8x amplitude ratio.
    try std.testing.expect(try subBlockError(32, quantizeSuperBlockQ4_K, dequant.q4_k, BlockQ4_K, 1) < 0.25);
    try std.testing.expect(try subBlockError(32, quantizeSuperBlockQ5_K, dequant.q5_k, BlockQ5_K, 2) < 0.15);
    try std.testing.expect(try subBlockError(16, quantizeSuperBlockQ3_K, dequant.q3_k, BlockQ3_K, 3) < 0.5);
    try std.testing.expect(try subBlockError(16, quantizeSuperBlockQ6_K, dequant.q6_k, BlockQ6_K, 4) < 0.1);
    try std.testing.expect(try subBlockError(16, quantizeSuperBlockQ2_K, dequant.q2_k, BlockQ2_K, 5) < 1.2);
}
