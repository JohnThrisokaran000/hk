//! Device code for the CUDA backend. Compiled for nvptx64 only (never imported by host code); `build.zig` turns it
//! into PTX that the host embeds and the driver JIT-compiles for whatever GPU is present, so building HK needs no
//! CUDA toolkit.
//!
//! Weight formats are not re-implemented here. A matrix-vector kernel is instantiated per storage type at comptime
//! and calls the same exact block decoders the CPU engine is tested against (`quant/dequant.zig`), so a format that
//! the CPU decodes correctly decodes identically on the device. That is correctness first (plan §152); the kernels
//! are deliberately simple (one block of threads per output row) and are optimized only after they are right.

const std = @import("std");
const builtin = @import("builtin");
const format = @import("format.zig");
const blocks = @import("quant/blocks.zig");
const dequant = @import("quant/dequant.zig");

pub const panic = std.debug.no_panic;

// -- thread intrinsics -------------------------------------------------------------------------------------------

inline fn tid() u32 {
    return asm volatile ("mov.u32 %[r], %%tid.x;"
        : [r] "=r" (-> u32),
    );
}
inline fn bid() u32 {
    return asm volatile ("mov.u32 %[r], %%ctaid.x;"
        : [r] "=r" (-> u32),
    );
}
inline fn bidY() u32 {
    return asm volatile ("mov.u32 %[r], %%ctaid.y;"
        : [r] "=r" (-> u32),
    );
}
inline fn bdim() u32 {
    return asm volatile ("mov.u32 %[r], %%ntid.x;"
        : [r] "=r" (-> u32),
    );
}
inline fn barSync() void {
    asm volatile ("bar.sync 0;" ::: .{ .memory = true });
}
inline fn shflDown(v: f32, off: u32) f32 {
    return asm volatile ("shfl.sync.down.b32 %[r], %[v], %[o], 31, 0xffffffff;"
        : [r] "=f" (-> f32),
        : [v] "f" (v),
          [o] "r" (off),
    );
}

/// One slot per warp, so up to 1024 threads.
var reduce_scratch: [32]f32 addrspace(.shared) = undefined;

/// Sum of `v` over every thread of the block. All threads get the result.
inline fn blockSum(v: f32) f32 {
    var s = v;
    var off: u32 = 16;
    while (off > 0) : (off >>= 1) s += shflDown(s, off);
    const lane = tid() & 31;
    const warp = tid() >> 5;
    if (lane == 0) reduce_scratch[warp] = s;
    barSync();
    var r: f32 = 0;
    if (warp == 0) {
        r = if (lane < ((bdim() + 31) >> 5)) reduce_scratch[lane] else 0;
        off = 16;
        while (off > 0) : (off >>= 1) r += shflDown(r, off);
        if (lane == 0) reduce_scratch[0] = r;
    }
    barSync();
    const total = reduce_scratch[0];
    barSync(); // the scratch may be reused by the next reduction
    return total;
}

/// Sum of `v` over the 32 lanes of a warp; every lane gets the result.
inline fn warpSum(v: f32) f32 {
    var s = v;
    var off: u32 = 16;
    while (off > 0) : (off >>= 1) s += shflDown(s, off);
    // shfl.down leaves the total in lane 0 only; broadcast it.
    return asm volatile ("shfl.sync.idx.b32 %[r], %[v], 0, 31, 0xffffffff;"
        : [r] "=f" (-> f32),
        : [v] "f" (s),
    );
}

/// Maximum of `v` over every thread of the block. All threads get the result.
inline fn blockMax(v: f32) f32 {
    var m = v;
    var off: u32 = 16;
    while (off > 0) : (off >>= 1) m = @max(m, shflDown(m, off));
    const lane = tid() & 31;
    const warp = tid() >> 5;
    if (lane == 0) reduce_scratch[warp] = m;
    barSync();
    var r: f32 = -std.math.inf(f32);
    if (warp == 0) {
        r = if (lane < ((bdim() + 31) >> 5)) reduce_scratch[lane] else -std.math.inf(f32);
        off = 16;
        while (off > 0) : (off >>= 1) r = @max(r, shflDown(r, off));
        if (lane == 0) reduce_scratch[0] = r;
    }
    barSync();
    const total = reduce_scratch[0];
    barSync();
    return total;
}

/// e^x through the hardware 2^x instruction (NVPTX has no libm). Relative error is a few ulp, the same order as the
/// polynomial the CPU engine uses.
inline fn expf(x: f32) f32 {
    const y = x * 1.4426950408889634;
    return asm volatile ("ex2.approx.ftz.f32 %[r], %[v];"
        : [r] "=f" (-> f32),
        : [v] "f" (y),
    );
}

inline fn globalId() u32 {
    return bid() * bdim() + tid();
}

// -- matrix product ---------------------------------------------------------------------------------------------

/// Decodes one block of `t` into `out`. Dense float types have one element per "block".
inline fn decodeBlock(comptime t: format.StorageType, src: []const u8, out: *[blocks.info(t).?.elems]f32) void {
    switch (t) {
        .f32 => out[0] = @bitCast(std.mem.readInt(u32, src[0..4], .little)),
        .f16 => out[0] = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, src[0..2], .little)))),
        .bf16 => out[0] = @bitCast(@as(u32, std.mem.readInt(u16, src[0..2], .little)) << 16),
        .q4_0 => dequant.q4_0(src, out),
        .q4_1 => dequant.q4_1(src, out),
        .q5_0 => dequant.q5_0(src, out),
        .q5_1 => dequant.q5_1(src, out),
        .q8_0 => dequant.q8_0(src, out),
        .iq4_nl => dequant.iq4_nl(src, out),
        .mxfp4 => dequant.mxfp4(src, out),
        .nvfp4 => dequant.nvfp4(src, out),
        .q2_k => dequant.q2_k(src, out),
        .q3_k => dequant.q3_k(src, out),
        .q4_k => dequant.q4_k(src, out),
        .q5_k => dequant.q5_k(src, out),
        .q6_k => dequant.q6_k(src, out),
        .iq2_xxs => dequant.iq2_xxs(src, out),
        .iq2_xs => dequant.iq2_xs(src, out),
        .iq2_s => dequant.iq2_s(src, out),
        .iq3_xxs => dequant.iq3_xxs(src, out),
        .iq3_s => dequant.iq3_s(src, out),
        .iq1_s => dequant.iq1_s(src, out),
        .iq1_m => dequant.iq1_m(src, out),
        .iq4_xs => dequant.iq4_xs(src, out),
        .tq1_0 => dequant.tq1_0(src, out),
        .tq2_0 => dequant.tq2_0(src, out),
        else => @compileError("no device decoder for " ++ @tagName(t)),
    }
}

/// Rows one thread block of the decode kernel computes: one warp per row.
const gemv_rows = 4;
/// Activation elements staged in shared memory per step of the decode kernel.
const gemv_chunk = 1024;
/// The staged chunk is padded by one float per 32 so that 32 lanes reading 32 consecutive blocks hit 32 different
/// banks instead of one.
var gemv_x: [gemv_chunk + gemv_chunk / 32]f32 addrspace(.shared) = undefined;

/// Single token matrix-vector product, the decode kernel. A warp owns one output row: its lanes take the row's weight
/// blocks in turn, decode them, and multiply against the activations from shared memory (staged once per block of
/// `gemv_rows` rows). The warp reduces with shuffles, so the only barriers are the ones around the staging.
/// `w` holds `k / elems` blocks per row, row after row.
inline fn gemvRow(comptime t: format.StorageType, w: [*]const u8, x: [*]const f32, y: [*]f32, k: u32, m: u32) void {
    const info = comptime blocks.info(t).?;
    const warp = tid() >> 5;
    const lane = tid() & 31;
    const row = bid() * gemv_rows + warp;
    const valid = row < m;
    const nb_row: usize = k / info.elems;
    const row_bytes = w + @as(usize, row) * nb_row * info.bytes;
    var acc: f32 = 0;

    var k0: usize = 0;
    while (k0 < k) : (k0 += gemv_chunk) {
        const len: usize = @min(@as(usize, gemv_chunk), k - k0);
        var i: usize = tid();
        while (i < len) : (i += bdim()) gemv_x[i + i / 32] = x[k0 + i];
        barSync();
        if (valid) {
            const first = k0 / info.elems;
            const nbc = len / info.elems;
            var b: usize = lane;
            while (b < nbc) : (b += 32) {
                var tmp: [info.elems]f32 = undefined;
                decodeBlock(t, row_bytes[(first + b) * info.bytes ..][0..info.bytes], &tmp);
                const base = b * info.elems;
                var e: usize = 0;
                while (e < info.elems) : (e += 1) acc += tmp[e] * gemv_x[base + e + (base + e) / 32];
            }
        }
        barSync(); // the staged chunk is overwritten by the next step
    }
    var off: u32 = 16;
    while (off > 0) : (off >>= 1) acc += shflDown(acc, off);
    if (lane == 0 and valid) y[row] = acc;
}

/// Output rows and tokens of one tile of the batched product, and the slice of the inner dimension staged per step.
/// A block has `tile_rows * tile_tokens` threads, one per output element of the tile.
const tile_rows = 16;
const tile_tokens = 8;
const tile_k = 256;
/// Row stride of the staged tiles: one extra float breaks the power of two stride that would put every thread of a
/// warp on the same shared memory bank.
const tile_stride = tile_k + 1;

var tile_w: [tile_rows * tile_stride]f32 addrspace(.shared) = undefined;
var tile_x: [tile_tokens * tile_stride]f32 addrspace(.shared) = undefined;

/// Decodes `unit` consecutive weights starting at `src` into `out`. Dense types decode element by element.
inline fn decodeUnit(comptime t: format.StorageType, comptime unit: usize, src: []const u8, out: *[unit]f32) void {
    const info = comptime blocks.info(t).?;
    if (comptime info.elems == unit) {
        decodeBlock(t, src[0..info.bytes], out);
    } else {
        var e: usize = 0;
        while (e < unit) : (e += 1) {
            var one: [1]f32 = undefined;
            decodeBlock(t, src[e * info.bytes ..][0..info.bytes], &one);
            out[e] = one[0];
        }
    }
}

/// y[token][row] = dot(W[row, :], x[token, :]) for a tile of `tile_rows` rows and `tile_tokens` tokens per block, for
/// batches. The activation tile is staged in shared memory once and reused by every row of the tile (global activation
/// traffic drops by `tile_rows`), and each weight block is decoded once into shared memory and reused by every token.
/// Grid x = row tiles, grid y = token tiles. `x` is [n][k] and `y` is [n][m].
inline fn gemmTile(comptime t: format.StorageType, w: [*]const u8, x: [*]const f32, y: [*]f32, k: u32, m: u32, n: u32) void {
    const info = comptime blocks.info(t).?;
    // Dense types are decoded 32 at a time so every format has work items of at least 32 weights.
    const unit = comptime if (info.elems < 32) 32 else info.elems;
    const unit_bytes = comptime unit / info.elems * info.bytes;
    const units_per_step = comptime tile_k / unit;
    const row0 = bid() * tile_rows;
    const tok0 = bidY() * tile_tokens;
    const row_stride: usize = (@as(usize, k) / info.elems) * info.bytes;
    const r = tid() / tile_tokens;
    const tk = tid() % tile_tokens;
    var acc: f32 = 0;

    var k0: u32 = 0;
    while (k0 < k) : (k0 += tile_k) {
        const len = @min(@as(u32, tile_k), k - k0);

        var idx: u32 = tid();
        while (idx < tile_tokens * tile_k) : (idx += bdim()) {
            const tt = idx / tile_k;
            const kk = idx % tile_k;
            const tok = tok0 + tt;
            tile_x[tt * tile_stride + kk] = if (tok < n and kk < len) x[@as(usize, tok) * k + k0 + kk] else 0;
        }

        idx = tid();
        while (idx < tile_rows * units_per_step) : (idx += bdim()) {
            const rr = idx / units_per_step;
            const off: u32 = (idx % units_per_step) * unit;
            const dst = tile_w[rr * tile_stride + off ..][0..unit];
            if (row0 + rr < m and off < len) {
                var tmp: [unit]f32 = undefined;
                const src = w + @as(usize, row0 + rr) * row_stride + (@as(usize, k0 + off) / info.elems) * info.bytes;
                decodeUnit(t, unit, src[0..unit_bytes], &tmp);
                var e: usize = 0;
                while (e < unit) : (e += 1) dst[e] = tmp[e];
            } else {
                var e: usize = 0;
                while (e < unit) : (e += 1) dst[e] = 0;
            }
        }
        barSync();

        var kk: u32 = 0;
        while (kk < len) : (kk += 1) acc += tile_w[r * tile_stride + kk] * tile_x[tk * tile_stride + kk];
        barSync(); // the tiles are overwritten by the next step
    }
    if (row0 + r < m and tok0 + tk < n) y[@as(usize, tok0 + tk) * m + row0 + r] = acc;
}

// -- tensor core matrix product (sm_80 and newer) --------------------------------------------------------------------

/// `mma.sync.m16n8k16` with f16 operands needs sm_80; the sm_70 module has no such kernels and the host falls back to
/// the tiled kernel above.
pub const has_tensor_cores = builtin.cpu.arch == .nvptx64 and builtin.cpu.has(.nvptx, .sm_80);

/// Output rows and tokens of one block of the tensor core kernel, and the weights per row decoded to f16 in one round.
const tc_rows = 64;
const tc_tokens = 128;
const tc_round = 256;
/// Activations staged per step inside a round.
const tc_step = 32;
/// Row strides of the staged tiles in halves. 8 extra halves put the eight rows a fragment load touches on distinct
/// groups of banks.
const tc_a_ld = tc_round + 8;
const tc_b_ld = tc_step + 8;
/// Threads per block: eight warps, a 2 x 4 grid of 32 x 32 sub-tiles.
pub const tc_threads = 256;

var tc_a: [tc_rows * tc_a_ld]f16 addrspace(.shared) = undefined;
var tc_b: [tc_tokens * tc_b_ld]f16 addrspace(.shared) = undefined;

/// A shared array of halves as an array of 32-bit words (two halves each).
inline fn wordsOf(p: anytype) [*]addrspace(.shared) u32 {
    return @ptrCast(@alignCast(p));
}

inline fn packHalves(lo: f32, hi: f32) u32 {
    const a: f16 = @floatCast(lo);
    const b: f16 = @floatCast(hi);
    return @as(u32, @as(u16, @bitCast(a))) | (@as(u32, @as(u16, @bitCast(b))) << 16);
}

/// d += a * b for one 16 x 8 x 16 tile, accumulating in f32. The operands are fragments in the layout PTX defines:
/// with g = lane / 4 and t = lane % 4, a0..a3 hold A[g][2t..2t+1], A[g+8][2t..], A[g][2t+8..], A[g+8][2t+8..] and b0, b1
/// hold B[2t..2t+1][g], B[2t+8..2t+9][g] as packed halves; d0..d3 are D[g][2t], D[g][2t+1], D[g+8][2t], D[g+8][2t+1].
inline fn mma16816(d: *[4]f32, a: [4]u32, b: [2]u32) void {
    var d0 = d[0];
    var d1 = d[1];
    var d2 = d[2];
    var d3 = d[3];
    asm volatile ("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%[d0], %[d1], %[d2], %[d3]}, {%[a0], %[a1], %[a2], %[a3]}, {%[b0], %[b1]}, {%[d0], %[d1], %[d2], %[d3]};"
        : [d0] "+f" (d0),
          [d1] "+f" (d1),
          [d2] "+f" (d2),
          [d3] "+f" (d3),
        : [a0] "r" (a[0]),
          [a1] "r" (a[1]),
          [a2] "r" (a[2]),
          [a3] "r" (a[3]),
          [b0] "r" (b[0]),
          [b1] "r" (b[1]),
    );
    d.* = .{ d0, d1, d2, d3 };
}

/// y[token][row] = dot(W[row, :], x[token, :]) on tensor cores. A block owns `tc_rows` rows and `tc_tokens` tokens.
/// Per round of `tc_round` weights along k, the block decodes its rows to f16 in shared memory (every format goes
/// through the same block decoders as the other kernels), then walks the round in steps of `tc_step`, staging that
/// slice of the activations as f16 and multiplying with `mma.sync`. Operands are rounded to f16; products accumulate
/// in f32. Grid x = row tiles, grid y = token tiles.
inline fn gemmTensor(comptime t: format.StorageType, w: [*]const u8, x: [*]const f16, y: [*]f32, k: u32, m: u32, n: u32) void {
    const info = comptime blocks.info(t).?;
    const unit = comptime if (info.elems < 32) 32 else info.elems;
    const unit_bytes = comptime unit / info.elems * info.bytes;
    const units_per_row = comptime tc_round / unit;
    const row0 = bid() * tc_rows;
    const tok0 = bidY() * tc_tokens;
    const row_stride: usize = (@as(usize, k) / info.elems) * info.bytes;
    const warp = tid() >> 5;
    const lane = tid() & 31;
    const g = lane >> 2;
    const q = lane & 3;
    const wr = (warp & 1) * 32;
    const wc = (warp >> 1) * 32;

    var acc: [2][4][4]f32 = undefined;
    for (&acc) |*a| for (a) |*b| {
        b.* = .{ 0, 0, 0, 0 };
    };

    var k0: u32 = 0;
    while (k0 < k) : (k0 += tc_round) {
        const len = @min(@as(u32, tc_round), k - k0);

        // Decode this round of weights for all rows of the block.
        var idx: u32 = tid();
        while (idx < tc_rows * units_per_row) : (idx += tc_threads) {
            const rr = idx / units_per_row;
            const off: u32 = (idx % units_per_row) * unit;
            var tmp: [unit]f32 = undefined;
            if (row0 + rr < m and off < len) {
                const src = w + @as(usize, row0 + rr) * row_stride + (@as(usize, k0 + off) / info.elems) * info.bytes;
                decodeUnit(t, unit, src[0..unit_bytes], &tmp);
            } else {
                for (&tmp) |*v| v.* = 0;
            }
            const words = wordsOf(&tc_a) + (rr * tc_a_ld + off) / 2;
            var e: usize = 0;
            while (e < unit) : (e += 2) words[e / 2] = packHalves(tmp[e], tmp[e + 1]);
        }

        // Activations of tokens tok0.. for k in k0 + step .. + tc_step (already f16), `tc_vecs` copies of 16 bytes per
        // thread. They are fetched into registers while the previous step is multiplied, so the global load overlaps
        // the tensor core work; they are parked in shared memory once everyone has finished reading the old ones.
        const tc_vecs = tc_tokens * (tc_step / 8) / tc_threads;
        const stage = struct {
            inline fn fetch(xp: [*]const f16, tok0_: u32, n_: u32, k_: u32, k0_: u32, step_: u32, len_: u32, out: *[tc_vecs]@Vector(4, u32)) void {
                inline for (0..tc_vecs) |u| {
                    const p: u32 = tid() + @as(u32, u) * tc_threads;
                    const tok_ = tok0_ + p / (tc_step / 8);
                    const kq_: u32 = (p % (tc_step / 8)) * 8;
                    out[u] = .{ 0, 0, 0, 0 };
                    if (tok_ < n_ and step_ + kq_ < len_) {
                        const src: *align(2) const @Vector(4, u32) = @ptrCast(xp + @as(usize, tok_) * k_ + k0_ + step_ + kq_);
                        out[u] = src.*;
                    }
                }
            }
            inline fn park(out: *const [tc_vecs]@Vector(4, u32)) void {
                inline for (0..tc_vecs) |u| {
                    const p: u32 = tid() + @as(u32, u) * tc_threads;
                    const tt_ = p / (tc_step / 8);
                    const kq_: u32 = (p % (tc_step / 8)) * 8;
                    const dst: *addrspace(.shared) align(4) @Vector(4, u32) = @ptrCast(wordsOf(&tc_b) + (tt_ * tc_b_ld + kq_) / 2);
                    dst.* = out[u];
                }
            }
        };
        var v: [tc_vecs]@Vector(4, u32) = undefined;
        stage.fetch(x, tok0, n, k, k0, 0, len, &v);
        stage.park(&v);
        barSync(); // the decoded weights and the first activations are ready

        var step: u32 = 0;
        while (step < len) : (step += tc_step) {
            const more = step + tc_step < len;
            if (more) stage.fetch(x, tok0, n, k, k0, step + tc_step, len, &v);

            comptime var half_k = 0;
            inline while (half_k < tc_step) : (half_k += 16) {
                var af: [2][4]u32 = undefined;
                inline for (0..2) |mi| {
                    const r = wr + mi * 16 + g;
                    const kc = step + half_k + q * 2;
                    af[mi][0] = wordsOf(&tc_a)[(r * tc_a_ld + kc) / 2];
                    af[mi][1] = wordsOf(&tc_a)[((r + 8) * tc_a_ld + kc) / 2];
                    af[mi][2] = wordsOf(&tc_a)[(r * tc_a_ld + kc + 8) / 2];
                    af[mi][3] = wordsOf(&tc_a)[((r + 8) * tc_a_ld + kc + 8) / 2];
                }
                inline for (0..4) |ni| {
                    const c = wc + ni * 8 + g;
                    const kc = half_k + q * 2;
                    const bf = [2]u32{
                        wordsOf(&tc_b)[(c * tc_b_ld + kc) / 2],
                        wordsOf(&tc_b)[(c * tc_b_ld + kc + 8) / 2],
                    };
                    inline for (0..2) |mi| mma16816(&acc[mi][ni], af[mi], bf);
                }
            }
            barSync(); // everyone has read this step's activations
            if (more) stage.park(&v);
            barSync();
        }
    }

    inline for (0..2) |mi| inline for (0..4) |ni| {
        const r0 = row0 + wr + mi * 16 + g;
        const c0 = tok0 + wc + ni * 8 + q * 2;
        inline for (0..4) |i| {
            const r = r0 + (i / 2) * 8;
            const c = c0 + (i % 2);
            if (r < m and c < n) y[@as(usize, c) * m + r] = acc[mi][ni][i];
        }
    };
}

fn GemmTensorKernel(comptime t: format.StorageType) type {
    return struct {
        fn run(w: [*]const u8, x: [*]const f16, y: [*]f32, k: u32, m: u32, n: u32) callconv(.kernel) void {
            gemmTensor(t, w, x, y, k, m, n);
        }
    };
}

comptime {
    if (has_tensor_cores) {
        for (gemm_formats) |t| @export(&GemmTensorKernel(t).run, .{ .name = "gemm_tc_" ++ @tagName(t) });
    }
}

// -- tensor core prompt attention (sm_80 and newer) ---------------------------------------------------------------------

/// Positions of keys and values staged per step, and threads per block of the attention kernel.
const fa_positions = 32;
const fa_threads = 128;

fn pairShuffleMax(v: f32) f32 {
    var m = v;
    inline for (.{ 1, 2 }) |off| {
        const o = asm volatile ("shfl.sync.bfly.b32 %[r], %[v], %[o], 31, 0xffffffff;"
            : [r] "=f" (-> f32),
            : [v] "f" (m),
              [o] "r" (@as(u32, off)),
        );
        m = @max(m, o);
    }
    return m;
}

fn quadShuffleSum(v: f32) f32 {
    var m = v;
    inline for (.{ 1, 2 }) |off| {
        const o = asm volatile ("shfl.sync.bfly.b32 %[r], %[v], %[o], 31, 0xffffffff;"
            : [r] "=f" (-> f32),
            : [v] "f" (m),
              [o] "r" (@as(u32, off)),
        );
        m += o;
    }
    return m;
}

/// Causal attention for a prompt on tensor cores (flash attention). A block is four warps; a warp owns 16 query rows of
/// one head, a block 64 consecutive tokens of ONE sequence (the host launches this kernel only for such batches).
/// Keys and values are staged 32 positions at a time (values transposed, so both operands of the second product are
/// read as packed pairs). S = Q K^T and O += P V use `mma.sync` on f16 operands with f32 accumulation; the online
/// softmax works on the accumulator registers directly (a row lives in the four lanes of a quad, so its maximum and
/// sum are two shuffles) and O is rescaled in registers. Head sizes up to 128. Grid x = 64-token tiles, grid y = heads.
inline fn flashAttention(
    comptime hd: u32,
    q: [*]const f32,
    kptrs: [*]const u64,
    vptrs: [*]const u64,
    layer_off: u64,
    slots: [*]const u32,
    poss: [*]const u32,
    out: [*]f32,
    ctx: u32,
    n_tokens: u32,
    n_heads: u32,
    group: u32,
    scale: f32,
) void {
    const kld = hd + 8; // halves per staged key row
    const vld = fa_positions + 8; // halves per staged (transposed) value row
    const warp = tid() >> 5;
    const lane = tid() & 31;
    const g = lane >> 2;
    const qd = lane & 3;
    const h = bidY();
    const t0 = bid() * 64;
    const kvh = h / group;
    const slot = slots[t0];
    const pos0 = poss[t0];
    const t_end = @min(t0 + 64, n_tokens);
    const pos_last = poss[t_end - 1];
    const ck: [*]const f16 = @ptrFromInt(kptrs[slot] + layer_off);
    const cv: [*]const f16 = @ptrFromInt(vptrs[slot] + layer_off);
    const kbase = ck + @as(usize, kvh) * ctx * hd;
    const vbase = cv + @as(usize, kvh) * ctx * hd;

    const ra = t0 + warp * 16 + g; // the two query rows this lane holds results for
    const rb = ra + 8;
    const qpos_a: i64 = if (ra < n_tokens) @as(i64, pos0) + (ra - t0) else -1;
    const qpos_b: i64 = if (rb < n_tokens) @as(i64, pos0) + (rb - t0) else -1;
    const warp_last: i64 = @as(i64, pos0) + (t0 + warp * 16 + 15 - t0);

    // Queries, scaled, as A fragments for every 16-wide slice of the head.
    var qf: [hd / 16][4]u32 = undefined;
    inline for (0..hd / 16) |dk| {
        const c = dk * 16 + qd * 2;
        var v: [4][2]f32 = .{ .{ 0, 0 }, .{ 0, 0 }, .{ 0, 0 }, .{ 0, 0 } };
        if (ra < n_tokens) {
            const base = q + (@as(usize, ra) * n_heads + h) * hd;
            v[0] = .{ base[c] * scale, base[c + 1] * scale };
            v[2] = .{ base[c + 8] * scale, base[c + 9] * scale };
        }
        if (rb < n_tokens) {
            const base = q + (@as(usize, rb) * n_heads + h) * hd;
            v[1] = .{ base[c] * scale, base[c + 1] * scale };
            v[3] = .{ base[c + 8] * scale, base[c + 9] * scale };
        }
        inline for (0..4) |i| qf[dk][i] = packHalves(v[i][0], v[i][1]);
    }

    var o: [hd / 8][4]f32 = undefined;
    for (&o) |*a| a.* = .{ 0, 0, 0, 0 };
    const neg_inf = -std.math.inf(f32);
    var m_a: f32 = neg_inf;
    var m_b: f32 = neg_inf;
    var l_a: f32 = 0;
    var l_b: f32 = 0;

    var j0: u32 = 0;
    while (j0 <= pos_last) : (j0 += fa_positions) {
        barSync();
        // Stage keys and values of positions j0 .. j0 + 31; zero past the last position of the batch, where the cache
        // may hold anything (a probability of 0 times a NaN is NaN).
        var e: u32 = tid();
        while (e < fa_positions * (hd / 8)) : (e += fa_threads) {
            const row = e / (hd / 8);
            const c8 = (e % (hd / 8)) * 8;
            const j = j0 + row;
            var kv4: [4]u32 = .{ 0, 0, 0, 0 };
            var vv4: [4]u32 = .{ 0, 0, 0, 0 };
            if (j <= pos_last) {
                const ks: [*]const u32 = @ptrCast(@alignCast(kbase + @as(usize, j) * hd + c8));
                const vs: [*]const u32 = @ptrCast(@alignCast(vbase + @as(usize, j) * hd + c8));
                kv4 = .{ ks[0], ks[1], ks[2], ks[3] };
                vv4 = .{ vs[0], vs[1], vs[2], vs[3] };
            }
            const kd = wordsOf(&fa_k) + (row * kld + c8) / 2;
            inline for (0..4) |i| kd[i] = kv4[i];
            const vh: [*]addrspace(.shared) u16 = @ptrCast(&fa_vt);
            inline for (0..4) |i| {
                vh[(c8 + 2 * i) * vld + row] = @truncate(vv4[i]);
                vh[(c8 + 2 * i + 1) * vld + row] = @truncate(vv4[i] >> 16);
            }
        }
        barSync();
        if (@as(i64, j0) > warp_last) continue;

        var sacc: [4][4]f32 = undefined;
        for (&sacc) |*a| a.* = .{ 0, 0, 0, 0 };
        inline for (0..hd / 16) |dk| {
            inline for (0..4) |ni| {
                const row = ni * 8 + g;
                const kc = dk * 16 + qd * 2;
                const bf = [2]u32{ wordsOf(&fa_k)[(row * kld + kc) / 2], wordsOf(&fa_k)[(row * kld + kc + 8) / 2] };
                mma16816(&sacc[ni], qf[dk], bf);
            }
        }

        // Causal mask and online softmax of the two rows this lane holds columns of.
        var mx_a: f32 = neg_inf;
        var mx_b: f32 = neg_inf;
        inline for (0..4) |ni| {
            inline for (0..2) |c| {
                const col: i64 = @as(i64, j0) + ni * 8 + qd * 2 + c;
                if (col > qpos_a) sacc[ni][c] = neg_inf;
                if (col > qpos_b) sacc[ni][2 + c] = neg_inf;
                mx_a = @max(mx_a, sacc[ni][c]);
                mx_b = @max(mx_b, sacc[ni][2 + c]);
            }
        }
        mx_a = pairShuffleMax(mx_a);
        mx_b = pairShuffleMax(mx_b);
        const new_a = @max(m_a, mx_a);
        const new_b = @max(m_b, mx_b);
        // A row with nothing visible yet keeps m = -inf; its probabilities are 0 and nothing needs rescaling.
        const corr_a: f32 = if (new_a == neg_inf) 1 else expf(m_a - new_a);
        const corr_b: f32 = if (new_b == neg_inf) 1 else expf(m_b - new_b);
        m_a = new_a;
        m_b = new_b;
        l_a *= corr_a;
        l_b *= corr_b;
        inline for (0..hd / 8) |nt| {
            o[nt][0] *= corr_a;
            o[nt][1] *= corr_a;
            o[nt][2] *= corr_b;
            o[nt][3] *= corr_b;
        }
        inline for (0..4) |ni| {
            inline for (0..2) |c| {
                const pa: f32 = if (new_a == neg_inf) 0 else expf(sacc[ni][c] - new_a);
                const pb: f32 = if (new_b == neg_inf) 0 else expf(sacc[ni][2 + c] - new_b);
                sacc[ni][c] = pa;
                sacc[ni][2 + c] = pb;
                l_a += pa;
                l_b += pb;
            }
        }

        // O += P V: the accumulator layout of two neighbouring 8-column tiles is the A fragment of a 16-wide slice.
        inline for (0..2) |kc| {
            const pf = [4]u32{
                packHalves(sacc[2 * kc][0], sacc[2 * kc][1]),
                packHalves(sacc[2 * kc][2], sacc[2 * kc][3]),
                packHalves(sacc[2 * kc + 1][0], sacc[2 * kc + 1][1]),
                packHalves(sacc[2 * kc + 1][2], sacc[2 * kc + 1][3]),
            };
            inline for (0..hd / 8) |nt| {
                const row = nt * 8 + g;
                const kcol = kc * 16 + qd * 2;
                const bf = [2]u32{ wordsOf(&fa_vt)[(row * vld + kcol) / 2], wordsOf(&fa_vt)[(row * vld + kcol + 8) / 2] };
                mma16816(&o[nt], pf, bf);
            }
        }
    }

    const sum_a = quadShuffleSum(l_a);
    const sum_b = quadShuffleSum(l_b);
    const inv_a: f32 = if (sum_a > 0) 1.0 / sum_a else 0;
    const inv_b: f32 = if (sum_b > 0) 1.0 / sum_b else 0;
    inline for (0..hd / 8) |nt| {
        const c = nt * 8 + qd * 2;
        if (ra < n_tokens) {
            const dst = out + (@as(usize, ra) * n_heads + h) * hd + c;
            dst[0] = o[nt][0] * inv_a;
            dst[1] = o[nt][1] * inv_a;
        }
        if (rb < n_tokens) {
            const dst = out + (@as(usize, rb) * n_heads + h) * hd + c;
            dst[0] = o[nt][2] * inv_b;
            dst[1] = o[nt][3] * inv_b;
        }
    }
}

var fa_k: [fa_positions * (128 + 8)]f16 addrspace(.shared) = undefined;
var fa_vt: [128 * (fa_positions + 8)]f16 addrspace(.shared) = undefined;

fn FlashKernel(comptime hd: u32) type {
    return struct {
        fn run(
            q: [*]const f32,
            kptrs: [*]const u64,
            vptrs: [*]const u64,
            layer_off: u64,
            slots: [*]const u32,
            poss: [*]const u32,
            out: [*]f32,
            ctx: u32,
            n_tokens: u32,
            n_heads: u32,
            group: u32,
            scale: f32,
        ) callconv(.kernel) void {
            flashAttention(hd, q, kptrs, vptrs, layer_off, slots, poss, out, ctx, n_tokens, n_heads, group, scale);
        }
    };
}

comptime {
    if (has_tensor_cores) {
        @export(&FlashKernel(64).run, .{ .name = "attention_tc_64" });
        @export(&FlashKernel(128).run, .{ .name = "attention_tc_128" });
    }
}

/// Storage types the device multiplies, in the order of the CPU's `vecdot.supported`.
pub const gemm_formats = [_]format.StorageType{
    .f32,   .f16,   .bf16,     .q4_0,    .q4_1,    .q5_0, .q5_1, .q8_0, .iq4_nl, .mxfp4, .nvfp4,
    .q2_k,  .q3_k,  .q4_k,     .q5_k,    .q6_k,    .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s,
    .iq1_s, .iq1_m, .iq4_xs,   .tq1_0,   .tq2_0,
};

fn GemvKernel(comptime t: format.StorageType) type {
    return struct {
        fn run(w: [*]const u8, x: [*]const f32, y: [*]f32, k: u32, m: u32, n: u32) callconv(.kernel) void {
            _ = n;
            gemvRow(t, w, x, y, k, m);
        }
    };
}

fn GemmTileKernel(comptime t: format.StorageType) type {
    return struct {
        fn run(w: [*]const u8, x: [*]const f32, y: [*]f32, k: u32, m: u32, n: u32) callconv(.kernel) void {
            gemmTile(t, w, x, y, k, m, n);
        }
    };
}

comptime {
    for (gemm_formats) |t| {
        @export(&GemvKernel(t).run, .{ .name = "gemv_" ++ @tagName(t) });
        @export(&GemmTileKernel(t).run, .{ .name = "gemm_" ++ @tagName(t) });
    }
}

// -- normalization, rotary embedding, cache, attention, activation ------------------------------------------------

/// out = x * w / rms(x) for each of `n` rows of `dim`; one block per row. `out` may alias `x`.
fn rmsNormKernel(out: [*]f32, x: [*]const f32, w: [*]const f32, dim: u32, eps: f32) callconv(.kernel) void {
    const xr = x + @as(usize, bid()) * dim;
    const outr = out + @as(usize, bid()) * dim;
    var ss: f32 = 0;
    var i: u32 = tid();
    while (i < dim) : (i += bdim()) ss += xr[i] * xr[i];
    const total = blockSum(ss);
    const inv = 1.0 / @sqrt(total / @as(f32, @floatFromInt(dim)) + eps);
    i = tid();
    while (i < dim) : (i += bdim()) outr[i] = xr[i] * inv * w[i];
}

/// In place RMS norm of every `head_dim` slice of `x` (Qwen3 q/k norm), one block per head, weight shared by heads.
fn headNormKernel(x: [*]f32, w: [*]const f32, head_dim: u32, eps: f32) callconv(.kernel) void {
    const hv = x + @as(usize, bid()) * head_dim;
    var ss: f32 = 0;
    var i: u32 = tid();
    while (i < head_dim) : (i += bdim()) ss += hv[i] * hv[i];
    const total = blockSum(ss);
    const inv = 1.0 / @sqrt(total / @as(f32, @floatFromInt(head_dim)) + eps);
    i = tid();
    while (i < head_dim) : (i += bdim()) hv[i] = hv[i] * inv * w[i];
}

/// Rotates the first `2 * half` dims of every head of `x` in place. One block per head; the block's token is
/// `bid / heads_per_row`, whose `half` cos/sin values start at `cos + token * half`. `neox` != 0: pairs (i, i + half);
/// else (2i, 2i + 1).
fn ropeKernel(x: [*]f32, cos: [*]const f32, sin: [*]const f32, head_dim: u32, half: u32, neox: u32, heads_per_row: u32) callconv(.kernel) void {
    const hv = x + @as(usize, bid()) * head_dim;
    const row = bid() / heads_per_row;
    const cr = cos + @as(usize, row) * half;
    const sr = sin + @as(usize, row) * half;
    var i: u32 = tid();
    while (i < half) : (i += bdim()) {
        const c = cr[i];
        const s = sr[i];
        if (neox != 0) {
            const a = hv[i];
            const b = hv[i + half];
            hv[i] = a * c - b * s;
            hv[i + half] = a * s + b * c;
        } else {
            const a = hv[2 * i];
            const b = hv[2 * i + 1];
            hv[2 * i] = a * c - b * s;
            hv[2 * i + 1] = a * s + b * c;
        }
    }
}

/// x[i] += b[i % len] for i < total: a bias added to every row.
fn addBiasKernel(x: [*]f32, b: [*]const f32, len: u32, total: u32) callconv(.kernel) void {
    const i = globalId();
    if (i >= total) return;
    x[i] += b[i % len];
}

/// Stores each token's keys and values (f32, `kv_dim` each) into the f16 cache of its slot at its position.
/// `kptrs`/`vptrs` hold the base address of each slot's cache; `layer_off` is the byte offset of this layer in it.
/// Cache layout per layer: [kv head][position][head_dim].
fn kvWriteKernel(
    k: [*]const f32,
    v: [*]const f32,
    kptrs: [*]const u64,
    vptrs: [*]const u64,
    layer_off: u64,
    slots: [*]const u32,
    poss: [*]const u32,
    ctx: u32,
    head_dim: u32,
    kv_dim: u32,
    total: u32,
) callconv(.kernel) void {
    const i = globalId();
    if (i >= total) return;
    const row = i / kv_dim;
    const j = i % kv_dim;
    const head = j / head_dim;
    const d = j % head_dim;
    const ck: [*]f16 = @ptrFromInt(kptrs[slots[row]] + layer_off);
    const cv: [*]f16 = @ptrFromInt(vptrs[slots[row]] + layer_off);
    const dst = (@as(usize, head) * ctx + poss[row]) * head_dim + d;
    ck[dst] = @floatCast(k[i]);
    cv[dst] = @floatCast(v[i]);
}

/// Largest head width the attention kernel supports (its query lives in shared memory).
const max_head_dim = 256;
/// Positions per attention tile (one per thread).
const attn_tile = 128;
var attn_query: [max_head_dim]f32 addrspace(.shared) = undefined;
var attn_probs: [attn_tile]f32 addrspace(.shared) = undefined;
var attn_scores: [attn_tile]f32 addrspace(.shared) = undefined;

/// Running softmax state of one (token, head) over a range of positions: the maximum score, the sum of exponentials
/// relative to it, and the weighted sum of values for the output dimensions this thread owns.
const AttnState = struct { m: f32, l: f32, acc0: f32, acc1: f32 };

/// Online softmax attention of one (token, head) over positions `lo .. hi` of the token's slot, computed tile by tile so
/// no per position scores are stored. All threads of the block must call it together (it uses barriers). Launch with
/// 128 threads (`attn_tile`); head widths up to 256.
inline fn attendRange(
    q: [*]const f32,
    kptrs: [*]const u64,
    vptrs: [*]const u64,
    layer_off: u64,
    slots: [*]const u32,
    row: u32,
    h: u32,
    ctx: u32,
    head_dim: u32,
    n_heads: u32,
    group: u32,
    scale: f32,
    lo: u32,
    hi: u32,
) AttnState {
    const kvh = h / group;
    const ck: [*]const f16 = @ptrFromInt(kptrs[slots[row]] + layer_off);
    const cv: [*]const f16 = @ptrFromInt(vptrs[slots[row]] + layer_off);
    const kbase = ck + @as(usize, kvh) * ctx * head_dim;
    const vbase = cv + @as(usize, kvh) * ctx * head_dim;
    const qh = q + (@as(usize, row) * n_heads + h) * head_dim;

    var d: u32 = tid();
    while (d < head_dim) : (d += bdim()) attn_query[d] = qh[d];
    barSync();

    const neg_inf = -std.math.inf(f32);
    var st = AttnState{ .m = neg_inf, .l = 0, .acc0 = 0, .acc1 = 0 };
    // Each thread owns output dimensions tid and tid + 128 (head widths up to 256).
    const d0 = tid();
    const d1 = tid() + attn_tile;
    const warp = tid() >> 5;
    const lane = tid() & 31;

    var p0: u32 = lo;
    while (p0 < hi) : (p0 += attn_tile) {
        const p = p0 + tid();
        // Scores of the tile: a warp works on one position at a time with its lanes spread over the head dimensions, so
        // the key row is read coalesced (a thread per position would read rows a whole head width apart).
        var qi: u32 = warp;
        while (qi < attn_tile) : (qi += attn_tile / 32) {
            const pp = p0 + qi;
            var sc: f32 = neg_inf;
            if (pp < hi) {
                const kr = kbase + @as(usize, pp) * head_dim;
                var part: f32 = 0;
                var j: u32 = lane;
                while (j < head_dim) : (j += 32) part += attn_query[j] * @as(f32, @floatCast(kr[j]));
                sc = warpSum(part) * scale;
            }
            if (lane == 0) attn_scores[qi] = sc;
        }
        barSync();
        const s: f32 = attn_scores[tid()];
        const m_new = @max(st.m, blockMax(s));
        const corr = expf(st.m - m_new); // 0 on the first tile, where m is -inf
        const pj = if (p < hi) expf(s - m_new) else 0;
        attn_probs[tid()] = pj;
        const tile_sum = blockSum(pj); // also publishes attn_probs to the block
        st.l = st.l * corr + tile_sum;
        st.acc0 *= corr;
        st.acc1 *= corr;
        const live = @min(@as(u32, attn_tile), hi - p0);
        var jj: u32 = 0;
        while (jj < live) : (jj += 1) {
            const pr = attn_probs[jj];
            const vr = vbase + @as(usize, p0 + jj) * head_dim;
            if (d0 < head_dim) st.acc0 += pr * @as(f32, @floatCast(vr[d0]));
            if (d1 < head_dim) st.acc1 += pr * @as(f32, @floatCast(vr[d1]));
        }
        barSync(); // attn_probs and attn_scores are rewritten by the next tile
        st.m = m_new;
    }
    return st;
}

/// Causal attention, one block per (token, query head): softmax(q . K^T * scale) V over positions 0..pos of the token's
/// slot. Used when there are enough (token, head) blocks to fill the device (prompt processing).
fn attentionKernel(
    q: [*]const f32,
    kptrs: [*]const u64,
    vptrs: [*]const u64,
    layer_off: u64,
    slots: [*]const u32,
    poss: [*]const u32,
    out: [*]f32,
    ctx: u32,
    head_dim: u32,
    n_heads: u32,
    group: u32,
    scale: f32,
) callconv(.kernel) void {
    const row = bid() / n_heads;
    const h = bid() % n_heads;
    const st = attendRange(q, kptrs, vptrs, layer_off, slots, row, h, ctx, head_dim, n_heads, group, scale, 0, poss[row] + 1);
    const inv = 1.0 / st.l;
    const o = out + (@as(usize, row) * n_heads + h) * head_dim;
    if (tid() < head_dim) o[tid()] = st.acc0 * inv;
    if (tid() + attn_tile < head_dim) o[tid() + attn_tile] = st.acc1 * inv;
}

/// Split-KV attention, first half: block (token, head, split) covers one contiguous slice of the token's positions and
/// writes its partial softmax state to `part` as [acc (head_dim), m, l]. For few (token, head) blocks, as in decode,
/// slicing the context across blocks is what keeps the device busy.
fn attentionPartKernel(
    q: [*]const f32,
    kptrs: [*]const u64,
    vptrs: [*]const u64,
    layer_off: u64,
    slots: [*]const u32,
    poss: [*]const u32,
    part: [*]f32,
    ctx: u32,
    head_dim: u32,
    n_heads: u32,
    group: u32,
    scale: f32,
    splits: u32,
) callconv(.kernel) void {
    const split = bid() % splits;
    const rh = bid() / splits;
    const row = rh / n_heads;
    const h = rh % n_heads;
    const n = poss[row] + 1;
    const chunk = (n + splits - 1) / splits;
    const lo = @min(n, split * chunk);
    const hi = @min(n, lo + chunk);
    const st = attendRange(q, kptrs, vptrs, layer_off, slots, row, h, ctx, head_dim, n_heads, group, scale, lo, hi);
    const o = part + @as(usize, bid()) * (head_dim + 2);
    if (tid() < head_dim) o[tid()] = st.acc0;
    if (tid() + attn_tile < head_dim) o[tid() + attn_tile] = st.acc1;
    if (tid() == 0) {
        o[head_dim] = st.m;
        o[head_dim + 1] = st.l;
    }
}

/// Split-KV attention, second half: merges the `splits` partial states of each (token, head) exactly (rescaling each by
/// its maximum) and writes the normalized output. One block per (token, head).
fn attentionCombineKernel(part: [*]const f32, out: [*]f32, head_dim: u32, splits: u32) callconv(.kernel) void {
    const rh = bid();
    const base = part + @as(usize, rh) * splits * (head_dim + 2);
    var gm: f32 = -std.math.inf(f32);
    var s: u32 = 0;
    while (s < splits) : (s += 1) gm = @max(gm, base[s * (head_dim + 2) + head_dim]);
    var l: f32 = 0;
    s = 0;
    while (s < splits) : (s += 1) {
        const o = base + s * (head_dim + 2);
        l += o[head_dim + 1] * expf(o[head_dim] - gm);
    }
    var d: u32 = tid();
    while (d < head_dim) : (d += bdim()) {
        var acc: f32 = 0;
        s = 0;
        while (s < splits) : (s += 1) {
            const o = base + s * (head_dim + 2);
            acc += o[d] * expf(o[head_dim] - gm);
        }
        out[@as(usize, rh) * head_dim + d] = acc / l;
    }
}

/// gate[i] = silu(gate[i]) * up[i]
/// dst[i] = f16(src[i]) for pairs of floats; the matrix kernels on tensor cores take their activations as halves.
fn toHalfKernel(dst: [*]u32, src: [*]const f32, pairs: u32) callconv(.kernel) void {
    const i = globalId();
    if (i < pairs) dst[i] = packHalves(src[2 * i], src[2 * i + 1]);
}

fn swigluKernel(gate: [*]f32, up: [*]const f32, n: u32) callconv(.kernel) void {
    const i = globalId();
    if (i >= n) return;
    const g = gate[i];
    gate[i] = (g / (1.0 + expf(-g))) * up[i];
}

/// x[i] += y[i]
fn addKernel(x: [*]f32, y: [*]const f32, n: u32) callconv(.kernel) void {
    const i = globalId();
    if (i >= n) return;
    x[i] += y[i];
}

comptime {
    @export(&rmsNormKernel, .{ .name = "rms_norm" });
    @export(&headNormKernel, .{ .name = "head_norm" });
    @export(&ropeKernel, .{ .name = "rope" });
    @export(&addBiasKernel, .{ .name = "add_bias" });
    @export(&kvWriteKernel, .{ .name = "kv_write" });
    @export(&attentionKernel, .{ .name = "attention" });
    @export(&attentionPartKernel, .{ .name = "attention_part" });
    @export(&attentionCombineKernel, .{ .name = "attention_combine" });
    @export(&swigluKernel, .{ .name = "swiglu" });
    @export(&toHalfKernel, .{ .name = "to_half" });
    @export(&addKernel, .{ .name = "add" });
}
