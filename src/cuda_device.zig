//! Device code for the CUDA backend. Compiled for nvptx64 only (never imported by host code); `build.zig` turns it
//! into PTX that the host embeds and the driver JIT-compiles for whatever GPU is present, so building HK needs no
//! CUDA toolkit.
//!
//! Weight formats are not re-implemented here. A matrix-vector kernel is instantiated per storage type at comptime
//! and calls the same exact block decoders the CPU engine is tested against (`quant/dequant.zig`), so a format that
//! the CPU decodes correctly decodes identically on the device. That is correctness first (plan §152); the kernels
//! are deliberately simple (one block of threads per output row) and are optimized only after they are right.

const std = @import("std");
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

// -- matrix-vector product ---------------------------------------------------------------------------------------

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

/// y[row] = dot(W[row, :], x) for one row per thread block. `w` holds `k / elems` blocks per row, row after row.
inline fn gemvRow(comptime t: format.StorageType, w: [*]const u8, x: [*]const f32, y: [*]f32, k: u32) void {
    const info = comptime blocks.info(t).?;
    const row = bid();
    const nb: usize = k / info.elems;
    const row_bytes = w + @as(usize, row) * nb * info.bytes;
    var acc: f32 = 0;
    var b: usize = tid();
    while (b < nb) : (b += bdim()) {
        var tmp: [info.elems]f32 = undefined;
        decodeBlock(t, row_bytes[b * info.bytes ..][0..info.bytes], &tmp);
        const xs = x + b * info.elems;
        inline for (0..info.elems) |i| acc += tmp[i] * xs[i];
    }
    const total = blockSum(acc);
    if (tid() == 0) y[row] = total;
}

/// Storage types the device multiplies, in the order of the CPU's `vecdot.supported`.
pub const gemv_formats = [_]format.StorageType{
    .f32,    .f16,     .bf16,   .q4_0,    .q4_1,   .q5_0,    .q5_1, .q8_0, .iq4_nl, .mxfp4, .nvfp4,
    .q2_k,   .q3_k,    .q4_k,   .q5_k,    .q6_k,   .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s,
    .iq1_s,  .iq1_m,   .iq4_xs, .tq1_0,   .tq2_0,
};

fn GemvKernel(comptime t: format.StorageType) type {
    return struct {
        fn run(w: [*]const u8, x: [*]const f32, y: [*]f32, k: u32) callconv(.kernel) void {
            gemvRow(t, w, x, y, k);
        }
    };
}

comptime {
    for (gemv_formats) |t| @export(&GemvKernel(t).run, .{ .name = "gemv_" ++ @tagName(t) });
}

// -- normalization, rotary embedding, cache, attention, activation ------------------------------------------------

/// out = x * w / rms(x) over one vector of `n`. Launch one block. `out` may alias `x`.
fn rmsNormKernel(out: [*]f32, x: [*]const f32, w: [*]const f32, n: u32, eps: f32) callconv(.kernel) void {
    var ss: f32 = 0;
    var i: u32 = tid();
    while (i < n) : (i += bdim()) ss += x[i] * x[i];
    const total = blockSum(ss);
    const inv = 1.0 / @sqrt(total / @as(f32, @floatFromInt(n)) + eps);
    i = tid();
    while (i < n) : (i += bdim()) out[i] = x[i] * inv * w[i];
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

/// Rotates the first `2 * half` dims of every head of `x` in place. `neox` != 0: pairs (i, i + half); else (2i, 2i + 1).
/// One block per head; `cos`/`sin` hold `half` values for this token's position.
fn ropeKernel(x: [*]f32, cos: [*]const f32, sin: [*]const f32, head_dim: u32, half: u32, neox: u32) callconv(.kernel) void {
    const hv = x + @as(usize, bid()) * head_dim;
    var i: u32 = tid();
    while (i < half) : (i += bdim()) {
        const c = cos[i];
        const s = sin[i];
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

/// Stores one token's keys and values (f32, `n_kv * head_dim` each) into the f16 cache of a slot at `pos`.
/// Cache layout per layer: [kv head][position][head_dim].
fn kvWriteKernel(k: [*]const f32, v: [*]const f32, ck: [*]f16, cv: [*]f16, pos: u32, ctx: u32, head_dim: u32, n: u32) callconv(.kernel) void {
    const i = globalId();
    if (i >= n) return;
    const head = i / head_dim;
    const d = i % head_dim;
    const dst = (@as(usize, head) * ctx + pos) * head_dim + d;
    ck[dst] = @floatCast(k[i]);
    cv[dst] = @floatCast(v[i]);
}

/// Largest head width the attention kernel supports (its query lives in shared memory).
const max_head_dim = 256;
var attn_query: [max_head_dim]f32 addrspace(.shared) = undefined;

/// Causal attention for one token, one block per query head: softmax(q . K^T * scale) V over positions 0..pos.
/// `scores` is scratch of `n_heads * ctx` floats.
fn attentionKernel(
    q: [*]const f32,
    ck: [*]const f16,
    cv: [*]const f16,
    out: [*]f32,
    scores: [*]f32,
    pos: u32,
    ctx: u32,
    head_dim: u32,
    group: u32,
    scale: f32,
) callconv(.kernel) void {
    const h = bid();
    const kvh = h / group;
    var d: u32 = tid();
    while (d < head_dim) : (d += bdim()) attn_query[d] = q[@as(usize, h) * head_dim + d];
    barSync();

    const n = pos + 1;
    const sc = scores + @as(usize, h) * ctx;
    const kbase = ck + @as(usize, kvh) * ctx * head_dim;
    const vbase = cv + @as(usize, kvh) * ctx * head_dim;

    var mx: f32 = -std.math.inf(f32);
    var p: u32 = tid();
    while (p < n) : (p += bdim()) {
        const kr = kbase + @as(usize, p) * head_dim;
        var s: f32 = 0;
        var j: u32 = 0;
        while (j < head_dim) : (j += 1) s += attn_query[j] * @as(f32, @floatCast(kr[j]));
        s *= scale;
        sc[p] = s;
        mx = @max(mx, s);
    }
    const gmax = blockMax(mx);

    var sum: f32 = 0;
    p = tid();
    while (p < n) : (p += bdim()) {
        const e = expf(sc[p] - gmax);
        sc[p] = e;
        sum += e;
    }
    const inv = 1.0 / blockSum(sum);

    d = tid();
    while (d < head_dim) : (d += bdim()) {
        var acc: f32 = 0;
        var i: u32 = 0;
        while (i < n) : (i += 1) acc += sc[i] * @as(f32, @floatCast(vbase[@as(usize, i) * head_dim + d]));
        out[@as(usize, h) * head_dim + d] = acc * inv;
    }
}

/// gate[i] = silu(gate[i]) * up[i]
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
    @export(&kvWriteKernel, .{ .name = "kv_write" });
    @export(&attentionKernel, .{ .name = "attention" });
    @export(&swigluKernel, .{ .name = "swiglu" });
    @export(&addKernel, .{ .name = "add" });
}
