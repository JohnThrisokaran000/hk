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
