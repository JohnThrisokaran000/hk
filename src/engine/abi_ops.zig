//! Matrix products for the C ABI and the Python bindings, on the engine's kernels.
//!
//! There is one set of CPU kernels in this repository (`quant/` and `kernels/`); the C ABI no longer carries its own.
//! A product here is therefore the product `hk run` computes: quantized weights are multiplied against activations
//! quantized to 8 bits (about 0.4 % of the term magnitude, the same trade llama.cpp makes), dense f32/f16/bf16 weights
//! against f32 activations. Unlike the lenient routines these replace, sizes are validated and a mismatch is an error.
//!
//! The thread pool is created on first use and shared. Calls are serialized by a mutex because the pool runs one job
//! at a time and the C ABI may be entered from several threads (Python threads, for instance).

const std = @import("std");
const format = @import("../format.zig");
const quant = @import("../quant.zig");
const pool_mod = @import("../pool.zig");
const cores = @import("../cores.zig");
const kernels = @import("../kernels.zig");
const matmul = @import("matmul.zig");
const weights = @import("weights.zig");

const vecdot = quant.vecdot;
const StorageType = format.StorageType;

pub const Error = error{
    /// The engine has no kernel for this storage type.
    UnsupportedType,
    /// `k` is not a whole number of blocks, or a slice length disagrees with the shape.
    BadShape,
    OutOfMemory,
    PoolFailed,
};

const Mutex = struct {
    state: std.atomic.Value(u32) = .init(0),

    fn lock(self: *Mutex) void {
        while (self.state.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }

    fn unlock(self: *Mutex) void {
        self.state.store(0, .release);
    }
};

var mutex: Mutex = .{};
var shared_pool: ?pool_mod.Pool = null;

fn poolLocked() Error!*pool_mod.Pool {
    if (shared_pool == null) {
        const threads = cores.physicalCores(std.Options.debug_io);
        shared_pool = pool_mod.Pool.init(std.heap.page_allocator, threads) catch return error.PoolFailed;
    }
    return &shared_pool.?;
}

/// Bytes of `rows` rows of `cols` weights stored as `t`, or an error if `cols` is not a whole number of blocks.
pub fn matrixBytes(t: StorageType, rows: usize, cols: usize) Error!usize {
    const info = quant.info(t) orelse return error.UnsupportedType;
    if (!vecdot.supported(t)) return error.UnsupportedType;
    if (cols % info.elems != 0) return error.BadShape;
    return rows * (cols / info.elems) * info.bytes;
}

/// y[token * m + row] = dot(W[row], x[token]) + bias[row] for `n` tokens. `w` is `m` rows of `k` weights stored as `t`
/// (the container's layout), `x` is [n][k] and `y` is [n][m].
pub fn gemm(
    t: StorageType,
    w: []const u8,
    x: []const f32,
    bias: ?[]const f32,
    y: []f32,
    m: usize,
    k: usize,
    n: usize,
) Error!void {
    const need = try matrixBytes(t, m, k);
    if (w.len < need or x.len < n * k or y.len < n * m) return error.BadShape;
    if (bias) |b| if (b.len < m) return error.BadShape;
    if (m == 0 or n == 0) return;

    const allocator = std.heap.page_allocator;
    const a8_stride = k / 32 + 1;
    const q8k_stride = k / 256 + 1;
    const acts = allocator.alloc(vecdot.Act, n) catch return error.OutOfMemory;
    defer allocator.free(acts);
    const a8 = allocator.alloc(vecdot.BlockA8, n * a8_stride) catch return error.OutOfMemory;
    defer allocator.free(a8);
    const q8k = allocator.alloc(vecdot.BlockQ8K, n * q8k_stride) catch return error.OutOfMemory;
    defer allocator.free(q8k);

    _ = kernels.get();
    const kind = vecdot.actKind(t);
    for (acts, 0..) |*a, i| {
        a.* = vecdot.Act.init(x[i * k ..][0..k], a8[i * a8_stride ..], q8k[i * q8k_stride ..]);
        a.prepare(kind);
    }

    const mat = weights.Mat{ .data = w[0..need], .t = t, .rows = m, .cols = k };
    mutex.lock();
    defer mutex.unlock();
    const pool = try poolLocked();
    matmul.run(pool, mat, acts, y[0 .. n * m], if (bias) |b| b[0..m] else null);
}

/// y[row] = dot(W[row], x) + bias[row]. See `gemm`.
pub fn gemv(t: StorageType, w: []const u8, x: []const f32, bias: ?[]const f32, y: []f32, m: usize, k: usize) Error!void {
    return gemm(t, w, x, bias, y, m, k, 1);
}

/// dot(a, b) of two f32 vectors.
pub fn dotF32(a: []const f32, b: []const f32) Error!f32 {
    if (a.len != b.len) return error.BadShape;
    var y: [1]f32 = undefined;
    try gemm(.f32, std.mem.sliceAsBytes(a), b, null, &y, 1, a.len, 1);
    return y[0];
}

/// dot(a, b) of an f16 vector and an f32 vector.
pub fn dotF16(a: []const f16, b: []const f32) Error!f32 {
    if (a.len != b.len) return error.BadShape;
    var y: [1]f32 = undefined;
    try gemm(.f16, std.mem.sliceAsBytes(a), b, null, &y, 1, a.len, 1);
    return y[0];
}

/// dot(a, b) of a bf16 vector (raw 16-bit words) and an f32 vector.
pub fn dotBF16(a: []const u16, b: []const f32) Error!f32 {
    if (a.len != b.len) return error.BadShape;
    var y: [1]f32 = undefined;
    try gemm(.bf16, std.mem.sliceAsBytes(a), b, null, &y, 1, a.len, 1);
    return y[0];
}

/// C[m][n] = A[m][k] * B[k][n], all row-major f32.
pub fn gemmF32(a: []const f32, b: []const f32, c: []f32, m: usize, k: usize, n: usize) Error!void {
    if (a.len < m * k or b.len < k * n or c.len < m * n) return error.BadShape;
    if (m == 0 or n == 0) return;
    const allocator = std.heap.page_allocator;
    // The kernels take activations token major, so B is transposed to [n][k] and the [n][m] result transposed back.
    const bt = allocator.alloc(f32, n * k) catch return error.OutOfMemory;
    defer allocator.free(bt);
    const ct = allocator.alloc(f32, n * m) catch return error.OutOfMemory;
    defer allocator.free(ct);
    for (0..k) |ki| for (0..n) |ni| {
        bt[ni * k + ki] = b[ki * n + ni];
    };
    try gemm(.f32, std.mem.sliceAsBytes(a[0 .. m * k]), bt, null, ct, m, k, n);
    for (0..m) |mi| for (0..n) |ni| {
        c[mi * n + ni] = ct[ni * m + mi];
    };
}

// -- typed entry points, in the argument order the bindings and tests use -----------------------------------------------

pub fn gemvF32(w: []const f32, x: []const f32, bias: ?[]const f32, y: []f32, m: usize, k: usize) Error!void {
    return gemv(.f32, std.mem.sliceAsBytes(w), x, bias, y, m, k);
}
pub fn gemvF16(w: []const f16, x: []const f32, bias: ?[]const f32, y: []f32, m: usize, k: usize) Error!void {
    return gemv(.f16, std.mem.sliceAsBytes(w), x, bias, y, m, k);
}
pub fn gemvBF16(w: []const u16, x: []const f32, bias: ?[]const f32, y: []f32, m: usize, k: usize) Error!void {
    return gemv(.bf16, std.mem.sliceAsBytes(w), x, bias, y, m, k);
}
pub fn gemvQ8_0(w: []const u8, x: []const f32, bias: ?[]const f32, y: []f32, m: usize, k: usize) Error!void {
    return gemv(.q8_0, w, x, bias, y, m, k);
}
pub fn gemvQ4_0(w: []const u8, x: []const f32, bias: ?[]const f32, y: []f32, m: usize, k: usize) Error!void {
    return gemv(.q4_0, w, x, bias, y, m, k);
}
pub fn gemvQ4_K(w: []const u8, x: []const f32, bias: ?[]const f32, y: []f32, m: usize, k: usize) Error!void {
    return gemv(.q4_k, w, x, bias, y, m, k);
}
