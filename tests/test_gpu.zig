//! A device engine must compute what the CPU engine computes. Each check runs for Vulkan and for CUDA and is skipped
//! (per backend) when the machine has no usable device of that kind.

const std = @import("std");
const hk = @import("hk");
const tiny = @import("support/tiny_model.zig");

const gpa = std.testing.allocator;
const io = std.testing.io;
const Choice = hk.backend.select.Choice;

fn cosine(a: []const f32, b: []const f32) f64 {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (a, b) |x, y| {
        dot += @as(f64, x) * y;
        na += @as(f64, x) * x;
        nb += @as(f64, y) * y;
    }
    return dot / (@sqrt(na) * @sqrt(nb));
}

fn load(path: []const u8, gpu: hk.engine.model.GpuMode, backend: Choice, slots: usize, reader: *hk.HKReader) !hk.engine.Model {
    var diag = hk.engine.Diag{};
    return hk.engine.Model.init(gpa, reader, .{ .n_ctx = 256, .n_threads = 2, .gpu = gpu, .backend = backend, .gpu_slots = slots }, &diag) catch |e| {
        if (gpu == .off) std.debug.print("load {s}: {s}: {s}\n", .{ path, @errorName(e), diag.message() });
        return e;
    };
}

fn makeModel(dir: []const u8, name: []const u8, o: tiny.Options) ![]u8 {
    const g = try std.fmt.allocPrint(gpa, "{s}/{s}.gguf", .{ dir, name });
    defer gpa.free(g);
    const h = try std.fmt.allocPrint(gpa, "{s}/{s}.hk", .{ dir, name });
    errdefer gpa.free(h);
    try tiny.makeTinyGguf(gpa, io, g, o);
    var cdiag = hk.convert.gguf.Diag{};
    try hk.convert.gguf.convertFile(gpa, io, g, h, .{ .diag = &cdiag });
    return h;
}

/// `min_cos` is how close the two engines must be: the CPU quantizes activations to 8 bits for
/// quantized weights and the GPU does not, so those agree to about three digits, not five.
fn check(dir: []const u8, name: []const u8, o: tiny.Options, min_cos: f64, backend: Choice) !void {
    const h = try makeModel(dir, name, o);
    defer gpa.free(h);
    var reader = try hk.HKReader.open(h, gpa);
    defer reader.deinit();
    var gpu_model = load(h, .on, backend, 1, &reader) catch |e| switch (e) {
        error.GpuFailed => return error.SkipZigTest,
        else => return e,
    };
    defer gpu_model.deinit();
    // The device must really be the one asked for, not a quiet CPU run.
    try std.testing.expect(gpu_model.gpu != null);
    try std.testing.expectEqual(expectedKind(backend), gpu_model.gpu.?.kind());
    var cpu_model = try load(h, .off, .cpu, 1, &reader);
    defer cpu_model.deinit();

    // A prompt batch (the batched kernels), then single steps (the decode kernels).
    const prompt = [_]u32{ 3, 17, 5, 5, 29, 0, 11, 8, 40, 41, 9 };
    try gpu_model.forward(&prompt, 0);
    try cpu_model.forward(&prompt, 0);
    const lg = try gpa.dupe(f32, gpu_model.logitsFor(prompt.len - 1));
    defer gpa.free(lg);
    const lc = cpu_model.logitsFor(prompt.len - 1);
    try std.testing.expect(cosine(lg, lc) > min_cos);

    var pos: usize = prompt.len;
    var tok: u32 = 7;
    for (0..6) |_| {
        try gpu_model.forward(&.{tok}, pos);
        try cpu_model.forward(&.{tok}, pos);
        const a = try gpa.dupe(f32, gpu_model.logitsFor(0));
        defer gpa.free(a);
        const b = cpu_model.logitsFor(0);
        try std.testing.expect(cosine(a, b) > min_cos);
        var best: usize = 0;
        for (b, 0..) |x, i| if (x > b[best]) {
            best = i;
        };
        tok = @intCast(best);
        pos += 1;
    }
}

fn expectedKind(backend: Choice) hk.backend.Kind {
    return switch (backend) {
        .vulkan => .vulkan,
        .cuda => .cuda,
        else => unreachable,
    };
}

fn checkAllVariants(backend: Choice) !void {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);
    // Plain float weights, and Q8_0 weights, with grouped query attention.
    try check(dir, "f32", .{}, 0.9999, backend);
    try check(dir, "q8", .{ .q8_0 = true }, 0.99, backend);
    // Plain multi head attention and a wider model.
    try check(dir, "mha", .{ .n_heads = 4, .n_kv = 4, .dim = 128, .ffn = 256 }, 0.9999, backend);
    // Qwen2: q/k/v biases and NeoX rotary embedding. Qwen3: per head q/k norms and NeoX rotary embedding.
    try check(dir, "qwen2", .{ .arch = .qwen2 }, 0.9999, backend);
    try check(dir, "qwen2q8", .{ .arch = .qwen2, .q8_0 = true }, 0.99, backend);
    try check(dir, "qwen3", .{ .arch = .qwen3 }, 0.9999, backend);
    try check(dir, "qwen3q8", .{ .arch = .qwen3, .q8_0 = true }, 0.99, backend);
}

test "the Vulkan engine matches the CPU engine" {
    try checkAllVariants(.vulkan);
}

test "the CUDA engine matches the CPU engine" {
    try checkAllVariants(.cuda);
}

/// Two sequences share one pass, positions advance independently, a third sequence has no slot until one is released.
fn checkSlots(backend: Choice) !void {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);
    const h = try makeModel(dir, "slots", .{ .arch = .qwen3 });
    defer gpa.free(h);
    var reader = try hk.HKReader.open(h, gpa);
    defer reader.deinit();
    var dev = load(h, .on, backend, 2, &reader) catch |e| switch (e) {
        error.GpuFailed => return error.SkipZigTest,
        else => return e,
    };
    defer dev.deinit();
    var cpu = try load(h, .off, .cpu, 1, &reader);
    defer cpu.deinit();

    const Kv = hk.engine.KvCache;
    var a = try Kv.init(gpa, dev.cfg, dev.n_ctx);
    defer a.deinit();
    var b = try Kv.init(gpa, dev.cfg, dev.n_ctx);
    defer b.deinit();
    var c = try Kv.init(gpa, dev.cfg, dev.n_ctx);
    defer c.deinit();
    var ca = try Kv.init(gpa, cpu.cfg, cpu.n_ctx);
    defer ca.deinit();
    var cb = try Kv.init(gpa, cpu.cfg, cpu.n_ctx);
    defer cb.deinit();

    const Item = hk.engine.Item;
    // Pass 1: A and B each get a 3 token prompt in one batch. Pass 2: one new token each, at different positions.
    const p1 = [_]Item{
        .{ .kv = &a, .token = 3, .pos = 0 }, .{ .kv = &a, .token = 17, .pos = 1 }, .{ .kv = &a, .token = 5, .pos = 2 },
        .{ .kv = &b, .token = 29, .pos = 0 }, .{ .kv = &b, .token = 11, .pos = 1 },
    };
    const c1 = [_]Item{
        .{ .kv = &ca, .token = 3, .pos = 0 }, .{ .kv = &ca, .token = 17, .pos = 1 }, .{ .kv = &ca, .token = 5, .pos = 2 },
        .{ .kv = &cb, .token = 29, .pos = 0 }, .{ .kv = &cb, .token = 11, .pos = 1 },
    };
    try dev.forwardItems(&p1);
    try cpu.forwardItems(&c1);
    const want_idx = [_]usize{ 2, 4 };
    const got1 = try gpa.dupe(f32, try dev.logitsMany(&want_idx));
    defer gpa.free(got1);
    const ref1 = try gpa.dupe(f32, try cpu.logitsMany(&want_idx));
    defer gpa.free(ref1);
    const v = dev.cfg.vocab;
    try std.testing.expect(cosine(got1[0..v], ref1[0..v]) > 0.9999);
    try std.testing.expect(cosine(got1[v .. 2 * v], ref1[v .. 2 * v]) > 0.9999);

    const p2 = [_]Item{ .{ .kv = &a, .token = 8, .pos = 3 }, .{ .kv = &b, .token = 40, .pos = 2 } };
    const c2 = [_]Item{ .{ .kv = &ca, .token = 8, .pos = 3 }, .{ .kv = &cb, .token = 40, .pos = 2 } };
    try dev.forwardItems(&p2);
    try cpu.forwardItems(&c2);
    const idx2 = [_]usize{ 0, 1 };
    const got2 = try gpa.dupe(f32, try dev.logitsMany(&idx2));
    defer gpa.free(got2);
    const ref2 = try gpa.dupe(f32, try cpu.logitsMany(&idx2));
    defer gpa.free(ref2);
    try std.testing.expect(cosine(got2[0..v], ref2[0..v]) > 0.9999);
    try std.testing.expect(cosine(got2[v .. 2 * v], ref2[v .. 2 * v]) > 0.9999);

    // Both slots are taken: a third sequence is refused, not silently run against someone else's cache.
    const p3 = [_]Item{.{ .kv = &c, .token = 3, .pos = 0 }};
    try std.testing.expectError(error.GpuFailed, dev.forwardItems(&p3));
    // Releasing A frees its slot for C.
    dev.forget(&a);
    try dev.forwardItems(&p3);
}

test "the Vulkan engine keeps sequences apart and runs out of slots loudly" {
    try checkSlots(.vulkan);
}

test "the CUDA engine keeps sequences apart and runs out of slots loudly" {
    try checkSlots(.cuda);
}
