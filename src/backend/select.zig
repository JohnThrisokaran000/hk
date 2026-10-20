//! Device selection and memory planning (plan §1.2, §134, §137).
//!
//! `plan` is a pure function of what was probed, so every branch (no driver, unsupported format, does not fit,
//! context shrunk to fit) is tested without any hardware. Probing the real devices lives next to each backend.
//!
//! Policy, in one place:
//!   * candidates are tried in the order given (best first); `Choice` can restrict the search to one target;
//!   * a device is rejected, with a recorded reason, when it is unusable, cannot run a weight format of the model, or
//!     cannot hold the model plus a context of at least `min_ctx` positions;
//!   * the context is halved until the model fits in 90 % of free device memory (headroom for the driver and
//!     fragmentation), and the caller learns the context it actually got;
//!   * nothing is chosen silently: when no device is chosen the plan says why, per device.

const std = @import("std");
const caps = @import("caps.zig");

pub const Target = caps.Target;
pub const StorageType = caps.StorageType;

/// What the user asked for.
pub const Choice = enum {
    auto,
    cpu,
    vulkan,
    cuda,

    pub fn wants(self: Choice, target: Target) bool {
        return switch (self) {
            .auto => true,
            .cpu => target == .cpu,
            .vulkan => target == .vulkan,
            .cuda => target == .cuda,
        };
    }
};

/// What a user's choice means for loading a model.
pub const Wants = struct {
    /// Whether a device should be tried at all.
    device: bool,
    /// Whether failing to get one is an error (an explicit request) or a quiet fall back to the CPU (`auto`).
    required: bool,
};

pub fn wants(choice: Choice) Wants {
    return switch (choice) {
        .cpu => .{ .device = false, .required = false },
        .auto => .{ .device = true, .required = false },
        .vulkan, .cuda => .{ .device = true, .required = true },
    };
}

/// Memory a target needs: a fixed part (weights, activations, runtime) plus a part that grows with the context
/// (the KV cache of every slot).
pub const Need = struct {
    base: u64,
    per_ctx: u64,

    pub fn at(self: Need, n_ctx: usize) u64 {
        return self.base +| self.per_ctx *| n_ctx;
    }
};

/// What probing one device found.
pub const Candidate = struct {
    target: Target,
    /// Null when the device can be used; otherwise a short reason (shown to the user as is).
    unusable: ?[]const u8 = null,
    device_name: []const u8 = "",
    free_bytes: u64 = 0,
    /// A weight format of the model this target cannot run, if any (from `caps.firstUnsupported`).
    unsupported: ?StorageType = null,
    need: Need,
};

pub const Why = union(enum) {
    unavailable: []const u8,
    unsupported_format: StorageType,
    does_not_fit: struct { needed: u64, free: u64 },
};

pub const Rejection = struct { target: Target, why: Why };

pub const max_rejections = 4;

pub const Plan = struct {
    /// The chosen target; `.cpu` when no device qualified.
    target: Target = .cpu,
    /// The context the chosen target will reserve (equals the request unless it had to shrink).
    n_ctx: usize,
    device_name: []const u8 = "",
    rejections: [max_rejections]Rejection = undefined,
    n_rejections: usize = 0,

    pub fn rejected(self: *const Plan) []const Rejection {
        return self.rejections[0..self.n_rejections];
    }

    fn reject(self: *Plan, target: Target, why: Why) void {
        if (self.n_rejections == max_rejections) return;
        self.rejections[self.n_rejections] = .{ .target = target, .why = why };
        self.n_rejections += 1;
    }
};

pub const Inputs = struct {
    choice: Choice = .auto,
    /// Context the caller would like to reserve.
    n_ctx: usize,
    /// Smallest context worth running a device with; below it the device is rejected.
    min_ctx: usize = 512,
    /// Device candidates, best first. The CPU is the implicit last resort and is not listed.
    candidates: []const Candidate,
};

/// Fraction of free device memory the plan may use, as numerator over 10.
const headroom_tenths: u64 = 9;

pub fn plan(in: Inputs) Plan {
    var p = Plan{ .n_ctx = in.n_ctx };
    var asked_target_present = false;
    for (in.candidates) |c| {
        if (c.target == .cpu or !in.choice.wants(c.target)) continue;
        asked_target_present = true;
        if (c.unusable) |why| {
            p.reject(c.target, .{ .unavailable = why });
            continue;
        }
        if (c.unsupported) |t| {
            p.reject(c.target, .{ .unsupported_format = t });
            continue;
        }
        const budget = c.free_bytes / 10 * headroom_tenths;
        var ctx = in.n_ctx;
        while (c.need.at(ctx) > budget and ctx > in.min_ctx) ctx = @max(in.min_ctx, ctx / 2);
        if (c.need.at(ctx) > budget) {
            p.reject(c.target, .{ .does_not_fit = .{ .needed = c.need.at(ctx), .free = c.free_bytes } });
            continue;
        }
        p.target = c.target;
        p.n_ctx = ctx;
        p.device_name = c.device_name;
        return p;
    }
    // An explicit request for a device this build has no backend for must say so rather than fall to the CPU quietly.
    if (!asked_target_present) switch (in.choice) {
        .vulkan => p.reject(.vulkan, .{ .unavailable = "this build has no Vulkan backend" }),
        .cuda => p.reject(.cuda, .{ .unavailable = "this build has no CUDA backend" }),
        .auto, .cpu => {},
    };
    return p;
}

/// The reason alone, e.g. "a weight format (q5_k) has no CUDA kernel yet".
pub fn writeReason(r: Rejection, w: *std.Io.Writer) std.Io.Writer.Error!void {
    const api = caps.apiName(r.target);
    switch (r.why) {
        .unavailable => |msg| try w.writeAll(msg),
        .unsupported_format => |t| try w.print("a weight format ({s}) has no {s} kernel yet", .{ @tagName(t), api }),
        .does_not_fit => |f| try w.print("the model needs {d} MiB and only {d} MiB are free", .{ f.needed >> 20, f.free >> 20 }),
    }
}

/// Writes one human-readable line per rejection, e.g. "CUDA: a weight format (q5_k) has no CUDA kernel yet".
pub fn describe(p: *const Plan, w: *std.Io.Writer) std.Io.Writer.Error!void {
    for (p.rejected()) |r| {
        try w.print("{s}: ", .{caps.apiName(r.target)});
        try writeReason(r, w);
        try w.writeByte('\n');
    }
}

const mib: u64 = 1 << 20;

fn cand(target: Target, free_mib: u64, base_mib: u64, per_ctx_kib: u64) Candidate {
    return .{
        .target = target,
        .device_name = "fake",
        .free_bytes = free_mib * mib,
        .need = .{ .base = base_mib * mib, .per_ctx = per_ctx_kib * 1024 },
    };
}

test "the first device that fits is chosen, best first" {
    const cs = [_]Candidate{ cand(.cuda, 6000, 1000, 100), cand(.vulkan, 6000, 1000, 100) };
    const p = plan(.{ .n_ctx = 4096, .candidates = &cs });
    try std.testing.expectEqual(Target.cuda, p.target);
    try std.testing.expectEqual(@as(usize, 4096), p.n_ctx);
    try std.testing.expectEqual(@as(usize, 0), p.n_rejections);
}

test "an unusable device is skipped with its reason and the next one is tried" {
    var cs = [_]Candidate{ cand(.cuda, 6000, 1000, 100), cand(.vulkan, 6000, 1000, 100) };
    cs[0].unusable = "no CUDA driver found";
    const p = plan(.{ .n_ctx = 4096, .candidates = &cs });
    try std.testing.expectEqual(Target.vulkan, p.target);
    try std.testing.expectEqual(@as(usize, 1), p.n_rejections);
    try std.testing.expectEqualStrings("no CUDA driver found", p.rejected()[0].why.unavailable);
}

test "a model with a format the device cannot run is rejected naming the format" {
    var cs = [_]Candidate{cand(.vulkan, 6000, 1000, 100)};
    cs[0].unsupported = .q5_k;
    const p = plan(.{ .n_ctx = 4096, .candidates = &cs });
    try std.testing.expectEqual(Target.cpu, p.target);
    try std.testing.expectEqual(StorageType.q5_k, p.rejected()[0].why.unsupported_format);
}

test "the context shrinks until the model fits and the plan reports the context it got" {
    // base 1000 MiB + 1 MiB per context position; budget = 90 % of 2000 MiB = 1800 MiB -> 800 positions at most.
    const cs = [_]Candidate{cand(.vulkan, 2000, 1000, 1024)};
    const p = plan(.{ .n_ctx = 4096, .candidates = &cs });
    try std.testing.expectEqual(Target.vulkan, p.target);
    try std.testing.expectEqual(@as(usize, 512), p.n_ctx);
}

test "a model that cannot fit even at the minimum context is rejected with the numbers" {
    const cs = [_]Candidate{cand(.vulkan, 1000, 2000, 1)};
    const p = plan(.{ .n_ctx = 4096, .candidates = &cs });
    try std.testing.expectEqual(Target.cpu, p.target);
    try std.testing.expect(p.rejected()[0].why.does_not_fit.needed > p.rejected()[0].why.does_not_fit.free);
}

test "an explicit choice restricts the search" {
    const cs = [_]Candidate{ cand(.cuda, 6000, 1000, 100), cand(.vulkan, 6000, 1000, 100) };
    const p = plan(.{ .choice = .vulkan, .n_ctx = 4096, .candidates = &cs });
    try std.testing.expectEqual(Target.vulkan, p.target);
    const none = plan(.{ .choice = .cpu, .n_ctx = 4096, .candidates = &cs });
    try std.testing.expectEqual(Target.cpu, none.target);
    try std.testing.expectEqual(@as(usize, 0), none.n_rejections);
}

test "naming a device means use it or fail; auto may fall back; cpu never tries a device" {
    try std.testing.expect(wants(.vulkan).required and wants(.vulkan).device);
    try std.testing.expect(wants(.cuda).required and wants(.cuda).device);
    try std.testing.expect(wants(.auto).device and !wants(.auto).required);
    try std.testing.expect(!wants(.cpu).device and !wants(.cpu).required);
}

test "asking for a device the build has no backend for is a recorded rejection" {
    const cs = [_]Candidate{cand(.vulkan, 6000, 1000, 100)};
    const p = plan(.{ .choice = .cuda, .n_ctx = 4096, .candidates = &cs });
    try std.testing.expectEqual(Target.cpu, p.target);
    try std.testing.expectEqual(@as(usize, 1), p.n_rejections);
    try std.testing.expectEqual(Target.cuda, p.rejected()[0].target);
}

test "the exact fit boundary uses 90 percent of free memory" {
    // need = 900 MiB exactly equals the budget of 1000 MiB free -> fits; one more MiB of base does not.
    const fit = [_]Candidate{cand(.vulkan, 1000, 900, 0)};
    try std.testing.expectEqual(Target.vulkan, plan(.{ .n_ctx = 512, .candidates = &fit }).target);
    const over = [_]Candidate{cand(.vulkan, 1000, 901, 0)};
    try std.testing.expectEqual(Target.cpu, plan(.{ .n_ctx = 512, .candidates = &over }).target);
}

test "describe prints one line per rejection" {
    var cs = [_]Candidate{ cand(.cuda, 6000, 1000, 100), cand(.vulkan, 6000, 1000, 100) };
    cs[0].unusable = "no CUDA driver found";
    cs[1].unsupported = .iq2_xs;
    const p = plan(.{ .n_ctx = 4096, .candidates = &cs });
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try describe(&p, &w);
    try std.testing.expectEqualStrings(
        "CUDA: no CUDA driver found\nVulkan: a weight format (iq2_xs) has no Vulkan kernel yet\n",
        w.buffered(),
    );
}
