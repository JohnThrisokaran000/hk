//! Operator capability table (plan §136, §137): which weight formats each execution target can run, and how.
//!
//! One place answers "can this target run this model?", so selection, messages and tests share a source of truth
//! instead of each backend growing its own ad-hoc list. A format is `.full` when the target has both the
//! matrix-vector kernel (decode) and the batched kernel (prefill), `.decode` when only the former exists (prompt
//! processing would fall back to repeated matrix-vector products), and `.none` otherwise.
//!
//! Entries are claims about code that exists today. A target with no engine yet (CUDA, until stage 3 of 2.0 lands)
//! answers `.none` for everything rather than advertising kernels nothing calls.

const std = @import("std");
const format = @import("../format.zig");
const quant = @import("../quant.zig");
const vk_engine = @import("../vk/engine.zig");
const weights_mod = @import("../engine/weights.zig");

pub const StorageType = format.StorageType;

pub const Target = enum { cpu, vulkan, cuda };

pub const Level = enum { none, decode, full };

/// How `target` runs a weight matrix stored as `t`.
pub fn weightFormat(target: Target, t: StorageType) Level {
    return switch (target) {
        .cpu => if (quant.vecdot.supported(t)) .full else .none,
        .vulkan => if (vk_engine.supports(t)) .full else .none,
        .cuda => .none,
    };
}

pub fn apiName(target: Target) []const u8 {
    return switch (target) {
        .cpu => "CPU",
        .vulkan => "Vulkan",
        .cuda => "CUDA",
    };
}

/// The first weight matrix of `w` that `target` cannot run, if any. Embedding and output matrices count: a device
/// that cannot read the token embedding cannot run the model either.
pub fn firstUnsupported(target: Target, w: *const weights_mod.Weights) ?StorageType {
    if (weightFormat(target, w.tok_embd.t) == .none) return w.tok_embd.t;
    if (weightFormat(target, w.output.t) == .none) return w.output.t;
    for (w.layers) |l| {
        for ([_]weights_mod.Mat{ l.wq, l.wk, l.wv, l.wo, l.w_gate, l.w_up, l.w_down }) |m| {
            if (weightFormat(target, m.t) == .none) return m.t;
        }
    }
    return null;
}

test "a target with no engine claims no formats" {
    inline for (comptime std.enums.values(StorageType)) |t| {
        try std.testing.expectEqual(Level.none, weightFormat(.cuda, t));
    }
}

test "every format Vulkan runs the CPU runs too" {
    // A device must never accept a model the CPU fallback cannot run, or `auto` could not fall back.
    inline for (comptime std.enums.values(StorageType)) |t| {
        if (weightFormat(.vulkan, t) != .none) try std.testing.expect(weightFormat(.cpu, t) != .none);
    }
}
