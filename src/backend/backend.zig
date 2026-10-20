//! The seam between the model and the device that runs the network.
//!
//! A backend executes the *whole* network for a batch of tokens (embedding to final hidden state) and reads
//! logits back, which is the granularity the Vulkan engine already had and the one CUDA will use. A per-operator
//! graph would force every backend through the same op boundaries and re-introduce host round trips between them;
//! whole-network executors keep each device free to fuse and keep activations resident.
//!
//! The set of backends is closed and known at build time, so this is a tagged union with `inline else` dispatch
//! rather than a vtable: no indirect call, and a backend that is not built for the target is simply absent.
//!
//! Weights are views into the mapped container on the CPU; a device backend uploads them once (a deliberate copy
//! to device memory, reported by `bytesNeeded`). The host `KvCache` is only an identity key for a device
//! backend: the device owns the real cache for the sequence while it holds a slot.

const std = @import("std");
const kv_mod = @import("../engine/kv.zig");
const vk_engine = @import("../vk/engine.zig");

pub const caps = @import("caps.zig");
pub const select = @import("select.zig");

pub const Kind = enum { vulkan };

pub const Error = vk_engine.Error;

pub const Item = kv_mod.Item;

pub const Device = union(Kind) {
    vulkan: *vk_engine.Engine,

    pub fn kind(self: Device) Kind {
        return std.meta.activeTag(self);
    }

    /// Name of the compute API, for messages ("running on X (Vulkan)").
    pub fn apiName(self: Device) []const u8 {
        return switch (self) {
            .vulkan => "Vulkan",
        };
    }

    pub fn deviceName(self: Device) []const u8 {
        return switch (self) {
            inline else => |e| e.deviceName(),
        };
    }

    /// Most logit rows `logits` can return at once.
    pub fn maxRows(self: Device) usize {
        return switch (self) {
            inline else => |e| e.max_rows,
        };
    }

    /// Runs one forward pass over `items` (tokens of one or more sequences).
    pub fn forward(self: Device, items: []const Item) Error!void {
        return switch (self) {
            inline else => |e| e.forward(items),
        };
    }

    /// Logits of the given items of the last pass, one vocabulary row each, valid until the next call.
    pub fn logits(self: Device, indices: []const usize) Error![]const f32 {
        return switch (self) {
            inline else => |e| e.logits(indices),
        };
    }

    /// Gives back whatever device state a sequence's cache held.
    pub fn release(self: Device, kv: *const kv_mod.KvCache) void {
        switch (self) {
            inline else => |e| e.release(kv),
        }
    }

    /// Frees the device resources and the engine object itself (which `allocator` created).
    pub fn destroy(self: Device, allocator: std.mem.Allocator) void {
        switch (self) {
            inline else => |e| {
                e.deinit();
                allocator.destroy(e);
            },
        }
    }
};
