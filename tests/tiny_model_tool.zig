//! Writes the tiny test model used by the end to end tests, for tests written in other languages.
//!
//!   hk-tiny-model out.gguf [--chat] [--q8] [--arch llama|qwen2|qwen3]
//!                          [--dim N] [--layers N] [--heads N] [--kv N] [--ffn N] [--ctx N] [--seed N]
//!
//! The size flags make larger synthetic models for comparing backends with each other. The weights are random, so
//! they say nothing about model quality and the numbers say nothing about other engines.

const std = @import("std");
const tiny = @import("support/tiny_model.zig");

pub fn main(init: std.process.Init) !void {
    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer it.deinit();
    _ = it.next();
    const out = it.next() orelse {
        std.debug.print("usage: hk-tiny-model out.gguf [--chat] [--q8]\n", .{});
        return error.BadUsage;
    };
    var o = tiny.Options{};
    var chat = false;
    while (it.next()) |a| {
        if (std.mem.eql(u8, a, "--chat")) {
            chat = true;
        } else if (std.mem.eql(u8, a, "--q8")) {
            o.q8_0 = true;
        } else if (std.mem.eql(u8, a, "--arch")) {
            const v = it.next() orelse return badUsage();
            o.arch = std.meta.stringToEnum(tiny.Arch, v) orelse return badUsage();
        } else inline for (.{ .{ "--dim", "dim" }, .{ "--layers", "n_layers" }, .{ "--heads", "n_heads" }, .{ "--kv", "n_kv" }, .{ "--ffn", "ffn" }, .{ "--ctx", "ctx" }, .{ "--seed", "seed" } }) |f| {
            if (std.mem.eql(u8, a, f[0])) {
                const v = it.next() orelse return badUsage();
                @field(o, f[1]) = std.fmt.parseInt(@TypeOf(@field(o, f[1])), v, 10) catch return badUsage();
                break;
            }
        } else {
            std.debug.print("unknown option '{s}'\n", .{a});
            return badUsage();
        }
    }
    if (chat) o.chat_template = tiny.chatml;
    try tiny.makeTinyGguf(init.gpa, init.io, out, o);
}

fn badUsage() error{BadUsage} {
    std.debug.print("usage: hk-tiny-model out.gguf [--chat] [--q8] [--arch A] [--dim N] [--layers N] [--heads N] [--kv N] [--ffn N] [--ctx N] [--seed N]\n", .{});
    return error.BadUsage;
}
