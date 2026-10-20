//! Writes the tiny test model used by the end to end tests, for tests written in other languages.
//!
//!   hk-tiny-model out.gguf [--chat] [--q8]

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
    var chat = false;
    var q8_0 = false;
    while (it.next()) |a| {
        if (std.mem.eql(u8, a, "--chat")) chat = true else if (std.mem.eql(u8, a, "--q8")) q8_0 = true else {
            std.debug.print("unknown option '{s}'\n", .{a});
            return error.BadUsage;
        }
    }
    try tiny.makeTinyGguf(init.gpa, init.io, out, .{ .chat_template = if (chat) tiny.chatml else null, .q8_0 = q8_0 });
}
