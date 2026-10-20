//! Crash recovery for in-place edits of a `.hk` file.
//!
//! A journal (`<file>.hkmeta-journal`) is written and synced *before* the first byte of the container
//! changes, and deleted only after the mutation and its re-seal finish. Two kinds:
//!
//! * `restore` (metadata patch, appendix append): snapshots the mutable prefix (the metadata region, or
//!   just the header for an append). Recovery truncates to the original size and puts the prefix back, so
//!   the file returns to its exact previous, sealed state. Tensor payloads are never copied.
//! * `roll_forward` (appendix rollback): the bytes being cut off cannot be saved cheaply, so the journal
//!   records the intended header and size instead. Recovery re-applies it and re-seals; the result is the
//!   state the caller asked for, never a guess.
//!
//! A leftover `restore` journal is rolled back on the next mutation: the caller never saw success, so the
//! last fully committed container wins. A torn journal (crash while writing it) proves the container was
//! not yet touched; it is discarded only if the container's own seal confirms that, otherwise recovery
//! fails loudly instead of guessing.
const std = @import("std");
const format = @import("format.zig");
const integrity = @import("integrity.zig");

const magic = "HKMJ";
const version: u16 = 2;
const header_len = @sizeOf(format.FileHeader);

pub const Kind = enum(u16) { restore = 0, roll_forward = 1 };

/// What a snapshot covers for a `restore` transaction.
pub const Scope = enum {
    /// Everything before the tensor payloads: header, metadata, tensor table.
    metadata_prefix,
    /// The 128-byte header only (appending to the end of the file).
    header_only,
};

pub const Recovery = enum { none, restored, completed, discarded_torn_journal };

pub const Error = error{ InvalidJournal, InvalidTransactionRange, UnexpectedEof } || integrity.Error ||
    std.Io.File.SetLengthError || std.Io.File.SyncError || std.Io.File.Writer.Error ||
    std.Io.Dir.DeleteFileError;

const JournalHeader = extern struct {
    magic_bytes: [4]u8 = magic.*,
    journal_version: u16 = version,
    kind: u16,
    /// File size before the mutation (`restore`) — unused for `roll_forward`.
    original_size: u64,
    /// Final file size after the mutation (`roll_forward`) — unused for `restore`.
    target_size: u64,
    /// Length of the body: the snapshot (`restore`) or the intended header (`roll_forward`).
    body_len: u64,
    body_crc32: u32,
    reserved: u32 = 0,
};

/// Rolls back or completes an interrupted mutation, if a journal exists.
pub fn recover(path: []const u8, allocator: std.mem.Allocator) Error!Recovery {
    const io = std.Options.debug_io;
    const cwd = std.Io.Dir.cwd();
    const journal_path = try journalPath(path, allocator);
    defer allocator.free(journal_path);

    const bytes = readWhole(journal_path, allocator) catch |err| switch (err) {
        error.FileNotFound => return .none,
        else => return err,
    };
    defer allocator.free(bytes);

    const parsed = parse(bytes) orelse return discardIfContainerIntact(path, journal_path, allocator);

    switch (@as(Kind, @enumFromInt(parsed.header.kind))) {
        .restore => {
            try setLength(path, parsed.header.original_size);
            try writeAt(path, parsed.body, 0);
            try cwd.deleteFile(io, journal_path);
            return .restored;
        },
        .roll_forward => {
            try writeAt(path, parsed.body, 0);
            try setLength(path, parsed.header.target_size);
            try integrity.sealFile(path, allocator);
            try cwd.deleteFile(io, journal_path);
            return .completed;
        },
    }
}

/// Snapshots the mutable prefix before an in-place edit. Pair with `commit` after the edit AND its re-seal.
pub fn begin(path: []const u8, allocator: std.mem.Allocator, scope: Scope) Error!void {
    _ = try recover(path, allocator);

    const io = std.Options.debug_io;
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const size = (try file.stat(io)).size;
    var hdr_bytes: [header_len]u8 = undefined;
    if (size < header_len or (try file.readPositionalAll(io, &hdr_bytes, 0)) != header_len) return error.FileTooSmall;
    const hdr = std.mem.bytesToValue(format.FileHeader, &hdr_bytes);
    if (!hdr.isValid()) return error.InvalidMagicOrVersion;

    const prefix_len: u64 = switch (scope) {
        .header_only => header_len,
        .metadata_prefix => hdr.tensor_data_offset,
    };
    if (prefix_len < header_len or prefix_len > size) return error.InvalidTransactionRange;

    const body = try allocator.alloc(u8, @intCast(prefix_len));
    defer allocator.free(body);
    if ((try file.readPositionalAll(io, body, 0)) != body.len) return error.FileTooSmall;

    try writeJournal(path, allocator, .{
        .kind = @intFromEnum(Kind.restore),
        .original_size = size,
        .target_size = 0,
        .body_len = body.len,
        .body_crc32 = std.hash.Crc32.hash(body),
    }, body);
}

/// Records an intended final header and size (appendix rollback) so a crash can be completed, not guessed at.
pub fn beginRollForward(path: []const u8, allocator: std.mem.Allocator, final_header: format.FileHeader, target_size: u64) Error!void {
    _ = try recover(path, allocator);
    const body = std.mem.asBytes(&final_header);
    try writeJournal(path, allocator, .{
        .kind = @intFromEnum(Kind.roll_forward),
        .original_size = 0,
        .target_size = target_size,
        .body_len = body.len,
        .body_crc32 = std.hash.Crc32.hash(body),
    }, body);
}

/// Deletes the journal. Call only once the mutation and the re-seal have both succeeded.
pub fn commit(path: []const u8, allocator: std.mem.Allocator) Error!void {
    const journal_path = try journalPath(path, allocator);
    defer allocator.free(journal_path);
    std.Io.Dir.cwd().deleteFile(std.Options.debug_io, journal_path) catch |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    };
}

const Parsed = struct { header: JournalHeader, body: []const u8 };

/// Null means the journal is torn or corrupt; the caller decides whether that is survivable.
fn parse(bytes: []const u8) ?Parsed {
    if (bytes.len < @sizeOf(JournalHeader)) return null;
    const header = std.mem.bytesToValue(JournalHeader, bytes[0..@sizeOf(JournalHeader)]);
    if (!std.mem.eql(u8, &header.magic_bytes, magic) or header.journal_version != version) return null;
    const kind = std.enums.fromInt(Kind, header.kind) orelse return null;
    const total = std.math.add(u64, @sizeOf(JournalHeader), header.body_len) catch return null;
    if (total != bytes.len) return null;
    const body = bytes[@sizeOf(JournalHeader)..];
    if (std.hash.Crc32.hash(body) != header.body_crc32) return null;
    switch (kind) {
        .restore => if (body.len < header_len or body.len > header.original_size) return null,
        .roll_forward => if (body.len != header_len) return null,
    }
    return .{ .header = header, .body = body };
}

fn discardIfContainerIntact(path: []const u8, journal_path: []const u8, allocator: std.mem.Allocator) Error!Recovery {
    // The journal is synced before any container byte changes, so a torn one means no change happened.
    // Trust that only when the container's seal agrees; an unsealed or mismatching file stays flagged.
    if (try integrity.verifyFile(path, allocator) != .valid) return error.InvalidJournal;
    try std.Io.Dir.cwd().deleteFile(std.Options.debug_io, journal_path);
    return .discarded_torn_journal;
}

fn writeJournal(path: []const u8, allocator: std.mem.Allocator, header: JournalHeader, body: []const u8) Error!void {
    const journal_path = try journalPath(path, allocator);
    defer allocator.free(journal_path);
    const io = std.Options.debug_io;
    var journal = try std.Io.Dir.cwd().createFile(io, journal_path, .{});
    defer journal.close(io);
    try journal.writeStreamingAll(io, std.mem.asBytes(&header));
    try journal.writeStreamingAll(io, body);
    try journal.sync(io);
}

fn readWhole(path: []const u8, allocator: std.mem.Allocator) ![]u8 {
    const io = std.Options.debug_io;
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const size: usize = @intCast((try file.stat(io)).size);
    const buf = try allocator.alloc(u8, size);
    errdefer allocator.free(buf);
    if ((try file.readPositionalAll(io, buf, 0)) != size) return error.UnexpectedEof;
    return buf;
}

fn writeAt(path: []const u8, data: []const u8, offset: u64) !void {
    const io = std.Options.debug_io;
    var file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_write });
    defer file.close(io);
    try file.writePositionalAll(io, data, offset);
}

fn setLength(path: []const u8, len: u64) !void {
    const io = std.Options.debug_io;
    var file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_write });
    defer file.close(io);
    try file.setLength(io, len);
}

fn journalPath(path: []const u8, allocator: std.mem.Allocator) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}.hkmeta-journal", .{path});
}
