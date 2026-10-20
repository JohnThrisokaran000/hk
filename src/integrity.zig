//! Container-wide integrity sealing for `.hk` files.
//!
//! The stored digest is SHA-256 over the complete container with its own
//! checksum fields cleared and the seal flag forced on. That makes the seal a
//! single 128-byte header write: the digest is computed over the bytes the file
//! will have, so no intermediate "flag set, digest zero" state can reach disk.
//! File-based entry points stream the container in fixed-size chunks, so peak
//! memory is one chunk regardless of model size (a mapping that fails must not
//! turn into an O(model) read).
const std = @import("std");
const format = @import("format.zig");

pub const Verification = enum { legacy_missing, valid, invalid };

pub const Error = error{ FileTooSmall, InvalidMagicOrVersion, ChecksumMismatch } ||
    std.mem.Allocator.Error || std.Io.File.OpenError || std.Io.File.ReadPositionalError ||
    std.Io.File.WritePositionalError || std.Io.File.StatError;

const header_len = @sizeOf(format.FileHeader);
const checksum_offset = @offsetOf(format.FileHeader, "checksum");
const checksum_len = @sizeOf(u64);
const digest_offset = @offsetOf(format.FileHeader, "reserved1");
pub const digest_len = 32;

/// Streaming read granularity. Large enough to keep SHA-256 throughput-bound, small enough to be noise in RSS.
pub const chunk_len: usize = 1 << 20;

comptime {
    std.debug.assert(header_len == 128);
    std.debug.assert(checksum_offset + checksum_len <= header_len);
    std.debug.assert(digest_offset + digest_len <= header_len);
}

/// The header exactly as it is hashed: seal flag on, checksum and digest slots zero.
fn canonicalHeader(header: format.FileHeader) [header_len]u8 {
    var h = header;
    h.flags |= format.HeaderFlags.HAS_SHA256_CHECKSUM;
    h.checksum = 0;
    @memset(h.reserved1[0..digest_len], 0);
    return std.mem.toBytes(h);
}

/// Computes the canonical container digest of an in-memory (usually mapped) container.
pub fn digest(bytes: []const u8) Error![digest_len]u8 {
    if (bytes.len < header_len) return error.FileTooSmall;
    const header = std.mem.bytesToValue(format.FileHeader, bytes[0..header_len]);
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(&canonicalHeader(header));
    hasher.update(bytes[header_len..]);
    var result: [digest_len]u8 = undefined;
    hasher.final(&result);
    return result;
}

/// Reads the header and the canonical digest of the file at `path` in `chunk_len` pieces.
fn digestFile(path: []const u8, allocator: std.mem.Allocator) Error!struct { header: format.FileHeader, digest: [digest_len]u8 } {
    const io = std.Options.debug_io;
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);

    const size = (try file.stat(io)).size;
    if (size < header_len) return error.FileTooSmall;
    var header_bytes: [header_len]u8 = undefined;
    if ((try file.readPositionalAll(io, &header_bytes, 0)) != header_len) return error.FileTooSmall;
    const header = std.mem.bytesToValue(format.FileHeader, &header_bytes);
    if (!header.isValid()) return error.InvalidMagicOrVersion;

    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(&canonicalHeader(header));

    const buf = try allocator.alloc(u8, chunk_len);
    defer allocator.free(buf);
    var offset: u64 = header_len;
    while (offset < size) {
        const want: usize = @intCast(@min(size - offset, buf.len));
        const got = try file.readPositionalAll(io, buf[0..want], offset);
        if (got != want) return error.FileTooSmall; // file shrank under us: never hash a partial read as the whole
        hasher.update(buf[0..want]);
        offset += want;
    }
    var result: [digest_len]u8 = undefined;
    hasher.final(&result);
    return .{ .header = header, .digest = result };
}

fn matches(header: format.FileHeader, computed: [digest_len]u8) bool {
    const compact = std.mem.readInt(u64, computed[0..checksum_len], .little);
    return header.checksum == compact and std.mem.eql(u8, header.reserved1[0..digest_len], &computed);
}

pub fn verify(bytes: []const u8) Error!Verification {
    if (bytes.len < header_len) return error.FileTooSmall;
    const header = std.mem.bytesToValue(format.FileHeader, bytes[0..header_len]);
    if (!header.isValid()) return error.InvalidMagicOrVersion;
    if ((header.flags & format.HeaderFlags.HAS_SHA256_CHECKSUM) == 0) return .legacy_missing;
    return if (matches(header, try digest(bytes))) .valid else .invalid;
}

/// Same verdict as `verify`, for a path, in bounded memory.
pub fn verifyFile(path: []const u8, allocator: std.mem.Allocator) Error!Verification {
    const d = try digestFile(path, allocator);
    if ((d.header.flags & format.HeaderFlags.HAS_SHA256_CHECKSUM) == 0) return .legacy_missing;
    return if (matches(d.header, d.digest)) .valid else .invalid;
}

/// Refuses to build on a container whose existing seal is wrong. Re-sealing after a mutation
/// would otherwise bless whatever corruption was already there. Legacy (unsealed) files pass:
/// they made no claim, and the next seal records their current contents.
pub fn requireIntactSeal(path: []const u8, allocator: std.mem.Allocator) Error!void {
    if (try verifyFile(path, allocator) == .invalid) return error.ChecksumMismatch;
}

/// Enables and refreshes the container seal after a mutation: one streamed pass to hash, then
/// one 128-byte header write that commits the flag, compact checksum and digest together.
pub fn sealFile(path: []const u8, allocator: std.mem.Allocator) Error!void {
    const d = try digestFile(path, allocator);
    var sealed = d.header;
    sealed.flags |= format.HeaderFlags.HAS_SHA256_CHECKSUM;
    sealed.checksum = std.mem.readInt(u64, d.digest[0..checksum_len], .little);
    @memcpy(sealed.reserved1[0..digest_len], &d.digest);

    const io = std.Options.debug_io;
    var file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_write });
    defer file.close(io);
    try file.writePositionalAll(io, std.mem.asBytes(&sealed), 0);
}
