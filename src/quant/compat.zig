//! Block level quantize/dequantize API for the C ABI, the importers and the public `hk.quantization` namespace.
//!
//! The names and signatures are the ones the repository's bindings were written against. Nothing is implemented
//! here: the encoders are `encode.zig`, the decoders are the engine's (`dequant.zig`, verified against llama.cpp),
//! and the research formats are `research.zig`. A decoder here takes a `count` of weights to produce, which may be
//! less than a block (the last block of a tensor whose length is not a multiple of the block size).

const std = @import("std");
const encode = @import("encode.zig");
const research = @import("research.zig");
const dequant = @import("dequant.zig");

pub const QK4_0 = encode.QK4_0;
pub const QK8_0 = encode.QK8_0;
pub const QK_K = encode.QK_K;

pub const BlockQ4_0 = encode.BlockQ4_0;
pub const BlockQ8_0 = encode.BlockQ8_0;
pub const BlockQ4_K = encode.BlockQ4_K;
pub const BlockQ5_K = encode.BlockQ5_K;
pub const BlockQ3_K = encode.BlockQ3_K;
pub const BlockQ6_K = encode.BlockQ6_K;
pub const BlockQ2_K = encode.BlockQ2_K;
pub const BlockQ8_K = encode.BlockQ8_K;

// -- encoders (GGML block layouts) -----------------------------------------------------------------------------------
pub const quantizeBlockQ4_0 = encode.quantizeBlockQ4_0;
pub const quantizeBlockQ8_0 = encode.quantizeBlockQ8_0;
pub const quantizeSuperBlockQ4_K = encode.quantizeSuperBlockQ4_K;
pub const quantizeSuperBlockQ5_K = encode.quantizeSuperBlockQ5_K;
pub const quantizeSuperBlockQ3_K = encode.quantizeSuperBlockQ3_K;
pub const quantizeSuperBlockQ6_K = encode.quantizeSuperBlockQ6_K;
pub const quantizeSuperBlockQ2_K = encode.quantizeSuperBlockQ2_K;
pub const quantizeSuperBlockQ8_K = encode.quantizeSuperBlockQ8_K;

// -- decoders: the engine's, with a partial block count ---------------------------------------------------------------

fn decodePartial(comptime elems: usize, comptime decodeFn: anytype, block: anytype, count: usize, out: []f32) void {
    var whole: [elems]f32 = undefined;
    decodeFn(std.mem.asBytes(block), &whole);
    const n = @min(count, elems);
    @memcpy(out[0..n], whole[0..n]);
}

pub fn dequantizeBlockQ4_0(block: *const BlockQ4_0, count: usize, out: []f32) void {
    decodePartial(32, dequant.q4_0, block, count, out);
}
pub fn dequantizeBlockQ8_0(block: *const BlockQ8_0, count: usize, out: []f32) void {
    decodePartial(32, dequant.q8_0, block, count, out);
}
pub fn dequantizeSuperBlockQ4_K(block: *const BlockQ4_K, count: usize, out: []f32) void {
    decodePartial(256, dequant.q4_k, block, count, out);
}
pub fn dequantizeSuperBlockQ5_K(block: *const BlockQ5_K, count: usize, out: []f32) void {
    decodePartial(256, dequant.q5_k, block, count, out);
}
pub fn dequantizeSuperBlockQ3_K(block: *const BlockQ3_K, count: usize, out: []f32) void {
    decodePartial(256, dequant.q3_k, block, count, out);
}
pub fn dequantizeSuperBlockQ6_K(block: *const BlockQ6_K, count: usize, out: []f32) void {
    decodePartial(256, dequant.q6_k, block, count, out);
}
pub fn dequantizeSuperBlockQ2_K(block: *const BlockQ2_K, count: usize, out: []f32) void {
    decodePartial(256, dequant.q2_k, block, count, out);
}
pub fn dequantizeSuperBlockQ8_K(block: *const BlockQ8_K, count: usize, out: []f32) void {
    decodePartial(256, dequant.q8_k, block, count, out);
}

// -- research formats (no GGML counterpart) --------------------------------------------------------------------------
pub const NF4_TABLE = research.NF4_TABLE;
pub const IQ4NL_TABLE = research.IQ4NL_TABLE;
pub const FP4_E2M1_TABLE = research.FP4_E2M1_TABLE;
pub const FP8_E4M3_TABLE = research.FP8_E4M3_TABLE;
pub const findClosestNF4 = research.findClosestNF4;
pub const findClosestIQ4NL = research.findClosestIQ4NL;
pub const quantizeBlockNF4 = research.quantizeBlockNF4;
pub const dequantizeBlockNF4 = research.dequantizeBlockNF4;
pub const quantizeBlockDQ8 = research.quantizeBlockDQ8;
pub const dequantizeBlockDQ8 = research.dequantizeBlockDQ8;
pub const quantizeBlockDQT = research.quantizeBlockDQT;
pub const dequantizeBlockDQT = research.dequantizeBlockDQT;
pub const quantizeBlockIQ4_NL = research.quantizeBlockIQ4_NL;
pub const dequantizeBlockIQ4_NL = research.dequantizeBlockIQ4_NL;
pub const dequantizeBlockMXFP4 = research.dequantizeBlockMXFP4;
pub const dequantizeBlockNVFP4 = research.dequantizeBlockNVFP4;
pub const dequantizeFP8_E4M3 = research.dequantizeFP8_E4M3;
