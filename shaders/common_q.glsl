// Helpers for the integer matrix-vector kernels (mmvq.comp). `w32` is the weight buffer in the GPU layout described in
// common.glsl; this file only needs byte access and the Q4_K / Q5_K scale unpacking.

uint b8(uint o) { return (w32[o >> 2] >> ((o & 3u) * 8u)) & 255u; }
uint b16(uint o) { return (w32[o >> 2] >> ((o & 2u) * 8u)) & 65535u; }
float f32at(uint o) { return uintBitsToFloat(w32[o >> 2]); }
float h16(uint o) { return unpackHalf2x16(b16(o)).x; }
int s8(uint o) { return int(w32[o >> 2] << (24u - (o & 3u) * 8u)) >> 24; }

const uint M0F = 0x0f0f0f0fu;
// Per byte x - bias for bytes below 128 - bias, as signed bytes, without borrows between bytes.
uint signedBytes(uint x, uint bias) { return ((x | 0x80808080u) - bias * 0x01010101u) ^ 0x80808080u; }
// The low four bits of `bits`, one per byte.
uint spreadBits(uint bits) {
    return (bits & 1u) | (((bits >> 1) & 1u) << 8) | (((bits >> 2) & 1u) << 16) | (((bits >> 3) & 1u) << 24);
}

void sm4(uint s, uint j, out uint sc, out uint mn) {
    if (j < 4u) {
        sc = b8(s + j) & 63u;
        mn = b8(s + j + 4u) & 63u;
    } else {
        sc = (b8(s + j + 4u) & 15u) | ((b8(s + j - 4u) >> 6) << 4);
        mn = (b8(s + j + 4u) >> 4) | ((b8(s + j) >> 6) << 4);
    }
}

int dot4(uint a, uint b) { return dotPacked4x8EXT(int(a), int(b)); }

