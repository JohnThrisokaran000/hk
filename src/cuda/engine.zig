//! The CUDA engine: the whole network on one NVIDIA device, driven through the CUDA driver API (loaded at run time)
//! with kernels written in Zig (`src/cuda_device.zig`).
//!
//! Scope of this first version, stated plainly:
//!   * correctness first (plan §152). Every item of a pass runs the full network on its own, one kernel launch after
//!     another, so a prompt of N tokens costs N sequential decode steps. There is no batched prefill GEMM, no fusion,
//!     no CUDA graphs and no Tensor Cores yet; those are later stages of HK 2.0/2.1;
//!   * weights are copied to the device once (a deliberate copy, counted by `bytesNeeded`) straight from the mapped
//!     container, with no staging copy in host memory;
//!   * the token embedding is decoded on the host (one row per token) and uploaded, so the embedding matrix itself
//!     only goes to the device when the model ties it to the output projection;
//!   * the KV cache lives on the device in f16, as on the CPU, per slot; the host `KvCache` is only an identity key.

const std = @import("std");
const cuda = @import("../cuda.zig");
const format = @import("../format.zig");
const quant = @import("../quant.zig");
const config_mod = @import("../engine/config.zig");
const weights_mod = @import("../engine/weights.zig");
const kv_mod = @import("../engine/kv.zig");
const ops = @import("../engine/ops.zig");

const Config = config_mod.Config;
const Mat = weights_mod.Mat;
const StorageType = format.StorageType;

pub const Error = error{
    UnsupportedType,
    UnsupportedShape,
    NoGpuSlot,
    ContextFull,
    OutOfMemory,
    OutOfDeviceMemory,
    DeviceFailed,
};

pub const Options = struct {
    /// Most tokens in one forward pass.
    n_batch: usize = 256,
    /// Context length reserved per sequence.
    n_ctx: usize = 2048,
    /// Sequences that can use the device at once.
    slots: usize = 1,
};

/// Threads per block of the reduction kernels (one block per row/head).
const block_threads: u32 = 128;
/// Threads per block of the elementwise kernels.
const elem_threads: u32 = 256;
/// Largest head width the attention kernel supports.
const max_head_dim = 256;
/// Logit rows that can be read back at once.
const max_logit_rows = 64;

/// Whether the device can multiply weights stored as `t`: every format the CPU decodes has a device kernel.
pub fn supports(t: StorageType) bool {
    return quant.vecdot.supported(t);
}

const DevMat = struct {
    mem: cuda.DeviceMem,
    t: StorageType,
    rows: usize,
    cols: usize,
};

const DevLayer = struct {
    attn_norm: cuda.DeviceMem,
    wq: DevMat,
    wk: DevMat,
    wv: DevMat,
    wo: DevMat,
    bq: ?cuda.DeviceMem,
    bk: ?cuda.DeviceMem,
    bv: ?cuda.DeviceMem,
    q_norm: ?cuda.DeviceMem,
    k_norm: ?cuda.DeviceMem,
    ffn_norm: cuda.DeviceMem,
    w_gate: DevMat,
    w_up: DevMat,
    w_down: DevMat,
};

const Kernels = struct {
    rms_norm: cuda.Kernel,
    head_norm: cuda.Kernel,
    rope: cuda.Kernel,
    kv_write: cuda.Kernel,
    attention: cuda.Kernel,
    swiglu: cuda.Kernel,
    add: cuda.Kernel,
};

/// Bytes the engine needs on the device for `w` with options `o`, with a margin for the driver.
pub fn bytesNeeded(w: *const weights_mod.Weights, cfg: Config, o: Options) u64 {
    var total: u64 = 0;
    for (w.layers) |l| {
        for ([_]Mat{ l.wq, l.wk, l.wv, l.wo, l.w_gate, l.w_up, l.w_down }) |m| total += m.data.len;
        total += 4 * (l.attn_norm.len + l.ffn_norm.len);
        for ([_]?[]const f32{ l.bq, l.bk, l.bv, l.q_norm, l.k_norm }) |v| {
            if (v) |s| total += 4 * s.len;
        }
    }
    total += w.output.data.len + 4 * w.out_norm.len;
    const kv_per_layer: u64 = 2 * @as(u64, o.slots) * o.n_ctx * cfg.n_kv_heads * cfg.head_dim * 2;
    total += kv_per_layer * cfg.n_layers;
    total += @as(u64, cfg.n_heads) * o.n_ctx * 4; // attention scores
    const act: u64 = (6 * @as(u64, cfg.dim) + 2 * cfg.qDim() + 2 * cfg.kvDim() + 2 * cfg.ffn_dim) * 4;
    total += act + @as(u64, o.n_batch) * (cfg.dim + cfg.rope_dim) * 4; // hidden rows, cos/sin
    total += @as(u64, max_logit_rows) * cfg.vocab * 4;
    return total + (64 << 20);
}

pub const Engine = struct {
    allocator: std.mem.Allocator,
    cfg: Config,
    n_batch: usize,
    n_ctx: usize,
    n_slots: usize,
    max_rows: usize,
    layers: []DevLayer,
    out_norm: cuda.DeviceMem,
    output: DevMat,
    tok_embd: Mat,
    inv_freq: []const f32,
    mscale: f32,
    k: Kernels,
    gemv: std.EnumArray(StorageType, ?cuda.Kernel),

    /// Single token activations.
    xb: cuda.DeviceMem,
    q: cuda.DeviceMem,
    kk: cuda.DeviceMem,
    vv: cuda.DeviceMem,
    att: cuda.DeviceMem,
    tmp: cuda.DeviceMem,
    gate: cuda.DeviceMem,
    up: cuda.DeviceMem,
    scores: cuda.DeviceMem,
    /// Final hidden state of every item of the last pass, and the residual stream while it runs.
    hidden: cuda.DeviceMem,
    cos_dev: cuda.DeviceMem,
    sin_dev: cuda.DeviceMem,
    logits_dev: cuda.DeviceMem,
    /// Per slot key and value caches: [layer][kv head][position][head_dim] in f16.
    kcache: []cuda.DeviceMem,
    vcache: []cuda.DeviceMem,
    owners: []?*const kv_mod.KvCache,

    embed_h: []f32,
    cos_h: []f32,
    sin_h: []f32,
    logits_h: []f32,
    n_last: usize = 0,
    name_buf: [128]u8 = undefined,
    name_len: usize = 0,

    fn mapErr(e: anyerror) Error {
        return switch (e) {
            error.OutOfDeviceMemory => error.OutOfDeviceMemory,
            error.OutOfMemory => error.OutOfMemory,
            else => error.DeviceFailed,
        };
    }

    pub fn init(
        allocator: std.mem.Allocator,
        w: *const weights_mod.Weights,
        cfg: Config,
        inv_freq: []const f32,
        mscale: f32,
        o: Options,
    ) Error!Engine {
        if (cfg.head_dim > max_head_dim or cfg.head_dim % 2 != 0) return error.UnsupportedShape;
        if (cfg.n_heads % cfg.n_kv_heads != 0 or cfg.rope_dim > cfg.head_dim) return error.UnsupportedShape;
        for (w.layers) |l| {
            for ([_]Mat{ l.wq, l.wk, l.wv, l.wo, l.w_gate, l.w_up, l.w_down }) |m| if (!supports(m.t)) return error.UnsupportedType;
        }
        if (!supports(w.output.t)) return error.UnsupportedType;
        if (!supports(w.tok_embd.t)) return error.UnsupportedType;

        var self: Engine = undefined;
        self.allocator = allocator;
        self.cfg = cfg;
        self.n_batch = o.n_batch;
        self.n_ctx = o.n_ctx;
        self.n_slots = o.slots;
        self.max_rows = @min(o.n_batch, max_logit_rows);
        self.tok_embd = w.tok_embd;
        self.inv_freq = inv_freq;
        self.mscale = mscale;
        self.n_last = 0;
        self.name_len = 0;
        if (cuda.getDeviceName(&self.name_buf)) |nm| self.name_len = nm.len else |_| {}

        self.k = .{
            .rms_norm = cuda.deviceKernel("rms_norm") catch return error.DeviceFailed,
            .head_norm = cuda.deviceKernel("head_norm") catch return error.DeviceFailed,
            .rope = cuda.deviceKernel("rope") catch return error.DeviceFailed,
            .kv_write = cuda.deviceKernel("kv_write") catch return error.DeviceFailed,
            .attention = cuda.deviceKernel("attention") catch return error.DeviceFailed,
            .swiglu = cuda.deviceKernel("swiglu") catch return error.DeviceFailed,
            .add = cuda.deviceKernel("add") catch return error.DeviceFailed,
        };
        self.gemv = std.EnumArray(StorageType, ?cuda.Kernel).initFill(null);

        // Everything below is released by `deinit`; track what exists so a failure part way frees exactly that.
        var made = Made{ .allocator = allocator };
        errdefer made.release();

        self.layers = try allocator.alloc(DevLayer, w.layers.len);
        made.layers = self.layers;
        for (w.layers, self.layers) |l, *d| {
            d.attn_norm = try made.vec(l.attn_norm);
            d.wq = try self.upload(&made, l.wq);
            d.wk = try self.upload(&made, l.wk);
            d.wv = try self.upload(&made, l.wv);
            d.wo = try self.upload(&made, l.wo);
            d.bq = try made.optVec(l.bq);
            d.bk = try made.optVec(l.bk);
            d.bv = try made.optVec(l.bv);
            d.q_norm = try made.optVec(l.q_norm);
            d.k_norm = try made.optVec(l.k_norm);
            d.ffn_norm = try made.vec(l.ffn_norm);
            d.w_gate = try self.upload(&made, l.w_gate);
            d.w_up = try self.upload(&made, l.w_up);
            d.w_down = try self.upload(&made, l.w_down);
        }
        self.out_norm = try made.vec(w.out_norm);
        self.output = try self.upload(&made, w.output);

        const f = 4; // bytes of an f32
        self.xb = try made.empty(cfg.dim * f);
        self.q = try made.empty(cfg.qDim() * f);
        self.kk = try made.empty(cfg.kvDim() * f);
        self.vv = try made.empty(cfg.kvDim() * f);
        self.att = try made.empty(cfg.qDim() * f);
        self.tmp = try made.empty(cfg.dim * f);
        self.gate = try made.empty(cfg.ffn_dim * f);
        self.up = try made.empty(cfg.ffn_dim * f);
        self.scores = try made.empty(cfg.n_heads * o.n_ctx * f);
        self.hidden = try made.empty(o.n_batch * cfg.dim * f);
        const half = cfg.rope_dim / 2;
        self.cos_dev = try made.empty(o.n_batch * half * f);
        self.sin_dev = try made.empty(o.n_batch * half * f);
        self.logits_dev = try made.empty(self.max_rows * cfg.vocab * f);

        const kv_bytes = cfg.n_layers * cfg.n_kv_heads * o.n_ctx * cfg.head_dim * 2;
        self.kcache = try allocator.alloc(cuda.DeviceMem, o.slots);
        made.kcache = self.kcache;
        self.vcache = try allocator.alloc(cuda.DeviceMem, o.slots);
        made.vcache = self.vcache;
        for (self.kcache, self.vcache) |*kc, *vc| {
            kc.* = try made.empty(kv_bytes);
            vc.* = try made.empty(kv_bytes);
        }

        self.owners = try allocator.alloc(?*const kv_mod.KvCache, o.slots);
        errdefer allocator.free(self.owners);
        @memset(self.owners, null);
        self.embed_h = try allocator.alloc(f32, o.n_batch * cfg.dim);
        errdefer allocator.free(self.embed_h);
        self.cos_h = try allocator.alloc(f32, o.n_batch * half);
        errdefer allocator.free(self.cos_h);
        self.sin_h = try allocator.alloc(f32, o.n_batch * half);
        errdefer allocator.free(self.sin_h);
        self.logits_h = try allocator.alloc(f32, self.max_rows * cfg.vocab);
        errdefer allocator.free(self.logits_h);

        made.disarm();
        return self;
    }

    /// Records device allocations so `release` frees whatever a failed `init` had made.
    const Made = struct {
        allocator: std.mem.Allocator,
        mems: std.ArrayList(cuda.DeviceMem) = .empty,
        layers: []DevLayer = &.{},
        kcache: []cuda.DeviceMem = &.{},
        vcache: []cuda.DeviceMem = &.{},
        armed: bool = true,

        fn track(self: *Made, m: cuda.DeviceMem) Error!cuda.DeviceMem {
            self.mems.append(self.allocator, m) catch {
                cuda.deviceFree(m);
                return error.OutOfMemory;
            };
            return m;
        }

        fn empty(self: *Made, nbytes: usize) Error!cuda.DeviceMem {
            return self.track(cuda.deviceAlloc(nbytes) catch |e| return mapErr(e));
        }

        fn bytes(self: *Made, src: []const u8) Error!cuda.DeviceMem {
            const m = try self.empty(src.len);
            cuda.copyToDevice(m.addr, src) catch |e| return mapErr(e);
            return m;
        }

        fn vec(self: *Made, v: []const f32) Error!cuda.DeviceMem {
            return self.bytes(std.mem.sliceAsBytes(v));
        }

        fn optVec(self: *Made, v: ?[]const f32) Error!?cuda.DeviceMem {
            return if (v) |s| try self.vec(s) else null;
        }

        fn disarm(self: *Made) void {
            self.armed = false;
            // The allocation list is only for failure cleanup; `deinit` frees from the engine's own fields.
            self.mems.deinit(self.allocator);
        }

        fn release(self: *Made) void {
            if (!self.armed) return;
            for (self.mems.items) |m| cuda.deviceFree(m);
            self.mems.deinit(self.allocator);
            if (self.layers.len != 0) self.allocator.free(self.layers);
            if (self.kcache.len != 0) self.allocator.free(self.kcache);
            if (self.vcache.len != 0) self.allocator.free(self.vcache);
        }
    };

    fn upload(self: *Engine, made: *Made, m: Mat) Error!DevMat {
        if (self.gemv.get(m.t) == null) {
            var nbuf: [48]u8 = undefined;
            const name = std.fmt.bufPrint(&nbuf, "gemv_{s}", .{@tagName(m.t)}) catch return error.UnsupportedType;
            self.gemv.set(m.t, cuda.deviceKernel(name) catch return error.UnsupportedType);
        }
        return .{ .mem = try made.bytes(m.data), .t = m.t, .rows = m.rows, .cols = m.cols };
    }

    pub fn deinit(self: *Engine) void {
        const a = self.allocator;
        for (self.layers) |l| {
            for ([_]DevMat{ l.wq, l.wk, l.wv, l.wo, l.w_gate, l.w_up, l.w_down }) |m| cuda.deviceFree(m.mem);
            for ([_]?cuda.DeviceMem{ l.attn_norm, l.ffn_norm, l.bq, l.bk, l.bv, l.q_norm, l.k_norm }) |v| {
                if (v) |mem| cuda.deviceFree(mem);
            }
        }
        a.free(self.layers);
        cuda.deviceFree(self.out_norm);
        cuda.deviceFree(self.output.mem);
        for ([_]cuda.DeviceMem{ self.xb, self.q, self.kk, self.vv, self.att, self.tmp, self.gate, self.up, self.scores, self.hidden, self.cos_dev, self.sin_dev, self.logits_dev }) |m| cuda.deviceFree(m);
        for (self.kcache, self.vcache) |kc, vc| {
            cuda.deviceFree(kc);
            cuda.deviceFree(vc);
        }
        a.free(self.kcache);
        a.free(self.vcache);
        a.free(self.owners);
        a.free(self.embed_h);
        a.free(self.cos_h);
        a.free(self.sin_h);
        a.free(self.logits_h);
    }

    pub fn deviceName(self: *const Engine) []const u8 {
        return self.name_buf[0..self.name_len];
    }

    /// The device slot of a sequence's cache, assigning one on first use.
    pub fn slotFor(self: *Engine, kv: *const kv_mod.KvCache) Error!usize {
        for (self.owners, 0..) |o, i| if (o == kv) return i;
        for (self.owners, 0..) |o, i| {
            if (o == null) {
                self.owners[i] = kv;
                return i;
            }
        }
        return error.NoGpuSlot;
    }

    /// Frees the slot of a cache that is going away.
    pub fn release(self: *Engine, kv: *const kv_mod.KvCache) void {
        for (self.owners) |*o| if (o.* == kv) {
            o.* = null;
        };
    }

    // -- launches ------------------------------------------------------------------------------------------------

    fn grid(n: usize, per: u32) u32 {
        return @intCast((n + per - 1) / per);
    }

    fn mv(self: *Engine, m: DevMat, x: u64, y: u64) Error!void {
        const kern = self.gemv.get(m.t).?;
        cuda.launchKernel(kern, @intCast(m.rows), block_threads, .{ m.mem.addr, x, y, @as(u32, @intCast(m.cols)) }) catch |e| return mapErr(e);
    }

    fn rmsNorm(self: *Engine, out: u64, x: u64, w: u64) Error!void {
        cuda.launchKernel(self.k.rms_norm, 1, block_threads, .{ out, x, w, @as(u32, @intCast(self.cfg.dim)), self.cfg.rms_eps }) catch |e| return mapErr(e);
    }

    fn addInto(self: *Engine, x: u64, y: u64, n: usize) Error!void {
        cuda.launchKernel(self.k.add, grid(n, elem_threads), elem_threads, .{ x, y, @as(u32, @intCast(n)) }) catch |e| return mapErr(e);
    }

    /// Runs the whole network for the token whose embedding is in `hidden[row]`, at position `pos` of `slot`.
    fn runItem(self: *Engine, row: usize, slot: usize, pos: u32) Error!void {
        const cfg = self.cfg;
        const f = 4;
        const x = self.hidden.at(row * cfg.dim * f);
        const half = cfg.rope_dim / 2;
        const cos = self.cos_dev.at(row * half * f);
        const sin = self.sin_dev.at(row * half * f);
        const neox: u32 = if (cfg.rope_style == .neox) 1 else 0;
        const hd: u32 = @intCast(cfg.head_dim);
        const group: u32 = @intCast(cfg.n_heads / cfg.n_kv_heads);
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
        const layer_elems = cfg.n_kv_heads * self.n_ctx * cfg.head_dim;

        for (self.layers, 0..) |l, li| {
            try self.rmsNorm(self.xb.addr, x, l.attn_norm.addr);
            try self.mv(l.wq, self.xb.addr, self.q.addr);
            try self.mv(l.wk, self.xb.addr, self.kk.addr);
            try self.mv(l.wv, self.xb.addr, self.vv.addr);
            if (l.bq) |b| try self.addInto(self.q.addr, b.addr, cfg.qDim());
            if (l.bk) |b| try self.addInto(self.kk.addr, b.addr, cfg.kvDim());
            if (l.bv) |b| try self.addInto(self.vv.addr, b.addr, cfg.kvDim());

            if (l.q_norm) |w| cuda.launchKernel(self.k.head_norm, @intCast(cfg.n_heads), block_threads, .{ self.q.addr, w.addr, hd, cfg.rms_eps }) catch |e| return mapErr(e);
            if (l.k_norm) |w| cuda.launchKernel(self.k.head_norm, @intCast(cfg.n_kv_heads), block_threads, .{ self.kk.addr, w.addr, hd, cfg.rms_eps }) catch |e| return mapErr(e);
            cuda.launchKernel(self.k.rope, @intCast(cfg.n_heads), block_threads, .{ self.q.addr, cos, sin, hd, @as(u32, @intCast(half)), neox }) catch |e| return mapErr(e);
            cuda.launchKernel(self.k.rope, @intCast(cfg.n_kv_heads), block_threads, .{ self.kk.addr, cos, sin, hd, @as(u32, @intCast(half)), neox }) catch |e| return mapErr(e);

            const kbase = self.kcache[slot].at(li * layer_elems * 2);
            const vbase = self.vcache[slot].at(li * layer_elems * 2);
            const kvd: u32 = @intCast(cfg.kvDim());
            cuda.launchKernel(self.k.kv_write, grid(cfg.kvDim(), elem_threads), elem_threads, .{ self.kk.addr, self.vv.addr, kbase, vbase, pos, @as(u32, @intCast(self.n_ctx)), hd, kvd }) catch |e| return mapErr(e);
            cuda.launchKernel(self.k.attention, @intCast(cfg.n_heads), block_threads, .{ self.q.addr, kbase, vbase, self.att.addr, self.scores.addr, pos, @as(u32, @intCast(self.n_ctx)), hd, group, scale }) catch |e| return mapErr(e);

            try self.mv(l.wo, self.att.addr, self.tmp.addr);
            try self.addInto(x, self.tmp.addr, cfg.dim);

            try self.rmsNorm(self.xb.addr, x, l.ffn_norm.addr);
            try self.mv(l.w_gate, self.xb.addr, self.gate.addr);
            try self.mv(l.w_up, self.xb.addr, self.up.addr);
            cuda.launchKernel(self.k.swiglu, grid(cfg.ffn_dim, elem_threads), elem_threads, .{ self.gate.addr, self.up.addr, @as(u32, @intCast(cfg.ffn_dim)) }) catch |e| return mapErr(e);
            try self.mv(l.w_down, self.gate.addr, self.tmp.addr);
            try self.addInto(x, self.tmp.addr, cfg.dim);
        }
    }

    /// Runs one forward pass over `items`. Items that share a cache must appear in increasing position order.
    pub fn forward(self: *Engine, items: []const kv_mod.Item) Error!void {
        const n = items.len;
        const cfg = self.cfg;
        std.debug.assert(n >= 1 and n <= self.n_batch);
        const half = cfg.rope_dim / 2;
        const rb = self.tok_embd.rowBytes();
        for (items, 0..) |it, t| {
            if (it.pos >= self.n_ctx) return error.ContextFull;
            const row = self.tok_embd.data[@as(usize, it.token) * rb ..][0..rb];
            quant.dequantizeRow(self.tok_embd.t, row, self.embed_h[t * cfg.dim ..][0..cfg.dim]) catch return error.UnsupportedType;
            ops.ropeAngles(self.cos_h[t * half ..][0..half], self.sin_h[t * half ..][0..half], self.inv_freq, it.pos, self.mscale);
        }
        cuda.copyToDevice(self.hidden.addr, std.mem.sliceAsBytes(self.embed_h[0 .. n * cfg.dim])) catch |e| return mapErr(e);
        cuda.copyToDevice(self.cos_dev.addr, std.mem.sliceAsBytes(self.cos_h[0 .. n * half])) catch |e| return mapErr(e);
        cuda.copyToDevice(self.sin_dev.addr, std.mem.sliceAsBytes(self.sin_h[0 .. n * half])) catch |e| return mapErr(e);

        for (items, 0..) |it, t| {
            const slot = try self.slotFor(it.kv);
            try self.runItem(t, slot, it.pos);
        }
        self.n_last = n;
    }

    /// Logits of the given items of the last pass, one vocabulary row each, in the order of `indices`. The slice stays
    /// valid until the next call.
    pub fn logits(self: *Engine, indices: []const usize) Error![]const f32 {
        std.debug.assert(indices.len >= 1 and indices.len <= self.max_rows);
        const cfg = self.cfg;
        const f = 4;
        for (indices, 0..) |idx, r| {
            std.debug.assert(idx < self.n_last);
            try self.rmsNorm(self.xb.addr, self.hidden.at(idx * cfg.dim * f), self.out_norm.addr);
            try self.mv(self.output, self.xb.addr, self.logits_dev.at(r * cfg.vocab * f));
        }
        const out = self.logits_h[0 .. indices.len * cfg.vocab];
        cuda.copyFromDevice(std.mem.sliceAsBytes(out), self.logits_dev.addr) catch |e| return mapErr(e);
        return out;
    }
};
