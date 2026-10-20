//! The CUDA engine: the whole network on one NVIDIA device, driven through the CUDA driver API (loaded at run time)
//! with kernels written in Zig (`src/cuda_device.zig`).
//!
//! Scope, stated plainly:
//!   * a pass runs the whole batch through each layer at once: the matrix kernel decodes every weight block once for
//!     groups of eight tokens, and attention, RoPE and the cache writes are batched over tokens. There is still no
//!     fusion, no CUDA graphs (one launch per kernel) and no Tensor Cores; those are later stages of HK 2.0/2.1;
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

fn nowSecs() f64 {
    return @as(f64, @floatFromInt(std.Io.Timestamp.now(std.Options.debug_io, .awake).nanoseconds)) / 1e9;
}

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

/// The matrix kernels of one weight format: matrix-vector for a single token, the tiled kernel for small batches and,
/// on GPUs of compute capability 8.0 and up, the tensor core kernel for prompts.
const GemmKernels = struct { decode: cuda.Kernel, tiled: cuda.Kernel, tensor: ?cuda.Kernel };
/// Threads per block of the elementwise kernels.
const elem_threads: u32 = 256;
/// Threads per block of the attention kernel: one per position of a tile (`attn_tile` in the device code).
const attn_threads: u32 = 128;
/// Largest head width the attention kernel supports.
const max_head_dim = 256;
/// Split-KV attention: slices of the context per (token, head) at most, and the smallest slice worth a block.
const max_splits = 8;
const min_split_positions = 256;
/// Attention uses one block per (token, head) when there are at least this many of them; fewer is decode-like work
/// that cannot fill the device without slicing the context.
const split_below_blocks = 64;
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
    add_bias: cuda.Kernel,
    kv_write: cuda.Kernel,
    attention: cuda.Kernel,
    attention_part: cuda.Kernel,
    attention_combine: cuda.Kernel,
    /// Tensor core prompt attention for this model's head size; null when the device has none or the size is not 64 / 128.
    attention_tc: ?cuda.Kernel,
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
    // Activations are [batch][...]: xb, hidden, tmp (dim); q, att (qDim); k, v (kvDim); gate, up (ffn); cos, sin.
    const per_token: u64 = (3 * @as(u64, cfg.dim) + 2 * cfg.qDim() + 2 * cfg.kvDim() + 2 * cfg.ffn_dim + cfg.rope_dim) * 4 + 8;
    total += per_token * o.n_batch;
    total += @as(u64, split_below_blocks) * max_splits * (cfg.head_dim + 2) * 4;
    total += 16 * @as(u64, o.slots);
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
    /// Matrix kernels per storage type: [0] decode (one token), [1] batch (tiled).
    gemm: std.EnumArray(StorageType, ?GemmKernels),

    /// Single token activations.
    xb: cuda.DeviceMem,
    q: cuda.DeviceMem,
    kk: cuda.DeviceMem,
    vv: cuda.DeviceMem,
    att: cuda.DeviceMem,
    /// Partial softmax states of split-KV attention: [block][head_dim + 2] floats.
    attn_part: cuda.DeviceMem,
    tmp: cuda.DeviceMem,
    gate: cuda.DeviceMem,
    up: cuda.DeviceMem,
    /// Final hidden state of every item of the last pass, and the residual stream while it runs.
    hidden: cuda.DeviceMem,
    cos_dev: cuda.DeviceMem,
    sin_dev: cuda.DeviceMem,
    logits_dev: cuda.DeviceMem,
    /// Base addresses of every slot's caches, and the slot and position of each item of the current pass.
    slot_k_tbl: cuda.DeviceMem,
    slot_v_tbl: cuda.DeviceMem,
    item_slot: cuda.DeviceMem,
    item_pos: cuda.DeviceMem,
    /// Per slot key and value caches: [layer][kv head][position][head_dim] in f16.
    kcache: []cuda.DeviceMem,
    vcache: []cuda.DeviceMem,
    owners: []?*const kv_mod.KvCache,

    slot_h: []u32,
    pos_h: []u32,
    embed_h: []f32,
    cos_h: []f32,
    sin_h: []f32,
    logits_h: []f32,
    n_last: usize = 0,
    /// Largest position of the current pass, which decides how finely attention is split.
    max_pos: u32 = 0,
    /// The current pass is consecutive positions of one sequence, which is what the tensor core attention takes.
    one_run: bool = false,
    /// Host seconds spent issuing launches (forward) and waiting for results (logits), summed; for HK_GPU_PROFILE.
    prof_launch_s: f64 = 0,
    prof_wait_s: f64 = 0,
    prof_passes: u64 = 0,
    /// GPU milliseconds across whole passes (two events around a pass, no per kernel syncing), when enabled.
    prof_gpu_ms: f64 = 0,
    pass_events: ?[2]cuda.CudaEvent = null,
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
        // `self` starts undefined, so the fields that have declared defaults must be set here too.
        self.n_last = 0;
        self.max_pos = 0;
        self.one_run = false;
        self.name_len = 0;
        self.prof_launch_s = 0;
        self.prof_wait_s = 0;
        self.prof_passes = 0;
        self.prof_gpu_ms = 0;
        self.pass_events = null;
        if (cuda.getDeviceName(&self.name_buf)) |nm| self.name_len = nm.len else |_| {}

        self.k = .{
            .rms_norm = cuda.deviceKernel("rms_norm") catch return error.DeviceFailed,
            .head_norm = cuda.deviceKernel("head_norm") catch return error.DeviceFailed,
            .rope = cuda.deviceKernel("rope") catch return error.DeviceFailed,
            .add_bias = cuda.deviceKernel("add_bias") catch return error.DeviceFailed,
            .kv_write = cuda.deviceKernel("kv_write") catch return error.DeviceFailed,
            .attention = cuda.deviceKernel("attention") catch return error.DeviceFailed,
            .attention_part = cuda.deviceKernel("attention_part") catch return error.DeviceFailed,
            .attention_combine = cuda.deviceKernel("attention_combine") catch return error.DeviceFailed,
            .attention_tc = if (!cuda.hasTensorCores()) null else switch (cfg.head_dim) {
                64 => cuda.deviceKernel("attention_tc_64") catch null,
                128 => cuda.deviceKernel("attention_tc_128") catch null,
                else => null,
            },
            .swiglu = cuda.deviceKernel("swiglu") catch return error.DeviceFailed,
            .add = cuda.deviceKernel("add") catch return error.DeviceFailed,
        };
        self.gemm = std.EnumArray(StorageType, ?GemmKernels).initFill(null);

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
        const rows = o.n_batch;
        self.xb = try made.empty(rows * cfg.dim * f);
        self.q = try made.empty(rows * cfg.qDim() * f);
        self.kk = try made.empty(rows * cfg.kvDim() * f);
        self.vv = try made.empty(rows * cfg.kvDim() * f);
        self.att = try made.empty(rows * cfg.qDim() * f);
        self.attn_part = try made.empty(split_below_blocks * max_splits * (cfg.head_dim + 2) * f);
        self.tmp = try made.empty(rows * cfg.dim * f);
        self.gate = try made.empty(rows * cfg.ffn_dim * f);
        self.up = try made.empty(rows * cfg.ffn_dim * f);
        self.hidden = try made.empty(rows * cfg.dim * f);
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

        self.slot_k_tbl = try made.empty(o.slots * 8);
        self.slot_v_tbl = try made.empty(o.slots * 8);
        self.item_slot = try made.empty(o.n_batch * 4);
        self.item_pos = try made.empty(o.n_batch * 4);
        {
            const kt = try allocator.alloc(u64, o.slots);
            defer allocator.free(kt);
            const vt = try allocator.alloc(u64, o.slots);
            defer allocator.free(vt);
            for (self.kcache, self.vcache, kt, vt) |kc, vc, *k_, *v_| {
                k_.* = kc.addr;
                v_.* = vc.addr;
            }
            cuda.copyToDevice(self.slot_k_tbl.addr, std.mem.sliceAsBytes(kt)) catch |e| return mapErr(e);
            cuda.copyToDevice(self.slot_v_tbl.addr, std.mem.sliceAsBytes(vt)) catch |e| return mapErr(e);
        }

        self.owners = try allocator.alloc(?*const kv_mod.KvCache, o.slots);
        errdefer allocator.free(self.owners);
        @memset(self.owners, null);
        self.slot_h = try allocator.alloc(u32, o.n_batch);
        errdefer allocator.free(self.slot_h);
        self.pos_h = try allocator.alloc(u32, o.n_batch);
        errdefer allocator.free(self.pos_h);
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
        if (self.gemm.get(m.t) == null) {
            const one = cuda.gemmKernel(m.t, 1) catch return error.UnsupportedType;
            const many = cuda.gemmKernel(m.t, 2) catch return error.UnsupportedType;
            self.gemm.set(m.t, .{ .decode = one, .tiled = many, .tensor = cuda.tensorKernel(m.t) });
        }
        return .{ .mem = try made.bytes(m.data), .t = m.t, .rows = m.rows, .cols = m.cols };
    }

    /// True when prompt passes of one sequence run the tensor core attention kernel.
    pub fn promptAttentionIsTensor(self: *const Engine) bool {
        return self.k.attention_tc != null;
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
        for ([_]cuda.DeviceMem{ self.xb, self.q, self.kk, self.vv, self.att, self.attn_part, self.tmp, self.gate, self.up, self.hidden, self.cos_dev, self.sin_dev, self.logits_dev, self.slot_k_tbl, self.slot_v_tbl, self.item_slot, self.item_pos }) |m| cuda.deviceFree(m);
        for (self.kcache, self.vcache) |kc, vc| {
            cuda.deviceFree(kc);
            cuda.deviceFree(vc);
        }
        a.free(self.kcache);
        a.free(self.vcache);
        a.free(self.owners);
        a.free(self.slot_h);
        a.free(self.pos_h);
        a.free(self.embed_h);
        a.free(self.cos_h);
        a.free(self.sin_h);
        a.free(self.logits_h);
    }

    /// Starts timing whole passes on the device (see `prof_gpu_ms`).
    pub fn enablePassTiming(self: *Engine) Error!void {
        if (self.pass_events != null) return;
        const a = cuda.CudaEvent.create() catch |e| return mapErr(e);
        const b = cuda.CudaEvent.create() catch |e| return mapErr(e);
        self.pass_events = .{ a, b };
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

    /// y[token][0..rows] = m * x[token] for `n` tokens.
    fn mm(self: *Engine, m: DevMat, x: u64, y: u64, n: usize) Error!void {
        const kernels = self.gemm.get(m.t).?;
        if (n >= cuda.tensor_min_tokens) if (kernels.tensor) |tc| {
            const gx: u32 = @intCast((m.rows + cuda.tensor_tile_rows - 1) / cuda.tensor_tile_rows);
            const gy: u32 = @intCast((n + cuda.tensor_tile_tokens - 1) / cuda.tensor_tile_tokens);
            return cuda.launchKernel2D(tc, gx, gy, cuda.tensor_block_threads, .{ m.mem.addr, x, y, @as(u32, @intCast(m.cols)), @as(u32, @intCast(m.rows)), @as(u32, @intCast(n)) }) catch |e| return mapErr(e);
        };
        const kern = if (n <= 1) kernels.decode else kernels.tiled;
        const g = cuda.gemmGrid(m.rows, n);
        cuda.launchKernel2D(kern, g[0], g[1], block_threads, .{ m.mem.addr, x, y, @as(u32, @intCast(m.cols)), @as(u32, @intCast(m.rows)), @as(u32, @intCast(n)) }) catch |e| return mapErr(e);
    }

    fn rmsNorm(self: *Engine, out: u64, x: u64, w: u64, rows: usize) Error!void {
        cuda.launchKernel(self.k.rms_norm, @intCast(rows), block_threads, .{ out, x, w, @as(u32, @intCast(self.cfg.dim)), self.cfg.rms_eps }) catch |e| return mapErr(e);
    }

    fn addInto(self: *Engine, x: u64, y: u64, n: usize) Error!void {
        cuda.launchKernel(self.k.add, grid(n, elem_threads), elem_threads, .{ x, y, @as(u32, @intCast(n)) }) catch |e| return mapErr(e);
    }

    fn addBias(self: *Engine, x: u64, bias: u64, len: usize, rows: usize) Error!void {
        const total = len * rows;
        cuda.launchKernel(self.k.add_bias, grid(total, elem_threads), elem_threads, .{ x, bias, @as(u32, @intCast(len)), @as(u32, @intCast(total)) }) catch |e| return mapErr(e);
    }

    /// Runs the whole network for the `n` tokens whose embeddings are in `hidden`, one layer at a time.
    fn runBatch(self: *Engine, n: usize) Error!void {
        const cfg = self.cfg;
        const half = cfg.rope_dim / 2;
        const neox: u32 = if (cfg.rope_style == .neox) 1 else 0;
        const hd: u32 = @intCast(cfg.head_dim);
        const n_heads: u32 = @intCast(cfg.n_heads);
        const n_kv: u32 = @intCast(cfg.n_kv_heads);
        const group: u32 = n_heads / n_kv;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
        const layer_elems = cfg.n_kv_heads * self.n_ctx * cfg.head_dim;
        const ctx: u32 = @intCast(self.n_ctx);
        const x = self.hidden.addr;
        // Few (token, head) blocks cannot fill the device, so a long context is sliced across several blocks each.
        const splits: u32 = if (n * cfg.n_heads >= split_below_blocks) 1 else @intCast(std.math.clamp((self.max_pos + min_split_positions) / min_split_positions, 1, max_splits));

        for (self.layers, 0..) |l, li| {
            try self.rmsNorm(self.xb.addr, x, l.attn_norm.addr, n);
            try self.mm(l.wq, self.xb.addr, self.q.addr, n);
            try self.mm(l.wk, self.xb.addr, self.kk.addr, n);
            try self.mm(l.wv, self.xb.addr, self.vv.addr, n);
            if (l.bq) |b| try self.addBias(self.q.addr, b.addr, cfg.qDim(), n);
            if (l.bk) |b| try self.addBias(self.kk.addr, b.addr, cfg.kvDim(), n);
            if (l.bv) |b| try self.addBias(self.vv.addr, b.addr, cfg.kvDim(), n);

            if (l.q_norm) |w| cuda.launchKernel(self.k.head_norm, @intCast(n * cfg.n_heads), block_threads, .{ self.q.addr, w.addr, hd, cfg.rms_eps }) catch |e| return mapErr(e);
            if (l.k_norm) |w| cuda.launchKernel(self.k.head_norm, @intCast(n * cfg.n_kv_heads), block_threads, .{ self.kk.addr, w.addr, hd, cfg.rms_eps }) catch |e| return mapErr(e);
            cuda.launchKernel(self.k.rope, @intCast(n * cfg.n_heads), block_threads, .{ self.q.addr, self.cos_dev.addr, self.sin_dev.addr, hd, @as(u32, @intCast(half)), neox, n_heads }) catch |e| return mapErr(e);
            cuda.launchKernel(self.k.rope, @intCast(n * cfg.n_kv_heads), block_threads, .{ self.kk.addr, self.cos_dev.addr, self.sin_dev.addr, hd, @as(u32, @intCast(half)), neox, n_kv }) catch |e| return mapErr(e);

            // Keys and values of the whole batch go in first so a token sees its predecessors in the same pass.
            const layer_off: u64 = @as(u64, li) * layer_elems * 2;
            const kvd: u32 = @intCast(cfg.kvDim());
            const kv_total: u32 = @intCast(n * cfg.kvDim());
            cuda.launchKernel(self.k.kv_write, grid(kv_total, elem_threads), elem_threads, .{ self.kk.addr, self.vv.addr, self.slot_k_tbl.addr, self.slot_v_tbl.addr, layer_off, self.item_slot.addr, self.item_pos.addr, ctx, hd, kvd, kv_total }) catch |e| return mapErr(e);
            if (self.k.attention_tc != null and self.one_run and n >= cuda.tensor_min_tokens) {
                cuda.launchKernel2D(self.k.attention_tc.?, @intCast((n + 63) / 64), n_heads, 128, .{ self.q.addr, self.slot_k_tbl.addr, self.slot_v_tbl.addr, layer_off, self.item_slot.addr, self.item_pos.addr, self.att.addr, ctx, @as(u32, @intCast(n)), n_heads, group, scale }) catch |e| return mapErr(e);
            } else if (splits == 1) {
                cuda.launchKernel(self.k.attention, @intCast(n * cfg.n_heads), attn_threads, .{ self.q.addr, self.slot_k_tbl.addr, self.slot_v_tbl.addr, layer_off, self.item_slot.addr, self.item_pos.addr, self.att.addr, ctx, hd, n_heads, group, scale }) catch |e| return mapErr(e);
            } else {
                cuda.launchKernel(self.k.attention_part, @intCast(n * cfg.n_heads * splits), attn_threads, .{ self.q.addr, self.slot_k_tbl.addr, self.slot_v_tbl.addr, layer_off, self.item_slot.addr, self.item_pos.addr, self.attn_part.addr, ctx, hd, n_heads, group, scale, splits }) catch |e| return mapErr(e);
                cuda.launchKernel(self.k.attention_combine, @intCast(n * cfg.n_heads), attn_threads, .{ self.attn_part.addr, self.att.addr, hd, splits }) catch |e| return mapErr(e);
            }

            try self.mm(l.wo, self.att.addr, self.tmp.addr, n);
            try self.addInto(x, self.tmp.addr, n * cfg.dim);

            try self.rmsNorm(self.xb.addr, x, l.ffn_norm.addr, n);
            try self.mm(l.w_gate, self.xb.addr, self.gate.addr, n);
            try self.mm(l.w_up, self.xb.addr, self.up.addr, n);
            cuda.launchKernel(self.k.swiglu, grid(n * cfg.ffn_dim, elem_threads), elem_threads, .{ self.gate.addr, self.up.addr, @as(u32, @intCast(n * cfg.ffn_dim)) }) catch |e| return mapErr(e);
            try self.mm(l.w_down, self.gate.addr, self.tmp.addr, n);
            try self.addInto(x, self.tmp.addr, n * cfg.dim);
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
        for (items, 0..) |it, t| {
            self.slot_h[t] = @intCast(try self.slotFor(it.kv));
            self.pos_h[t] = it.pos;
        }
        cuda.copyToDevice(self.hidden.addr, std.mem.sliceAsBytes(self.embed_h[0 .. n * cfg.dim])) catch |e| return mapErr(e);
        cuda.copyToDevice(self.cos_dev.addr, std.mem.sliceAsBytes(self.cos_h[0 .. n * half])) catch |e| return mapErr(e);
        cuda.copyToDevice(self.sin_dev.addr, std.mem.sliceAsBytes(self.sin_h[0 .. n * half])) catch |e| return mapErr(e);
        cuda.copyToDevice(self.item_slot.addr, std.mem.sliceAsBytes(self.slot_h[0..n])) catch |e| return mapErr(e);
        cuda.copyToDevice(self.item_pos.addr, std.mem.sliceAsBytes(self.pos_h[0..n])) catch |e| return mapErr(e);

        self.max_pos = 0;
        for (items) |it| self.max_pos = @max(self.max_pos, it.pos);
        self.one_run = true;
        for (items, 0..) |it, t| {
            if (it.kv != items[0].kv or it.pos != items[0].pos + t) self.one_run = false;
        }
        const t0 = nowSecs();
        if (self.pass_events) |ev| ev[0].record(null) catch |e| return mapErr(e);
        try self.runBatch(n);
        if (self.pass_events) |ev| ev[1].record(null) catch |e| return mapErr(e);
        self.prof_launch_s += nowSecs() - t0;
        self.prof_passes += 1;
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
            try self.rmsNorm(self.xb.at(r * cfg.dim * f), self.hidden.at(idx * cfg.dim * f), self.out_norm.addr, 1);
        }
        // One pass over the output matrix for every requested row.
        try self.mm(self.output, self.xb.addr, self.logits_dev.addr, indices.len);
        const out = self.logits_h[0 .. indices.len * cfg.vocab];
        const t0 = nowSecs();
        cuda.copyFromDevice(std.mem.sliceAsBytes(out), self.logits_dev.addr) catch |e| return mapErr(e);
        self.prof_wait_s += nowSecs() - t0;
        if (self.pass_events) |ev| self.prof_gpu_ms += ev[0].elapsedMs(&ev[1]) catch 0;
        return out;
    }
};
