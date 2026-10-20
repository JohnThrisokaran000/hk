# HK — Handoff

Volatile, session-to-session state. **Newest information goes at the top of "Now"; the log at the bottom is
append-only.** Stable orientation: [CONTEXT.md](../../CONTEXT.md). Plan vs. reality: [STATUS.md](STATUS.md).

Rules for editing this file: state what was *run* vs. *only read*; never write "done" for something not
verified; when a "Now" item is finished, move one line of it to the log and delete it from "Now".

---

## Now

### Owner direction (2026-10-20)

Do **not** ship a "3.0.0". Implement the whole of `MASTER_PLAN.md` and release **in the plan's order**: 2.0 → 2.1 → … → 2.7 → 3.0
(§150), built along the dependency chain in §151. 3.0 is the *last* milestone. Also: **keep as much as possible in Zig**
(Python stays a thin layer over the C ABI; no second implementation of anything Zig already owns).

### A. DONE, COMMITTED (`a1ee4eb`, branch `feat/container-integrity`): container integrity + crash-safe edits (plan §4, §5; milestone 2.4 items, landed early)

Details and numbers: CHANGELOG `[Unreleased]`, STATUS rows 4–5, wiki `Format-Specification` "Integrity and crash safety".
Committed on top of `054964b`; not pushed, not tagged.

**Run (2026-10-20):** `zig build test` 148/148 at Debug, ReleaseFast, `-Dcpu=x86_64_v3`, `-Dcpu=x86_64`; `zig build test-e2e -Doptimize=ReleaseFast`
24/24; `bash tests/bindings/run.sh` 8 suites 0 failed; `pytest tests` 97 passed / 3 skipped (missing `openai`, `jax`, `tkinter`) in a scratch
venv at `/tmp/claude-1000/.../scratchpad/venv` (CPU torch) — recreate it, it is not in the repo. Mutation-checked: 9 Zig guards (G1–G9) and 4 CLI guards (C1–C4) each turn
red when disabled. Measured on a 2.1 GiB container: `hk verify` 1.27 s / 13 MiB RSS, `hk metadata set` 2.50 s / 13 MiB RSS (warm cache, one machine).

**Fixed while finishing (were silent failures):** converters never sealed; sealing could read the whole file into RAM when `mmap` failed;
an edit of an already-corrupt sealed file re-sealed it; `sealFile` wrote the header twice (crash window); appendix append/rollback had no
journal; a torn journal blocked all edits; `hk verify` / `hk metadata set` / `hk rollback` exited 0 on failure; `hk verify` hashed through
the mapping (2 GiB RSS); `hk_appendix_append` cast an untrusted byte to an enum; Python `metadata_set` fell back to a patcher that left a stale seal
and could corrupt an appendix, and Python `append_record`/`rollback_appendix` read the whole file and rewrote it non-atomically.
Python now goes through the C ABI only and raises without the native library.

**Decision I made (revisit if you disagree):** Open decision "seal cost" resolved as *keep whole-file SHA-256, streamed, verify-before-edit*.
Rationale: correctness first, bounded RAM, measured ~1.65 GiB/s. Sectioned/Merkle hashing would make edits O(changed) but needs a format change — fits the
2.4 "v2 format" item. Edits on a 70 GB model will take minutes; that is the known cost.

**Not done / limits:** no Windows/macOS/ARM run (`std.Io` replaced the Windows-specific writer in sealing, unrun there); crash tests simulate the
post-crash state by hand, no kill-9/power-loss harness; `hk_verify_integrity` is wrapped in C++ and Python only (Go/Rust/C#/Java/TS lack it);
no `hk compact`/`hk fsck`; no signatures (an attacker can recompute the seal); `hk run` does not verify the seal at load (O(file) cost) —
decide whether `hk pull` should.

### A2. Toolchain moved to Zig 0.17 (branch `feat/hk-2.0-backends`)

Owner asked for 0.17. `zig` on PATH is still 0.16 (pacman); 0.17 is `~/.local/bin/zig17`. Repo builds and passes on 0.17: unit 148/148 (Debug, ReleaseFast, v3, baseline),
e2e 24/24 (now never cached), bindings 8/8 (needs a `zig`→0.17 shim on PATH), pytest 97/3. CI action pin changed to 0.17.0 but **CI was not run**.
**Stage 0 spike result: GO** — a Zig-authored Q8_0 GEMV (shared memory, `shfl.sync`, `bar.sync`) compiled with
`zig17 build-obj -target nvptx64-cuda -mcpu sm_86 -O ReleaseFast k.zig -fno-emit-bin -femit-asm=k.ptx`, loaded with `cuModuleLoadData`, ran on the RTX 3050
and matched a CPU reference (rel err 1.25e-7). Shared memory must be file-scope `var x: [N]T addrspace(.shared)`. Spike files were in the scratchpad (not kept).

### B. HK 2.0 — Unified Runtime: progress (branch `feat/hk-2.0-backends`, commits not pushed)

Plan: `~/.claude/plans/snug-foraging-quasar.md` (approved 2026-10-20). Checklist (§150): backend abstraction · CUDA engine · CUDA decode · CUDA prefill · CUDA attention · Vulkan under common API · CPU backend finalized · unified memory planner · automatic device selection.

| Stage | State | Evidence |
|:--|:--|:--|
| 0 Zig-authored CUDA kernels feasible? | **GO** | Zig 0.17 nvptx64: Q8_0 GEMV with shared memory + `shfl.sync` + `bar.sync` ran on the RTX 3050, rel err 1.25e-7 vs CPU |
| Toolchain → Zig 0.17 | done | `1f040ee`; unit/e2e/bindings/pytest green; CI action pin bumped, CI not run |
| 1a `backend.Device` seam | done | `2a07119`; Vulkan under it |
| 1b CPU executor extracted (`engine/cpu_backend.zig`) | done | `01d80b0`; logits byte-identical old vs new in 70/70 comparisons; **speed not re-measured** |
| 1 capability table (`backend/caps.zig`) | done (formats only) | per-operator table (§136) not done |
| 2 planner + `--backend` | done | `fb6c2c1`; 159/159 unit ×4 configs, e2e 26/26, guards mutation-checked |
| 3 CUDA engine | **done (first version)** | `src/cuda/engine.zig` + `src/cuda_device.zig`; parity vs CPU for llama/qwen2/qwen3 × f32/Q8_0, 26 GEMV/GEMM formats, multi-slot, long context; 20+ mutations red |
| 4 CUDA prefill | **done, slow** | tiled shared-memory GEMM; prefill 1,190 tok/s vs Vulkan 5,719 (synthetic 136M Q8_0, 5 interleaved repeats) |
| 4b CUDA decode | **done** | warp-per-row GEMV + split-KV attention; decode 360 tok/s vs Vulkan 406 |
| 5 CUDA attention | **done** | online-softmax tiles, warp-cooperative key reads, split-KV for decode |
| 6 CUDA fusion + graphs | **evaluated, not done** | host launch cost measured ~0.4 ms/pass of ~7 ms; GEMV was 72 % of decode, so fixing memory traffic came first (§152). Revisit after Tensor Cores/integer dot products |
| 7 Vulkan under common API | partly (1a); parity harness not shared yet | |
| 8 CPU finalized + kernel-stack unification (owner decision 2) | **done** | `quantization.zig` and `tensor_ops.zig` deleted; encoders → `quant/encode.zig`, research formats → `quant/research.zig`, C ABI products → `engine/abi_ops.zig`. Found and fixed: private Q4_K layout, broken Q3_K encoder, silent block skipping in `dequantizeToF32`. The "legacy optimizations worth porting" turned out to be nothing: the legacy code was single-threaded (its thread pool ran inline) and slower |
| 9 Release gate: §148 compatibility contract | **contract written, release NOT cut** | `docs/wiki/Compatibility.md` "Release contract"; `hk_abi_version()` = 2. No version bump, tag, push or publish was done (owner approves those) |

**2.0 readiness (my honest assessment, branch `feat/hk-2.0-backends`, 17 commits ahead of `docs/context-system`):** every §150 item exists at a first-version level and is verified on this machine only. Gaps the owner should weigh before calling it 2.0: the memory planner is v0 (fit-or-shrink the context; no partial offload or cross-device placement); automatic selection is a fixed order (Vulkan, then CUDA), not measured; CUDA prompt processing is ~4.8x slower than Vulkan (decode within ~12%); CUDA fusion/graphs were evaluated and deliberately not done; nothing was run on a second machine, on Windows/macOS/ARM, or with a real model (none on disk) or against llama.cpp. The CPU executor is a peer struct rather than a union variant of `Device`.

**Release-cut checklist (do only when the owner says so):** (1) set the version to 2.0.0 in `build.zig.zon`, `pyproject.toml`, `python/hk/__init__.py`, `bindings/{js/package.json,rust/Cargo.toml,java/pom.xml,csharp/Hk.csproj}`, the CLI banner in `tools/hk_cli.zig` (`printUsage`, currently "1.2.0"), `CITATION.cff` if it carries one, README install lines; (2) turn `## [Unreleased]` in `CHANGELOG.md` into `## [2.0.0] - <date>` and add a short "upgrading" note (re-quantize Q4_K tensors written by the old Python/ABI quantizer); (3) update `SECURITY.md` (still lists 1.0.x); (4) run the whole matrix again, plus CI, which has **never run** against Zig 0.17 (the action pin was bumped, unverified); (5) merge `feat/hk-2.0-backends` through `feat/container-integrity` and `docs/context-system` into `main`; (6) tag `v2.0.0` and let the release workflow publish PyPI `hknt` / npm `hkntf`; (7) check the wiki sync.

**Next arcs** (if the owner wants to keep going): 2.1 items that *can* be run here — CUDA Tensor Cores via `mma.sync` PTX written in Zig (sm_86), 2:4 sparse Tensor Cores (`mma.sp`, sm_80+), Vulkan cooperative matrices; items that cannot — ROCm/HIP, Metal, Intel XPU, DirectML, mobile, NPU — need that hardware. Cheaper CUDA wins first (§152): prefill GEMM register blocking and vectorized loads, integer dot products (`dp4a`) for quantized weights, then fusion/graphs.

**Bugs of mine found by the process (for the record):** an edit script that died halfway and left `engine.zig` half-edited while I reported it done (assert every anchor); a benchmark that stopped the prefill clock before the async GPU finished, which inflated prefill and deflated decode (end prefill with the first logits); `Engine.init` leaving defaulted fields `undefined`, visible only in Debug (keep Debug in the matrix); equivalent mutants ("every split covers the whole context", "never split") that look like test gaps but are not.

**Gotchas learned this session:** shell `ls` alias is broken (use `find`); `/usr/bin/time` is not installed (use Python `resource`); zsh does not word-split `$var` (use `${=var}`); `pgrep -f X` inside a waiting loop matches the loop itself — poll a log file instead; Zig caches test *runs* (e2e is now `has_side_effects`); no real model on disk, so no speed claims; tiny models via `hk-tiny-model [--q8]`; parity harness lives in the scratchpad (`parity.py`, old-vs-new `hk-probe` over ISA levels × threads × prompts) — recreate from this description if needed: build the previous commit in a `git worktree`, run both `hk-probe`s with `HK_KERNELS`/`HK_THREADS`, compare raw f32 output bytes.

### Open decisions (owner)

1. ~~Seal cost model~~ — resolved above (revisitable).
2. ~~**Which kernel stack is canonical?**~~ **DECIDED 2026-10-20: the engine stack.** Port the legacy optimizations worth keeping into `src/engine` + `src/quant` + `src/kernels`, then route the C API/Python through the engine kernels and retire `tensor_ops.zig`/`quantization.zig`. (Original question follows.) Plan §9/§15/§17/§20 say "retain the earlier optimizations", but those live in the legacy stack
   (`tensor_ops.zig`), not the engine. Options: port them into the engine stack; or retire the legacy stack and route C API/Python through the engine kernels.
   *Blocks the backend interface design.*
3. ~~**CUDA path**~~ **DECIDED 2026-10-20: adopt `src/cuda.zig` as the base** (validate it first — it is unmeasured and wired into nothing). (Original question follows.) Adopt `src/cuda.zig` as the base for an engine backend, or restart from the Vulkan engine's structure? (Old commit messages
   disagree on how well the old CUDA code worked; nothing re-measured.) *Blocks CUDA engine.*
4. **Sandbox policy (§33):** `CodeSandbox` silently falls back from Docker to an unsandboxed subprocess. Make it fail closed? Likely yes; changes behavior.
5. ~~Version number~~ — superseded: no version is assigned until a milestone is complete; the CLI banner still says 1.2.0 while packages say 1.1.1 (fix when 2.0 is cut).
6. **`build.zig` `-Dcuda` default-on** for non-macOS builds although CUDA is off the engine path.
7. ~~Commit the integrity work?~~ **DONE 2026-10-20:** commit `a1ee4eb` on branch `feat/container-integrity` (off `docs/context-system`). No tag, no push, no release.

### Verified facts worth not re-deriving

- Engine architectures: `llama` (+`mistral` alias), `qwen2`, `qwen3` (`src/engine/config.zig`).
- ISA levels per arch: `src/kernels/variants.zig`; override with `HK_KERNELS`.
- Server routes: `/health`, `/v1/health`, `/v1/models`, `/models`, `/metrics`, `/v1/chat/completions`, `/v1/completions` (+ unprefixed aliases).
- `hk-cuda-tests` passes here in ~2 s; unknown whether it hit the GPU.
- Local hardware: AMD Ryzen 7 7445HS (6C), NVIDIA RTX 3050 6GB Laptop (driver present), Arch Linux, Zig 0.16.0 at `/usr/bin/zig`.
- Shell quirk: `ls` is aliased to a tool that rejects directory args as `--icons` values — use `find`/`git ls-files`.

---

## Log (append-only, newest last)

### 2026-10-06 — context system created
- Created `CONTEXT.md`, `AGENTS.md`, `CLAUDE.md`, `docs/plan/{MASTER_PLAN,STATUS,HANDOFF}.md`, `docs/graph/*`.
- Stored the owner's 160-section plan verbatim as `MASTER_PLAN.md`.
- Surveyed repo at `43cf891` + dirty tree; ran `zig build test` (138/138). Did **not** run Python tests, e2e tests, or GPU benchmarks.
- Findings recorded in STATUS: two kernel stacks; "137+ architectures" is a Python alias registry (79 canonical, 169 aliases) while the engine runs 3 families;
  `NativeSandbox` does not enforce its timeout and is unused; `CodeSandbox` silently degrades Docker→subprocess; CLI/packaging version mismatch; stale `SECURITY.md`.
- Graph: `docs/graph/gen_graph.py --check` passes (79 curated nodes, 167 files, 368 import edges, 0 unowned files) and was mutation-tested (5 breakages caught).
  Mermaid diagrams: rendered with mermaid-cli 12.0.0 and viewed. First render showed an unreadable hairball and a 4,389px strip, so the generator now
  splits the dependency diagram into 2a (built) / 2b (planned), lays systems out side by side, and shows only coupling >= 2 imports. All 5 render.
  Still not run: the `jq` recipes (no jq installed) — equivalent Python queries were.
- Added a pointer memory (`hk-context-system`) to the owner's Claude memory index.
- No source code was modified. Nothing committed. The pre-existing uncommitted integrity/transaction work (item A) is untouched.

### 2026-10-20 — container integrity finished; graphify added; direction reset
- Finished item A (see Now). Found and fixed the silent failures listed there; added 9+5 tests, mutation-checked 13 guards, measured cost/RSS.
- Owner redirected: no 3.0.0 now; follow the §150 milestone order and the §151 chain; keep as much as possible in Zig. Removed the Python seal/journal mirror I had started writing and routed Python through the C ABI instead.
- Installed **graphify** (`pipx install graphifyy` 0.9.92; command `graphify`). Built the code graph with `graphify extract . --code-only` (local AST, no API key): 4,848 nodes, 9,617 edges, Zig/Python/C#/Rust/TS/Java included. `graphify claude install` appended a "graphify" section to `CLAUDE.md` and wrote PreToolUse hooks to `.claude/settings.json` (machine-specific absolute path — excluded via `.git/info/exclude`; re-run `graphify claude install` on another machine). `graphify-out/` is git-ignored. Not installed: git hooks (`graphify hook install`) — they would run on every commit. Community names are placeholders (naming needs an LLM backend). Refresh with `graphify update .`. `docs/graph/` (curated plan/status graph) stays the source for *meaning*; graphify is the source for *code structure*; both pass their checks.
- Not run: Windows/macOS/ARM, GPU benchmarks, anything in section B.
