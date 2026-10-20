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

### A. DONE, UNCOMMITTED: container integrity + crash-safe edits (plan §4, §5; milestone 2.4 items, landed early)

Details and numbers: CHANGELOG `[Unreleased]`, STATUS rows 4–5, wiki `Format-Specification` "Integrity and crash safety".
Working tree on top of `054964b`; nothing committed (owner has not asked).

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

### B. Next: HK 2.0 — Unified Runtime (plan §150, §151)

Milestone items: backend abstraction · CUDA engine · CUDA decode · CUDA prefill · CUDA attention · Vulkan under the common API ·
CPU backend finalized · unified memory planner · automatic device selection. Chain: graph/model abstraction → backend abstraction → CUDA
(prefill → decode → attention → fusion/graphs) → memory planner → partial offload → Vulkan optimization.

Nothing of B is started. Backend-interface work is **plan-mode** work (CLAUDE.md) and is blocked on owner decisions 2 and 3 below.
First concrete step once unblocked: define `Backend` (capabilities, buffer alloc, matmul-by-format, attention, norm, rope, kv ops, sync) in
`src/backend/`, put the CPU stack and the Vulkan engine behind it with **no behavior change**, prove parity with the existing engine tests,
then add the operator capability table (§136).

**Hardware reality for later milestones:** this machine is Linux x86-64, Ryzen 7 7445HS + RTX 3050 6 GB (NVIDIA driver present, Vulkan works).
Metal, ROCm/HIP, Intel XPU, DirectML, NPU, mobile, browser, ARM, multi-GPU and multi-node **cannot be run here**. Per the truthful-claims rule
those can be written and cross-compiled but must be recorded as "not run on hardware" until someone with the hardware runs them.

### Open decisions (owner)

1. ~~Seal cost model~~ — resolved above (revisitable).
2. **Which kernel stack is canonical?** Plan §9/§15/§17/§20 say "retain the earlier optimizations", but those live in the legacy stack
   (`tensor_ops.zig`), not the engine. Options: port them into the engine stack; or retire the legacy stack and route C API/Python through the engine kernels.
   *Blocks the backend interface design.*
3. **CUDA path:** adopt `src/cuda.zig` as the base for an engine backend, or restart from the Vulkan engine's structure? (Old commit messages
   disagree on how well the old CUDA code worked; nothing re-measured.) *Blocks CUDA engine.*
4. **Sandbox policy (§33):** `CodeSandbox` silently falls back from Docker to an unsandboxed subprocess. Make it fail closed? Likely yes; changes behavior.
5. ~~Version number~~ — superseded: no version is assigned until a milestone is complete; the CLI banner still says 1.2.0 while packages say 1.1.1 (fix when 2.0 is cut).
6. **`build.zig` `-Dcuda` default-on** for non-macOS builds although CUDA is off the engine path.
7. **Commit the integrity work?** Large uncommitted tree on branch `docs/context-system`. Suggested: commit on a new branch `feat/container-integrity`
   (not main), no tag, no push.

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
