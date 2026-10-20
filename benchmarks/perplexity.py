#!/usr/bin/env python3
"""Perplexity of the same text, window protocol and weights in hk and llama.cpp (and optionally Transformers).

Accuracy is the half of a benchmark that speed numbers hide: an engine that is fast and subtly wrong is not faster.
All engines use llama.cpp's llama-perplexity protocol: the text is cut into windows of `ctx` tokens, only whole windows
are used, and only the second half of each window is scored.

    python3 perplexity.py --gguf Qwen3-0.6B-Q8_0.gguf --hk Qwen3-0.6B-Q8_0.hk \
        --tokenizer ~/hk-bench/models/Qwen__Qwen3-0.6B --text wiki.test.raw --ctx 512 --chunks 20

Token ids for hk are made with the Hugging Face tokenizer of the model; llama.cpp tokenizes the text itself. Models that
prepend a BOS token per window (Llama) are not supported by this script: hk-probe does not replace the first token of a
window the way llama-perplexity does, so use models without BOS (Qwen) for a like-for-like number.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HOME = Path.home() / "hk-bench"
LLAMA = HOME / "src" / "llama.cpp"
HK_BIN = Path(__file__).resolve().parent.parent / "zig-out" / "bin"


def run(cmd, env=None):
    e = dict(os.environ)
    e.update(env or {})
    p = subprocess.run([str(c) for c in cmd], env=e, capture_output=True, text=True)
    return p.stdout + p.stderr


def llama_ppl(build, gguf, text, ctx, chunks, ngl):
    exe = LLAMA / build / "bin" / "llama-perplexity"
    if not exe.exists():
        return None
    out = run([exe, "-m", gguf, "-f", text, "-c", ctx, "--chunks", chunks, "-ngl", ngl])
    m = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", out)
    return (float(m.group(1)), float(m.group(2))) if m else None


def hk_ppl(hk, ids, ctx, backend):
    out = run([HK_BIN / "hk-probe", hk, "--ppl", ids, ctx], env={"HK_BACKEND": backend})
    m = re.search(r"tokens (\d+) mean_nll ([\d.]+) ppl ([\d.]+)", out)
    return (float(m.group(3)), int(m.group(1))) if m else None


def hf_ppl(model_dir, ids_path, ctx, chunks, dtype):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    ids = np.fromfile(ids_path, dtype=np.uint32).astype(np.int64)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=getattr(torch, dtype)).eval()
    nll, count = 0.0, 0
    with torch.no_grad():
        for c in range(len(ids) // ctx):
            win = torch.tensor(ids[c * ctx : (c + 1) * ctx])[None]
            logits = model(win).logits[0].float()
            lp = torch.log_softmax(logits[:-1], dim=-1)
            tgt = win[0, 1:]
            sel = lp[torch.arange(len(tgt)), tgt]
            nll -= float(sel[ctx // 2 - 1 :].sum())
            count += len(sel[ctx // 2 - 1 :])
    return (float(np.exp(nll / count)), count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--hk", required=True)
    ap.add_argument("--tokenizer", required=True, help="Hugging Face directory of the same model (tokenizer, and weights for --hf)")
    ap.add_argument("--text", required=True)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=20)
    ap.add_argument("--hf", action="store_true", help="also run Transformers on the safetensors weights in --tokenizer")
    ap.add_argument("--hf-dtype", default="float32")
    ap.add_argument("--out", help="write the results as JSON")
    a = ap.parse_args()

    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ids = tok(Path(a.text).read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"]
    ids = ids[: a.ctx * a.chunks]
    ids_path = Path(tempfile.mkdtemp()) / "ids.u32"
    np.array(ids, dtype=np.uint32).tofile(ids_path)
    print(f"{len(ids)} tokens, {a.chunks} windows of {a.ctx}", file=sys.stderr)

    res = {}
    for name, build, ngl in [("llama.cpp cpu", "build-cpu", 0), ("llama.cpp vulkan", "build", 99), ("llama.cpp cuda", "build-cuda", 99)]:
        r = llama_ppl(build, a.gguf, a.text, a.ctx, a.chunks, ngl)
        if r:
            res[name] = {"ppl": r[0], "stderr": r[1]}
    for backend in ["cpu", "vulkan", "cuda"]:
        r = hk_ppl(a.hk, ids_path, a.ctx, backend)
        if r:
            res[f"hk {backend}"] = {"ppl": r[0], "scored_tokens": r[1]}
    if a.hf:
        r = hf_ppl(a.tokenizer, ids_path, a.ctx, a.chunks, a.hf_dtype)
        res[f"transformers cpu {a.hf_dtype}"] = {"ppl": r[0], "scored_tokens": r[1]}

    for k, v in res.items():
        extra = f" +/- {v['stderr']}" if "stderr" in v else ""
        print(f"{k:28s} PPL {v['ppl']:.4f}{extra}")
    if a.out:
        Path(a.out).write_text(json.dumps({"args": vars(a), "results": res}, indent=2))


if __name__ == "__main__":
    main()
