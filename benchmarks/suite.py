#!/usr/bin/env python3
"""Cross-engine benchmark: hk (CPU, Vulkan, CUDA) against llama.cpp and Hugging Face Transformers.

Only identical workloads are compared. Each entry names the weights it runs:

  gguf   a quantized GGUF file: hk (converted from it) and llama.cpp run exactly the same bytes.
  hf     a Hugging Face directory of safetensors: Transformers runs the original weights; hk and llama.cpp run
         f16 files made from those same weights, so all three multiply the same numbers.

Prompt processing and generation use synthetic token ids (speed does not depend on the text), the same lengths and
thread count for every engine. Engines run interleaved, one repeat each in turn, so a slow moment hits all of them.
Medians are reported with the min..max range; peak resident memory is the child's own (resource.getrusage), and peak
VRAM is sampled from nvidia-smi while it runs. Run it on an idle machine.

  benchmarks/suite.py --models name=gguf:/path/m.gguf name2=hf:/path/dir --engines hk-cpu,hk-vulkan,hk-cuda,llama-cpu,llama-vulkan,hf-cpu
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home() / "hk-bench"
HK_BIN = ROOT / "zig-out" / "bin"
LLAMA_VULKAN = HOME / "src" / "llama.cpp" / "build" / "bin"
LLAMA_CPU = HOME / "src" / "llama.cpp" / "build-cpu" / "bin"
LLAMA_CUDA = HOME / "src" / "llama.cpp" / "build-cuda" / "bin"
LLAMA_SRC = HOME / "src" / "llama.cpp"


def run_child(cmd, env=None, timeout=3600):
    """Runs `cmd`, returning (stdout+stderr, peak child RSS in MiB, peak VRAM in MiB)."""
    peak_vram = [0]
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
                if out:
                    peak_vram[0] = max(peak_vram[0], int(out[0]))
            except Exception:
                pass
            stop.wait(0.25)

    baseline = 0
    try:
        baseline = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0])
    except Exception:
        pass
    t = threading.Thread(target=poll, daemon=True)
    t.start()
    full_env = dict(os.environ)
    full_env.update(env or {})
    # Peak RSS of this child alone: run it through a tiny wrapper that reports its own rusage.
    wrapper = ("import resource,subprocess,sys;"
               "p=subprocess.run(sys.argv[1:],capture_output=True,text=True);"
               "sys.stdout.write(p.stdout);sys.stderr.write(p.stderr);"
               "sys.stderr.write('\\n__MAXRSS__ %d\\n'%resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss);"
               "sys.exit(p.returncode)")
    p = subprocess.run([sys.executable, "-c", wrapper] + [str(c) for c in cmd], capture_output=True, text=True,
                       env=full_env, timeout=timeout)
    stop.set()
    t.join(timeout=2)
    text = p.stdout + "\n" + p.stderr
    m = re.search(r"__MAXRSS__ (\d+)", text)
    rss = int(m.group(1)) / 1024 if m else 0.0
    return text, rss, max(0, peak_vram[0] - baseline)


# ---- engines -----------------------------------------------------------------------------------------------------

class Result:
    def __init__(self, ok, pp=None, tg=None, rss=0.0, vram=0.0, note=""):
        self.ok, self.pp, self.tg, self.rss, self.vram, self.note = ok, pp, tg, rss, vram, note


def hk_file(entry, kind):
    """Path of the .hk file this entry runs (converted once, next to the source)."""
    if entry["kind"] == "gguf":
        out = Path(entry["path"]).with_suffix(".hk")
        if not out.exists():
            subprocess.run([HK_BIN / "hk", "convert-gguf", entry["path"], out], check=True, capture_output=True)
        return out
    out = Path(entry["path"]) / "model.f16.hk"
    if not out.exists():
        subprocess.run([HK_BIN / "hk", "convert-hf", entry["path"], out], check=True, capture_output=True)
    return out


def bench_hk(entry, backend, args):
    f = hk_file(entry, "hk")
    text, rss, vram = run_child([HK_BIN / "hk-probe", f, "--bench", args.prompt, args.gen],
                                env={"HK_BACKEND": backend, "HK_THREADS": str(args.threads)})
    m = re.search(r"prefill \d+ tok in [\d.]+s = ([\d.]+) tok/s \| decode \d+ tok in [\d.]+s = ([\d.]+) tok/s", text)
    if not m:
        return Result(False, note=text.strip().splitlines()[-1][:120] if text.strip() else "no output")
    return Result(True, float(m.group(1)), float(m.group(2)), rss, vram)


def llama_gguf(entry):
    if entry["kind"] == "gguf":
        return Path(entry["path"])
    out = Path(entry["path"]) / "model.f16.gguf"
    if not out.exists():
        subprocess.run([sys.executable, LLAMA_SRC / "convert_hf_to_gguf.py", entry["path"], "--outfile", out,
                        "--outtype", "f16"], check=True, capture_output=True)
    return out


def bench_llama(entry, which, args):
    g = llama_gguf(entry)
    bindir, ngl = {"cpu": (LLAMA_CPU, 0), "vulkan": (LLAMA_VULKAN, 99), "cuda": (LLAMA_CUDA, 99)}[which]
    text, rss, vram = run_child([bindir / "llama-bench", "-m", g, "-p", args.prompt, "-n", args.gen, "-t", args.threads,
                                 "-ngl", ngl, "-r", 1, "-o", "json"])
    try:
        rows = json.loads(text[text.index("["):text.rindex("]") + 1])
        pp = next(r["avg_ts"] for r in rows if r.get("n_prompt") and not r.get("n_gen"))
        tg = next(r["avg_ts"] for r in rows if r.get("n_gen") and not r.get("n_prompt"))
    except Exception as e:
        return Result(False, note=f"parse: {e}: {text.strip()[-100:]}")
    return Result(True, pp, tg, rss, vram)


HF_SCRIPT = r"""
import sys, time, torch
from transformers import AutoModelForCausalLM
path, dev, prompt, gen, threads, dtype = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
torch.set_num_threads(threads)
dt = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[dtype]
model = AutoModelForCausalLM.from_pretrained(path, dtype=dt).to(dev).eval()
v = model.config.vocab_size
ids = torch.tensor([[1 + (i * 7919) % (v - 1) for i in range(prompt)]], device=dev)
def sync():
    if dev == "cuda": torch.cuda.synchronize()
with torch.inference_mode():
    model(ids[:, :16])  # warm up kernels
    sync(); t0 = time.perf_counter()
    out = model(ids, use_cache=True); sync(); t1 = time.perf_counter()
    past = out.past_key_values; tok = out.logits[:, -1:].argmax(-1)
    t2 = time.perf_counter()
    for _ in range(gen):
        o = model(tok, past_key_values=past, use_cache=True); past = o.past_key_values; tok = o.logits[:, -1:].argmax(-1)
    sync(); t3 = time.perf_counter()
print("HFRESULT %.3f %.3f" % (prompt / (t1 - t0), gen / (t3 - t2)))
"""


def bench_hf(entry, dev, args):
    if entry["kind"] != "hf":
        return Result(False, note="needs an hf: entry")
    text, rss, vram = run_child([sys.executable, "-c", HF_SCRIPT, entry["path"], dev, args.prompt, args.gen,
                                 args.threads, args.hf_dtype])
    m = re.search(r"HFRESULT ([\d.]+) ([\d.]+)", text)
    if not m:
        return Result(False, note=text.strip().splitlines()[-1][:120])
    return Result(True, float(m.group(1)), float(m.group(2)), rss, vram)


def run_engine(name, entry, args):
    fam, _, which = name.partition("-")
    if fam == "hk":
        return bench_hk(entry, which, args)
    if fam == "llama":
        return bench_llama(entry, which, args)
    if fam == "hf":
        return bench_hf(entry, which, args)
    return Result(False, note="unknown engine")


# ---- reporting ---------------------------------------------------------------------------------------------------

def fmt(xs):
    if not xs:
        return "-"
    return f"{statistics.median(xs):,.1f} ({min(xs):,.1f}..{max(xs):,.1f})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="name=gguf:PATH or name=hf:DIR")
    ap.add_argument("--engines", default="hk-cpu,hk-vulkan,hk-cuda,llama-cpu,llama-vulkan")
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--hf-dtype", default="bf16")
    ap.add_argument("--out", default=None, help="write markdown and JSON next to this path (no extension)")
    args = ap.parse_args()

    entries = {}
    for spec in args.models:
        name, _, rest = spec.partition("=")
        kind, _, path = rest.partition(":")
        entries[name] = {"kind": kind, "path": path}
    engines = args.engines.split(",")

    commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    llama_rev = subprocess.run(["git", "-C", LLAMA_SRC, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor())
    report = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "hk_commit": commit, "llama_cpp_commit": llama_rev,
              "cpu": cpu, "gpu": gpu, "threads": args.threads, "prompt": args.prompt, "gen": args.gen, "repeats": args.repeats, "results": {}}
    lines = [f"# Benchmark {report['date']}", "",
             f"- hk `{commit}`, llama.cpp `{llama_rev}`, CPU {cpu}, GPU {gpu}",
             f"- prompt {args.prompt} tokens, generate {args.gen}, {args.threads} threads, {args.repeats} interleaved repeats; median (min..max)", ""]
    for mname, entry in entries.items():
        samples = {e: [] for e in engines}
        notes = {}
        for r in range(args.repeats):
            for e in engines:
                res = run_engine(e, entry, args)
                if res.ok:
                    samples[e].append(res)
                else:
                    notes[e] = res.note
                print(f"[{mname}] repeat {r + 1}/{args.repeats} {e}: " + (f"pp {res.pp:.1f} tg {res.tg:.1f}" if res.ok else f"FAILED {res.note}"), flush=True)
        lines += [f"## {mname}  ({entry['kind']}: `{Path(entry['path']).name}`)", "",
                  "| engine | prompt tok/s | generation tok/s | peak RSS MiB | peak VRAM MiB |", "|:--|:--|:--|--:|--:|"]
        report["results"][mname] = {}
        for e in engines:
            s = samples[e]
            if not s:
                lines.append(f"| {e} | not run: {notes.get(e, '?')} | | | |")
                continue
            lines.append(f"| {e} | {fmt([x.pp for x in s])} | {fmt([x.tg for x in s])} | {max(x.rss for x in s):,.0f} | {max(x.vram for x in s):,.0f} |")
            report["results"][mname][e] = {"pp": [x.pp for x in s], "tg": [x.tg for x in s], "rss_mib": max(x.rss for x in s), "vram_mib": max(x.vram for x in s)}
        lines.append("")
    text = "\n".join(lines)
    print("\n" + text)
    if args.out:
        Path(args.out + ".md").write_text(text + "\n")
        Path(args.out + ".json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
