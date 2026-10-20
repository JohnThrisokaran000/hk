# Compatibility

What runs, in which format, on which hardware, and how well it is tested.

## Models

| Family | Status | Notes |
|:---|:---|:---|
| Llama 2 / 3, Mistral without sliding window, SmolLM2 (`llama`) | runs | rope scaling: none, linear, llama3, yarn |
| Qwen2, Qwen2.5 (`qwen2`) | runs | bias on the q, k, v projections |
| Qwen3 (`qwen3`) | runs | per head q and k norm |
| Gemma, Phi, Falcon, Mamba, mixture of experts, vision and audio models | not supported | the loader refuses them with a message naming the architecture |

The engine is a dense decoder with grouped query attention, RMS norm and a gated MLP (SiLU). Head width up to 256. Logits are checked against Hugging Face Transformers (cosine similarity 1.000000 on SmolLM2 f16 and Qwen3 BF16) and perplexity against llama.cpp.

## Weight formats

26 storage formats are read. Every one of them has a decoder, a CPU dot product kernel and a batch kernel; the GPUs decode the formats marked.

| Format | CPU | Vulkan | CUDA |
|:---|:---:|:---:|:---:|
| f32, f16, bf16 | yes | yes | yes |
| Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 | yes | yes | yes |
| Q2_K, Q3_K, Q4_K, Q5_K, Q6_K | yes | yes | yes |
| IQ4_NL, IQ4_XS | yes | yes | yes |
| IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S | yes | not yet | yes |
| TQ1_0, TQ2_0, MXFP4, NVFP4 | yes | not yet | yes |

CUDA runs every format the CPU runs because its kernels call the CPU's own block decoders; each of the 26 is checked against an f64 reference on the GPU (`tests/test_cuda.zig`). Only Q8_0 (and the dense types) have been timed on CUDA. A model that uses a format the chosen GPU cannot decode runs on the CPU, and hk says so (with `--backend vulkan|cuda` it fails instead). Quantized weights are decoded exactly: each decoder is tested against the reference `gguf` package on blocks of random data (`tests/test_quant.zig`).

## Tokenizers and chat templates

Byte level BPE and SentencePiece vocabularies, checked against llama.cpp on 301,948 tokens of real text. Chat templates run on a built-in Jinja subset, checked against Python's jinja2. A template that uses something outside the subset is reported, and the chat falls back to a plain transcript.

## Hardware

| Target | Status |
|:---|:---|
| x86-64 with AVX-512 and VNNI | tested (Ryzen 7 7445HS) |
| x86-64 with AVX2 | tested by forcing the level on the same machine (`HK_KERNELS=avx2`) |
| x86-64 without AVX2 | portable level, tested by forcing it; slow |
| x86-64 with AVX-VNNI and AVX2 only (Alder Lake and later) | built; has not run on such a processor |
| AArch64 (Apple Silicon, Graviton, Raspberry Pi 4/5) | built and cross compiled for Linux and macOS; **not run on ARM hardware** |
| Windows, macOS | cross compiled; not run |
| NVIDIA GPU through Vulkan | tested on one card (RTX 3050 Laptop) |
| AMD, Intel and Apple GPUs through Vulkan | not tested |
| CUDA | first engine (`--backend cuda`): all 26 CPU weight formats, llama/qwen2/qwen3, one NVIDIA GPU; verified against the CPU on one RTX 3050; early performance (see Hardware and Kernels) |
| Metal, ROCm, TPU | no backend |

See [Hardware and Kernels](Hardware-and-Kernels.md) for how the kernels are chosen and what has and has not been verified.

## Release contract

What a release promises, per the project plan (section 148). Every row says what was run.

### Backends

| Backend | Selected by | Verified | Known gaps |
|:---|:---|:---|:---|
| CPU | default; `--backend cpu` | unit tests at four instruction-set levels; logits byte-identical before/after the executor extraction in 70 comparisons; llama/qwen2/qwen3 | ARM and Windows not run |
| Vulkan | `-ngl N`, `--backend vulkan` | parity with the CPU for llama, wide MHA, qwen2 and qwen3 in f32 and Q8_0, multi-sequence slots and a 700-token context, on one RTX 3050 | one device; 15 of 26 formats; no partial offload |
| CUDA | `--backend cuda` | the same parity set as Vulkan, plus all 26 formats against an f64 reference, on the same card | one device; prompt processing about 4.8x slower than Vulkan on the synthetic benchmark; no fusion, CUDA graphs or Tensor Cores; heads up to 256 wide |

`--backend auto` (the default when a GPU flag is given) tries Vulkan, then CUDA, and keeps the CPU if neither fits, printing why. The preference order is fixed, not measured: Vulkan comes first because it is currently the faster of the two.

### Versions

| What | Value | Compatibility |
|:---|:---|:---|
| Container format | 1.0 (header `version_major` 1) | This release adds optional fields only: header flag bit 9 and the SHA-256 seal, a CRC-32 on new appendix records, and an edit journal beside the file. Reading the 1.1.1 source (not run) shows its reader checks only the magic and major version, ignores bit 9, never reads the checksum field and never verifies the appendix CRC, so it opens files written by this release. An older binary that *edits* a sealed file leaves the seal stale, which `hk verify` then reports. This release reads files without a seal. |
| C ABI | 2 (`HK_ABI_VERSION`, `hk_abi_version()`) | Version 1 is the unversioned ABI of 1.x. Incompatible changes in 2: Q4_K tensors written by the old quantizer must be re-quantized; the product functions report failure as NaN; sizes are validated. There is no struct-size negotiation or capability discovery yet. |
| Python package | follows the release | `metadata_set`, `append_record` and `rollback_appendix` need the native library |

### Known limitations

- Dense decoders only: llama, mistral without sliding window, qwen2, qwen3. No MoE, sliding window, state-space or multimodal models.
- One device per run; the whole model must fit on it (no partial offload). The context is halved to fit and the result is reported.
- Run and verified on one Linux x86-64 machine with one NVIDIA GPU. Windows, macOS, ARM, AMD and Intel GPUs, Metal, ROCm and multi-GPU have not been run.
- No signatures: the seal detects accidental damage and an unfinished edit, not tampering (anyone can recompute it).
- Sealing and verifying are O(file) in time (about 1.7 GiB/s here) at a constant 13 MiB of memory.
- No benchmark against llama.cpp was run for CUDA or for the synthetic model, and there was no real model on the test machine. Do not read the synthetic numbers as engine-versus-engine comparisons.

### Benchmark suite

Everything is reproducible from the repository:

```bash
zig build -Doptimize=ReleaseFast
zig-out/bin/hk-tiny-model m.gguf --q8 --arch qwen3 --dim 1024 --layers 12 --heads 8 --kv 2 --ffn 2816 --ctx 2048
zig-out/bin/hk convert-gguf m.gguf m.hk
HK_BACKEND=cpu|vulkan|cuda zig-out/bin/hk-probe m.hk --bench 512 64     # prefill and decode tok/s
HK_GPU_PROFILE=1  ... # host launch time and whole-pass GPU time;  HK_CUDA_EVENTS=1 ... # per-kernel GPU time
zig-out/bin/hk-compare --llama-bin <dir> --pair m.hk:m.gguf --threads 6 --repeats 5   # against llama.cpp, with a real model
```

Measured with that recipe (136M-parameter Qwen3-shaped Q8_0 model with random weights, RTX 3050 6 GB Laptop, prompt 512 and 64 generated tokens, 5 interleaved repeats in one session):

| Backend | Prompt tok/s (min..max) | Generation tok/s (min..max) |
|:---|:---|:---|
| CPU, 6 threads | 1,906 (1,686..2,019) | 91 (85..97) |
| Vulkan | 5,719 (5,490..5,830) | 406 (393..416) |
| CUDA | 1,190 (1,187..1,192) | 360 (356..362) |

The CPU numbers on this laptop swing about 2x between sessions; compare backends only within one run.
