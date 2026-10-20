# Hardware and Kernels

## One binary, every instruction set level

The compute kernels (integer dot products for every quantization format, the batched matrix multiply, attention) are compiled several times, once per instruction set level of the processor family, and linked into the same program. When hk starts it asks the processor what it supports and uses the best level. Nothing depends on the machine that built the binary.

| Architecture | Levels, best first |
|:---|:---|
| x86-64 | `avx512` (AVX-512 with VNNI), `avx2vnni`, `avx2`, `generic` |
| AArch64 | `dotprod` (NEON with the dot product extension), `neon` |
| other | `generic` |

`hk hardware-profile` prints the level in use and the levels the binary contains. To force another level, for testing or to measure it, set `HK_KERNELS=avx2` (or any name above) before running.

All levels compute exactly the same integers. The dot product primitive has one definition, the weight operand is at most 127 in magnitude (or signed, handled separately), and each level implements it with whatever instruction is best: `vpdpbusd` on AVX-512 VNNI and AVX-VNNI, `vpmaddubsw` plus `vpmaddwd` on AVX2 (the 16 bit intermediate cannot overflow under that bound), `sdot` on ARM, and a portable widening multiply elsewhere. Two checks enforce this: `zig build test` runs the portable path on every platform, and the quant tests run every compiled-in level the machine supports against the same reference.

```bash
zig build test -Doptimize=ReleaseFast                    # every level this CPU can run
zig build test -Doptimize=ReleaseFast -Dcpu=x86_64_v3    # the AVX2 code in the library itself
zig build test -Doptimize=ReleaseFast -Dcpu=x86_64       # the portable code in the library itself
```

### What has been run

- **x86-64**: `avx512`, `avx2` and `generic` run here and give identical perplexity. `avx2vnni` needs a processor with AVX-VNNI but no AVX-512 (Intel Alder Lake and Raptor Lake desktop and mobile parts); it is built and the dispatcher selects it by feature bits, but no such processor was available.
- **AArch64**: the `dotprod` and `neon` levels compile for Linux and macOS, including the inline `sdot` and `tbl` instructions, and the portable code paths they share with x86 are tested. **They have not been run on ARM hardware.** The first run on an Apple Silicon Mac or a Graviton instance is the real test; `hk run model.hk "hi" --temp 0` should print the same text as with `HK_KERNELS=neon`, and `zig build test -Doptimize=ReleaseFast` should pass.
- **Windows and macOS on x86-64**: cross compiled, not run.

## Threads

The default thread count is the number of physical cores, not logical CPUs: a second hardware thread on a core adds contention to kernels that are limited by memory bandwidth and synchronization. On processors with two kinds of cores (Intel hybrid parts, ARM big.LITTLE) only the fast cores are counted, using `cpu_capacity` on Linux and the performance level count on macOS. Override with `--threads`.

## Memory

- Weights are never copied. The `.hk` file is memory mapped and the kernels read it in place, so a model costs address space, not memory; pages come in as layers are first used.
- Prompt processing unpacks one tile of 16 rows (8 on AVX2) at a time into a small per thread buffer, a few hundred KiB, instead of keeping a repacked copy of the model. llama.cpp keeps repacked copies of the weights.
- The KV cache is f16 and is allocated in segments of 256 positions as the context fills.

## GPU

A Vulkan compute backend runs the whole network on one device: embedding lookup, every layer, and the final norm in one command buffer per forward pass, with the key and value caches on the device. Enable it with `-ngl N` (any N above 0) or `HK_GPU=1`.

```
$ hk run model.hk "Hello" -ngl 99
[gpu] running on NVIDIA GeForce RTX 3050 6GB Laptop GPU (Vulkan)
```

- **Portable by construction.** Vulkan runs on NVIDIA, AMD, Intel and, through MoltenVK, Apple GPUs. The loader is opened at run time, so a machine without a driver starts normally and uses the CPU. Shaders are written in GLSL (`shaders/`), compiled to SPIR-V and checked in (`src/vk/spv/`), so building hk needs no Vulkan SDK; `shaders/build.sh` recompiles them with `glslc`.
- **What is verified.** On an NVIDIA RTX 3050 Laptop GPU: logits agree with the CPU engine to a cosine similarity of 0.9994 or better for every format the GPU decodes (the CPU quantizes activations to 8 bits and the GPU does not, so exact equality is not expected), unit tests compare the two engines on tiny models, and the command line and server end to end tests pass with `HK_GPU=1`. Nothing was run on an AMD, Intel or Apple GPU.
- **Fit.** The whole model, its KV cache and the activations must fit in device memory. hk checks the free memory the driver reports, shrinks the context window if that helps, and otherwise reports that the model does not fit and stays on the CPU. Offloading some layers and keeping the rest on the CPU is not implemented.
- **Serving.** Each server slot gets its own region of the device KV cache (`hk serve -ngl 1 --slots 4`).
- **Formats.** The formats marked in [Compatibility](Compatibility.md).
- **Speed.** Decode and prompt numbers are in [Benchmarks and Performance](Benchmarks-and-Performance.md). The kernels are not as tuned as the CPU ones: decode reaches roughly 60 to 70 percent of the card's memory bandwidth, and there is no matrix engine (cooperative matrix) path for prompts yet.

### Not supported

- **Metal.** Apple Silicon runs the CPU kernels (NEON, untested) or the Vulkan backend through MoltenVK (untested). A Metal backend would need an Apple machine to build and check.
- **CUDA.** `--backend cuda` runs the whole model on one NVIDIA GPU through the CUDA driver API (loaded at run time; no CUDA toolkit is needed to build or run hk). The device code is written in Zig and compiled to PTX by `zig build`; its matrix-vector kernel calls the same block decoders as the CPU, so every weight format the CPU runs works. It has been checked against the CPU on one RTX 3050 (llama, Qwen2, Qwen3; f32 and Q8_0). It is an early version: batches go through each layer together with a tiled GEMM and decode uses split-KV attention, but there is no kernel fusion, CUDA graph, Tensor Core or integer dot-product use yet. On a synthetic 136M-parameter Q8_0 model (random weights; RTX 3050 6 GB Laptop; 5 interleaved repeats) CUDA reached about 1,190 tok/s prompt processing and 360 tok/s generation against Vulkan's 5,720 and 406, so `--backend auto` still prefers Vulkan. These numbers say nothing about other engines or models.
- **ROCm.** No backend. Vulkan covers NVIDIA and AMD cards.
- **TPUs.** Google TPUs are programmed through XLA and have no runtime that a program like this can drive directly. There is no TPU backend and none is planned.
- **NPUs.** No backend.
