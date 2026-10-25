# Brahma Engine Core

This project contains both an independent native C++20 runtime workstream and a llama.cpp-backed `brahma-engine` executable. The native runtime does not link to llama.cpp or GGML; its Qwen2 inference implementation uses upstream llama.cpp's public Qwen2 graph as a behavioral reference. The CLI below describes the llama.cpp-backed executable.

## Build and test (Windows / Visual Studio 2022)

```powershell
git pull --ff-only origin feature/brahma-engine-core
cmake -S engine -B engine/build -A x64
cmake --build engine/build --config Release
ctest --test-dir engine/build -C Release --output-on-failure
```

## CLI

```powershell
.\engine\build\bin\Release\brahma-engine.exe --model "C:\path\to\model.gguf" --prompt "Explain photosynthesis simply." --max-tokens 64 --ctx-size 2048 --batch-size 512 --threads 8 --gpu-layers 0 --temperature 0 --top-k 40 --top-p 0.95

# Existing positional form remains supported:
.\engine\build\bin\Release\brahma-engine.exe "C:\path\to\model.gguf" "Explain photosynthesis simply." 64
```

The `brahma-engine` executable uses the llama.cpp backend. `--model` is required; the other options have defaults shown by `--help`. Temperature zero selects greedy decoding; positive temperatures enable Top-K/Top-P sampling. `--threads` leaves llama.cpp's backend default unchanged unless specified. GPU layer offload requires a CUDA-enabled llama.cpp build and a correctly configured CUDA toolchain; setting `--gpu-layers` alone does not enable CUDA. Quote prompts containing spaces in PowerShell.

## Current implementation

- [x] GGUF v2/v3 metadata and tensor descriptor parsing.
- [x] Named GGUF tensor loading, layout validation, and tensor registry.
- [x] Payload bounds, alignment, overlap, and serialized-size checks for the tensor types listed in `tensor_validator.cpp`.
- [x] Q4_K and Q6_K block dequantization.
- [x] Fused Q4_K/Q6_K block dot products that avoid materializing temporary float blocks during quantized projections, with equivalence tests against dequantize-then-dot.
- [x] CPU F32/F16, Q4_0/Q4_1, Q5_0/Q5_1, Q8_0, Q4_K and Q6_K tensor decoding for the initial inference path.
- [x] CPU matrix-vector projections, RMSNorm, SwiGLU, NeoX RoPE, grouped-query causal attention, residual paths, and per-layer KV cache.
- [x] Qwen2 schema validation for required tensors and metadata.
- [x] Initial GGUF byte-level BPE tokenizer and special-token handling.
- [x] Native greedy text-generation CLI for the Qwen2.5-Instruct chat template.
- [x] Synthetic unit tests for GGUF parsing, tensor operations, schema validation, transformer block behavior, and tokenizer behavior.

## Important limitations

This is an initial independent runtime, not yet a production-equivalent replacement for llama.cpp. The generation path currently decodes F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, and Q6_K tensors; it fails explicitly on other tensor types rather than silently interpreting them. The pre-tokenizer is a lightweight native approximation and has not yet been proven token-for-token identical to the official Qwen2 tokenizer. Quantization decoders and logits have not yet been compared numerically against a trusted independent runtime. No end-to-end generation run on the user's real Qwen2.5-3B file has been confirmed yet. Treat the first real-model run as a validation step, not a finished correctness claim.

The current implementation is CPU-only and greedy. Linear projections use float accumulation and optional OpenMP row-parallel execution when CMake detects OpenMP. Q4_K/Q6_K projections now use fused block dot products without expanding each block into a temporary float array, and these kernels have synthetic equivalence tests. The runtime still keeps loaded quantized tensor payloads in host RAM; this is not yet memory-mapped SSD execution. GPU execution, CPU/GPU layer offload, automatic backend selection, robust context-window limits, and Brahma Desk adapter integration remain future work. Do not interpret the presence of these roadmap items as implemented hardware modes.

## Hardware and weight-placement roadmap

The execution design will separate **compute device** from **weight placement**:
- CPU compute with quantized weights in system RAM (current backend).
- GPU compute with quantized weights resident in VRAM (future CUDA backend).
- Hybrid CPU/GPU layer offload with explicit VRAM budgets (future).
- Memory-mapped GGUF weights backed by SSD and the OS page cache (future; not the same as repeatedly reading individual tensors from disk).
- Automatic mode selection based on detected hardware and memory budgets (future).

The current runtime is CPU-only. The next major performance step is a native SIMD quantized CPU kernel and then a real CUDA quantized backend; a CLI flag alone would not constitute GPU support.

## References

- [llama.cpp Qwen2 architecture graph](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen2.cpp) — behavioral reference only.
- [GGUF format definition](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/gguf.h) — file-format reference only.


## Correctness and profiling

See [Correctness and profiling baseline](docs/CORRECTNESS_AND_PROFILING.md) for tokenizer-ID comparison, logit export, repeatable benchmark commands, operator timing definitions, and OpenMP verification.
