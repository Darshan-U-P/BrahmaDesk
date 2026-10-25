# Brahma Engine Core

Independent native C++20 runtime workstream for Brahma Desk. It does not link to llama.cpp or GGML. The Qwen2 inference implementation uses upstream llama.cpp's public Qwen2 graph as a behavioral reference; no upstream runtime code is linked.

## Build and test (Windows / Visual Studio 2022)

```powershell
git pull --ff-only origin feature/brahma-engine-core
cmake -S engine -B engine/build -A x64
cmake --build engine/build --config Release
ctest --test-dir engine/build -C Release --output-on-failure
```

## CLI

```powershell
.\engine\build\Release\brahma-engine.exe "C:\path\to\model.gguf"
.\engine\build\Release\brahma-engine.exe "C:\path\to\model.gguf" --list-tensors
.\engine\build\Release\brahma-engine.exe "C:\path\to\model.gguf" --validate-qwen2
.\engine\build\Release\brahma-engine.exe "C:\path\to\model.gguf" --generate "Explain photosynthesis simply." 16
```

The generation command is a first native CPU inference path for Qwen2.5-Instruct GGUF files. It embeds the model's tokenizer vocabulary and merge ranks, uses greedy decoding, and prints generated text. Quote prompts containing spaces in PowerShell. First load may take time and memory because the current implementation decodes supported tensors to float32 and caches them in RAM.

## Current implementation

- [x] GGUF v2/v3 metadata and tensor descriptor parsing.
- [x] Named GGUF tensor loading, layout validation, and tensor registry.
- [x] Payload bounds, alignment, overlap, and serialized-size checks for the tensor types listed in `tensor_validator.cpp`.
- [x] Q4_K and Q6_K block dequantization.
- [x] CPU F32/F16, Q4_0/Q4_1, Q5_0/Q5_1, Q8_0, Q4_K and Q6_K tensor decoding for the initial inference path.
- [x] CPU matrix-vector projections, RMSNorm, SwiGLU, NeoX RoPE, grouped-query causal attention, residual paths, and per-layer KV cache.
- [x] Qwen2 schema validation for required tensors and metadata.
- [x] Initial GGUF byte-level BPE tokenizer and special-token handling.
- [x] Native greedy text-generation CLI for the Qwen2.5-Instruct chat template.
- [x] Synthetic unit tests for GGUF parsing, tensor operations, schema validation, transformer block behavior, and tokenizer behavior.

## Important limitations

This is an initial independent runtime, not yet a production-equivalent replacement for llama.cpp. The generation path currently decodes F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, and Q6_K tensors; it fails explicitly on other tensor types rather than silently interpreting them. The pre-tokenizer is a lightweight native approximation and has not yet been proven token-for-token identical to the official Qwen2 tokenizer. Quantization decoders and logits have not yet been compared numerically against a trusted independent runtime. No end-to-end generation run on the user's real Qwen2.5-3B file has been confirmed yet. Treat the first real-model run as a validation step, not a finished correctness claim.

The current implementation is CPU-only and greedy; sampling controls, robust context-window limits, performance kernels, GPU support, and Brahma Desk adapter integration remain future work.

## References

- [llama.cpp Qwen2 architecture graph](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen2.cpp) — behavioral reference only.
- [GGUF format definition](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/gguf.h) — file-format reference only.
