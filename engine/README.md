# Brahma Engine Core

Independent native C++20 runtime workstream for Brahma Desk. It does not link to llama.cpp or GGML.

## Build and test (Windows / Visual Studio 2022)

```powershell
cmake -S engine -B engine/build -A x64
cmake --build engine/build --config Release
ctest --test-dir engine/build -C Release --output-on-failure
.\engine\build\Release\brahma-engine.exe "C:\path\to\model.gguf"
```

## Current implementation status

- [x] GGUF v2/v3 metadata and tensor descriptor parsing.
- [x] Tensor element-count overflow checks and payload byte sizing for F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K and basic integer/f64 types.
- [x] Tensor payload bounds, alignment, and overlap validation.
- [x] Synthetic tests for Q4_K block sizing, truncated payload rejection and unknown-type rejection.
- [x] Initial Q4_K block/tensor dequantization implementation with deterministic known-value tests (needs Windows CI and real-model numerical validation).\n- [ ] Q6_K/Q5_K dequantization and broader quantization coverage.
- [x] Initial CPU float32 matrix multiplication and RMSNorm with unit tests.
- [x] Initial RoPE (interleaved and NeoX layouts) and single-sequence causal attention primitives with unit tests.
- [ ] Qwen2 feed-forward and complete transformer forward pass.
- [ ] Tokenizer and autoregressive generation.
- [ ] CUDA kernels and RTX 3050 Ti benchmark.
- [ ] Brahma Desk native-runtime adapter and accurate health status.

**Important:** current executable validates model layout only. It does not yet decode tensor values or generate tokens. The quantized payload validator intentionally fails closed on tensor types whose block layout has not been added.

## GGUF format reference

The parser is an independent implementation informed by the public format definitions in [gguf.h](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/gguf.h) and [gguf.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/gguf.cpp). These references are used for file-format compatibility only; there is no llama.cpp runtime dependency.
