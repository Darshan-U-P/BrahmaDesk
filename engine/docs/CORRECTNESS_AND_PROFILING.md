# Correctness and profiling baseline

This document defines a repeatable local validation procedure for the native Qwen2 runtime. Run it on the same machine, model file, Release build, and power mode when comparing commits.

## Build and test

```powershell
git pull --ff-only origin feature/brahma-engine-core
cmake -S engine -B engine/build -A x64
cmake --build engine/build --config Release
ctest --test-dir engine/build -C Release --output-on-failure
```

## Tokenizer ID comparison

Brahma can dump IDs produced by its embedded GGUF vocabulary and BPE merges:

```powershell
$model = "$env:APPDATA\brahma-desk\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
$prompt = "Hello, world! How are you?"
.engine\build\Release\brahma-engine.exe $model --tokenize $prompt
```

Compare the exact IDs with an independent tokenizer implementation for the *same model and exact text*. A llama.cpp build may provide `llama-tokenize`; check its own `--help` for the supported prompt/output syntax. Save the exact input and both ID sequences when reporting a mismatch. Do not treat a successful encode/decode round-trip or synthetic unit test as proof of Qwen2.5 tokenizer parity: the current self-contained pre-tokenizer is approximate.

## Repeatable inference benchmark

Run each case three times. Keep the same prompt and generated-token cap; record the median prefill and decode rates. The first run may be affected by Windows file cache, antivirus, and CPU boost state.

```powershell
$model = "$env:APPDATA\brahma-desk\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
$prompt = "Explain photosynthesis in one simple sentence."
1..3 | ForEach-Object {
    "===== Run $_ ====="
    Measure-Command {
        .\engine\build\Release\brahma-engine.exe $model --generate $prompt 8
    } | Select-Object TotalSeconds
}
```

The executable writes generated text to stdout and timing diagnostics to stderr. Capture both streams if keeping a benchmark log.

## Reported metrics and interpretation

- **Runtime construction** measures the `Qwen2Runtime` constructor. It includes GGUF inspection, schema/tokenizer setup and cache allocation. Weight payloads and decoded tensors are loaded lazily, so it is not a full eager-load measurement.
- **Prompt prefill** measures the sequential token-by-token prompt forward pass. It is not batched prompt prefill.
- **Decode loop** measures greedy token selection and the required forward steps for generated tokens. Its tokens/s figure is approximate because the final selected token does not require another forward pass.
- **Projections** accumulates wall time inside linear/quantized projection calls.
- **RMSNorm** accumulates time inside normalization calls.
- **Attention core** measures the attention score/softmax/value aggregation region, excluding Q/K/V and output projections.

Operator totals are instrumentation estimates, not hardware-counter measurements. Projection and normalization timings include any lazy tensor fetch/decode triggered inside those calls. Embedding lookup, RoPE, SwiGLU, residuals, tokenizer work, allocation overhead outside these regions, and output decoding are not separately reported. Do not sum the operator totals and expect them to equal prefill plus decode.

## OpenMP / parallel backend

The executable prints whether it was compiled with OpenMP. CMake uses `find_package(OpenMP QUIET)`; a serial report means CMake did not find an OpenMP C++ target for that build. The OpenMP label confirms compile-time wiring only, not that parallel execution improves performance. Benchmark serial and OpenMP builds under identical conditions before drawing conclusions.

## Logit/reference comparison

The runtime API exposes `Qwen2Runtime::logits(token_ids)`, which returns next-token logits after replaying token IDs from position zero. The CLI can dump the full logits vector as CSV (index,value):

```powershell
$model = "$env:APPDATA\\brahma-desk\\models\\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
.\\engine\\build\\Release\\brahma-engine.exe $model --logits "1,2,3" > brahma_logits.csv
```

Replace `1,2,3` with token IDs from the same model/tokenizer. Compare with an independent runtime using identical token IDs, model file, context positions, and tensor semantics. Report max absolute error, relative error, top-1 token agreement, and the first differing token. The repository does not yet contain a trusted-reference logit fixture, and no numerical parity result has been verified in this update.

## Current known limitations

- The Qwen2-family pre-tokenizer is approximate; exact tokenizer parity remains unproven.
- No independent reference-logit comparison is included yet.
- The profiler is CPU wall-clock instrumentation and adds a small amount of timing overhead.
- Current prompt prefill is sequential, and the runtime remains CPU-only.
