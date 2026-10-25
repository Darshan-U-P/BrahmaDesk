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

The self-contained tokenizer implements Qwen2/GPT-2-family pre-tokenization with Unicode-aware letter/number and whitespace classification. It remains important to compare it against the exact reference tokenizer for the same GGUF file.

A PowerShell regression script compares Brahma and llama.cpp token IDs for English, digits, C++ punctuation, Kannada, Hindi, newlines, URLs, contractions, and trailing whitespace. From the repository root:

```powershell
$model = "$env:APPDATA\brahma-desk\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
$ref = "D:\all project\llama-b11552-bin-win-cuda-12.4-x64\llama-tokenize.exe"
.\engine\scripts\compare-tokenizer.ps1 -ModelPath $model -LlamaTokenizer $ref
```

The script uses `--no-bos` for llama.cpp so its output can be compared with Brahma's current no-BOS tokenizer API. A mismatch exits with code 1 and prints both ID sequences. Passing this test suite is evidence for the included strings only, not proof of every Unicode category, special token, or chat-template case. Keep reference comparisons in the regression set when changing tokenizer code.

## Repeatable inference benchmark

Use the checked-in script to run a fixed prompt multiple times, parse runtime diagnostics, write per-run CSV data, and report medians. Run from the repository root:

```powershell
$model = "$env:APPDATA\brahma-desk\models\Qwen2.5-3B-Instruct-Q4_K_M.gguf"
.\engine\scripts\benchmark-inference.ps1 -ModelPath $model -Runs 5 -MaxNewTokens 8 -OutputCsv .\brahma-benchmark.csv
```

Optional parameters include `-Prompt "your fixed prompt"` and `-BrahmaExe .\engine\build\Release\brahma-engine.exe`. Use an odd run count (for example, 5) for a conventional median. The CSV contains per-run construction, prefill, decode, projection, RMSNorm, attention, token counts, backend label, and wall-clock time. The script keeps generated text and diagnostics in temporary files and removes those files after each run. The first run may be affected by Windows file cache, antivirus, and CPU boost state; keep machine, power mode, model, prompt, and token cap constant across commits.

The executable writes generated text to stdout and timing diagnostics to stderr. The benchmark script captures both streams separately.

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

- Unicode-aware Qwen2/GPT-2-family pre-tokenization passed the nine selected model-backed parity cases on the user's machine; broader tokenizer coverage is still unproven.
- Full-vector independent reference-logit comparison is not yet available; the top-1 and top-N checks are partial diagnostics.
- The profiler is CPU wall-clock instrumentation and adds a small amount of timing overhead.
- Current prompt prefill is sequential, and the runtime remains CPU-only.


## Phase 1 status snapshot

This status is based on the repository's current implementation and the latest locally reported test results; it is not a claim that tests were run by CI during this commit.

- **GGUF parsing and tensor validation:** implemented; the seven CTest suites passed in the latest user-reported run.
- **Tokenizer parity:** the reference script passed 9/9 selected strings on the user's machine, including Kannada and Hindi. This covers those cases only.
- **Next-token sanity check:** for the one-token input ID sequence `[9707]` corresponding to `Hello`, Brahma and llama.cpp both selected token ID `358`. This is a top-1 agreement on one case, not full numerical parity.
- **Numerical logit parity:** incomplete. Brahma can dump the full vocabulary logits; the current llama.cpp server check returned only top-N log-probabilities. These are sufficient for a limited relative-gap diagnostic, not a full-vector error metric. The repository does not yet include a trusted raw-reference-logit fixture.
- **Model-load timing:** runtime construction is timed, but tensor payloads are loaded lazily; this is not full eager model-load time.
- **Operator profiling:** projection, RMSNorm and attention-core wall time are reported. Projection has dominated the user's previously reported CPU run. Other operators and allocation overhead are not individually timed.
- **OpenMP:** compile-time backend reporting is present. A controlled OpenMP-versus-serial speedup comparison remains to be performed.
- **Repeatable benchmarks:** `engine/scripts/benchmark-inference.ps1` runs repeated inference trials, exports per-run CSV metrics, and reports medians. Run it locally to establish a baseline; no benchmark result is asserted by this commit.
