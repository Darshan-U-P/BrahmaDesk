# Brahma Engine Core — Gap Analysis vs. Official llama.cpp

**Research date:** 2026-10-25  
**Comparison basis:** official `ggml-org/llama.cpp` repository (README, server documentation, build guide, Qwen2 graph, performance notes, speculative decoding, function calling and multimodal documentation) compared with the current `feature/brahma-engine-core` branch of BrahmaDesk.

This is an engineering gap analysis, not a claim of feature or performance parity. The reference project's capabilities continue to evolve. This report summarizes the material differences relevant to Brahma Engine Core's current goal: an independent native C++ inference runtime.

## Executive summary

Brahma Engine Core has proven a small end-to-end native CPU inference path for one Qwen2.5-3B-Instruct GGUF model. The user's last supplied Windows run reported 7/7 CTest suites passing and 14.22 seconds total for an eight-token generation. That timing includes model construction/loading, prompt prefill and token generation; it must not be interpreted as steady-state decode throughput. The runtime prints phase timings on a newer commit, but those diagnostics have not yet been locally verified.

The official llama.cpp project is much more than an inference loop. Its repository documents optimized CPU kernels across architecture-specific instruction sets, multiple accelerator backends, CPU/GPU hybrid execution, model loading policies, a reusable C API, a feature-rich server, sampling/grammar controls, speculative decoding and multimodal execution. Brahma currently implements only a small subset.

## Scope and source links

Official references:
- Repository and capability overview: https://github.com/ggml-org/llama.cpp
- Server capabilities and CLI options: https://github.com/ggml-org/llama.cpp/tree/master/tools/server
- Backend/build support: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- Token generation performance guidance: https://github.com/ggml-org/llama.cpp/blob/master/docs/development/token_generation_performance_tips.md
- Qwen2 graph: https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen2.cpp
- Public C API: https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h
- Speculative decoding: https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- Function calling: https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md
- Multimodal support: https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
- Constrained generation grammar guide: https://github.com/ggml-org/llama.cpp/tree/master/grammars

Brahma files reviewed on `feature/brahma-engine-core`:
- `engine/src/qwen2_runtime.cpp`, `engine/include/brahma/qwen2_runtime.hpp`
- `engine/src/quantization.cpp`, `engine/include/brahma/quantization.hpp`
- `engine/src/qwen2_tokenizer.cpp`
- `engine/src/tensor_loader.cpp`, `engine/src/tensor_registry.cpp`, `engine/src/tensor_validator.cpp`
- `engine/src/main.cpp`, `engine/CMakeLists.txt`, `engine/README.md`

## Gap matrix

| Area | Official llama.cpp | Brahma current state | Gap / priority |
|---|---|---|---|
| Model architecture coverage | Many LLM architectures plus vision/language models | Qwen2-family-like inference path; schema explicitly Qwen2-oriented | **P0/P3:** stabilize Qwen2 correctness first; add architecture registry and more architectures later |
| Numerical correctness | Mature graph/runtime with model-specific operators and cross-backend implementations | Synthetic unit tests and real-model smoke test; no independent logit parity test | **P0:** compare logits and tokenization against a trusted reference on fixed inputs |
| CPU kernels | Architecture-specific optimized CPU implementations (e.g. AVX/AVX2/AVX-512, ARM NEON and other platform-specific paths depending on CPU) | Scalar C++ kernel loops with optional OpenMP row parallelism; fused Q4_K/Q6_K block dot products | **P0:** add CPU feature detection, SIMD dispatch, tuned quantized dot kernels and reproducible microbenchmarks |
| Quantized formats | Broad low-bit integer and importance-aware quantization formats, with per-kernel support varying by backend | Inference decodes F32/F16, Q4_0/Q4_1, Q5_0/Q5_1, Q8_0, Q4_K/Q6_K; fused projection path only for Q4_K/Q6_K | **P0:** format coverage and direct kernels are incomplete; add format support incrementally with numerical tests |
| Weight memory footprint | Quantized weights can remain quantized in backend buffers | Q4_K/Q6_K projections avoid whole-matrix float expansion, but other supported formats may decode to float32 and raw payloads are retained in RAM | **P0:** reduce duplicate resident representations; quantized direct kernels for all high-impact formats |
| Model loading | Documented auto/mmap/locking/DirectIO modes and lazy on-demand tensor reads (feature availability depends on platform/build) | Metadata-only index then per-tensor file reads cached as raw payloads; not memory-mapped | **P1:** introduce safe memory-mapped GGUF view and explicit resident/lazy policies |
| CPU+RAM mode | Optimized CPU execution; threads, affinity/priority options | CPU mode with optional OpenMP projection parallelism; no tunable thread count or affinity interface | **P0:** add controlled thread settings and benchmark thread sweeps |
| GPU/VRAM mode | CUDA plus other backends including HIP, Metal, Vulkan, SYCL and additional platform/vendor backends | No GPU compute backend | **P0/P1:** CUDA first for user's NVIDIA hardware; device abstraction only after a useful kernel path exists |
| CPU+GPU layer offload | GPU-layer count and configurable device placement/splits; can partially offload when VRAM is limited | No device buffers, layer placement or cross-device transfers | **P1:** implement layer-level placement, VRAM budget and explicit transfer boundaries after CUDA baseline |
| Multi-GPU | Backend-dependent device selection/splitting | Not implemented | **P3:** unnecessary for present laptop target |
| SSD-backed/lazy modes | mmap, mlock variants, lazy tensor-row reading and DirectIO where supported | Per-tensor reads into owned vectors; not OS memory mapping or true lazy row reads for tensor files | **P1:** mmap and page-fault-aware access; avoid repeated storage paging as an inference strategy |
| Graph execution / operator scheduler | Computation graph with backend placement, graph planning and buffer allocation | Explicit C++ single-token operations and vectors allocated throughout each layer step | **P0:** reuse workspaces/buffers; graph or operation planning for future backend scheduling |
| Prompt prefill | Batched prompt evaluation, independent batch/thread controls and attention optimizations | Replays prompt one token at a time through the same single-token step | **P0:** implement multi-token prompt prefill; this is a major performance and latency gap |
| Attention | Optimized attention implementations and Flash Attention options depending on backend/model | Manual per-head attention loops; no Flash Attention | **P0/P1:** optimize attention and KV layout; Flash Attention after GPU backend exists |
| KV cache | Configurable K/V cache types; KV offload; context and cache policies | Per-layer append-only FP32 vectors; no cache type, trimming, reuse or offload controls | **P1:** configurable cache dtype/layout, context guard and memory accounting |
| Output head | Optimized projection and optional output-head offload | Vocabulary-wide projection computed by the CPU path; no special optimized output path | **P0:** specialized quantized output-head kernel; profile as likely decode bottleneck |
| Tokenizer | Model-specific tokenizer behavior and multiple vocab/tokenizer types | Lightweight approximate Qwen2 byte-BPE pre-tokenizer; no proof of exact official token-ID parity | **P0:** exact tokenization parity tests for Unicode, whitespace, special tokens and realistic prompts |
| Sampling | Multiple samplers and tunable generation options | Greedy only | **P1:** temperature, top-k, top-p, repetition/frequency penalties, seed and stop sequences |
| Context/RoPE | Context-size controls and RoPE scaling options (model/back-end dependent) | Reads some Qwen2 config; no robust runtime context bound or extensible RoPE-scaling configuration | **P0:** prevent out-of-range/memory-unbounded contexts; then add supported scaling variants |
| Speculative decoding | Draft-model and additional speculative algorithms documented | Not implemented | **P2:** after correctness, continuous decode and prefill are stable |
| C/C++ library API | Public `libllama` C API for embedding | C++ `Qwen2Runtime` API exists, but no stable C ABI or general backend API | **P1:** stabilize model/context/session API; optionally provide C ABI for app integrations |
| HTTP server | OpenAI-compatible APIs, additional API compatibilities, parallel requests and monitoring | CLI only; no server | **P1:** a streaming local API after inference API stabilizes |
| Streaming token output | Supported by CLI/server application surfaces | Generation returns one completed string; CLI prints after it finishes | **P1:** callback/iterator-based token streaming |
| Concurrency / serving | Multi-user parallel decoding, continuous batching, request scheduling | One synchronous request and one cache set per runtime instance | **P2:** request/session isolation, batch scheduling and cancellation |
| Tool/function calling | Template-aware and generic tool calling support in serving stack | Not implemented | **P2:** add after chat-template and streaming interfaces are reliable |
| Constrained JSON/grammar output | Grammar-constrained decoding and JSON output options | Not implemented | **P2:** grammar/constrained decoding |
| Multimodal | Separate multimodal subsystem supports images/audio/video for supported architectures and models | Text-only Qwen2 path | **P3:** separate multimodal encoder/projector workstream; not a small feature addition |
| Model download/discovery | CLI can fetch supported HF models and manage cache in its ecosystem | User supplies a local GGUF file path | **P2:** model manager and integrity validation |
| Observability | Logs and performance controls for model load, backend/device offload and runtime phases | Recent CLI work prints prefill/decode phase timings; no complete profiling or memory/device telemetry | **P0:** log backend selection, load time, token rates, peak/estimated memory and unsupported ops |
| Benchmarking | Dedicated benchmark tooling and performance troubleshooting docs | User's wall-clock manual benchmark; fused kernel test | **P0:** separate kernel, load, prompt-prefill and decode benchmarks against same prompts/settings |
| Error handling / resource limits | More extensive operational controls in mature library/server | Initial validation exists; context/memory boundaries and compatibility are incomplete | **P0:** bounds, graceful allocation failures, malformed-model tests and stress tests |
| Packaging/install | CLI/server builds, backend build instructions and distribution options | CMake Windows project and CLI test executable | **P2:** packaging, runtime dependencies and reproducible install |
| CI / cross-platform | Project-wide build/test matrix and backend-specific CI | Six original tests plus quantization test have passed on user Windows build; no evidence here of broad CI matrix | **P1:** Windows CI first; Linux CI next; CUDA build test when backend is added |

Priority guide:
- **P0 — correctness and throughput blockers:** implement before considering the runtime a serious alternative for text inference.
- **P1 — backend and integration foundation:** needed for robust use in Brahma Desk.
- **P2 — advanced inference/serving features.**
- **P3 — broad ecosystem features not required for the first local-text milestone.

## What the latest user benchmark tells us

User-provided results:
- Before fused quantized dot kernels: 21.56 seconds total for an eight-token generation.
- After fused quantized dot kernels: 14.22 seconds total for the same prompt/token limit.
- All seven registered tests passed after the fused-kernel change.

This is about a 34.1% reduction in total wall time, but it is not a controlled decode-only comparison: the timing includes startup/model construction, prompt prefill and generation. A later branch update adds prefill/decode statistics, but those diagnostics need verification on the user's machine.

The llama.cpp user's stated 20–30 tokens/sec is not yet a fully controlled apples-to-apples benchmark against Brahma. Match model file, quantization, prompt, context, output-token count, hardware backend, thread count, warm/cold load conditions and measured decode-only tokens/sec before stating the magnitude of the performance gap.

## Recommended implementation order

### Milestone 1 — correctness and measurement
1. Add a trusted-reference comparison harness for tokenizer IDs and logits on fixed Qwen2.5 prompts.
2. Add a standalone quantized-kernel microbenchmark (Q4_K/Q6_K initially) and tests for malformed blocks/edge cases.
3. Report cold load, warm load, prefill tokens/sec and decode tokens/sec separately; record peak working set/RAM and available GPU memory.
4. Add maximum context and KV-cache memory checks.
5. Validate OpenMP detection and expose thread count at runtime; test a small thread sweep.

**Exit criterion:** deterministic tests and repeatable benchmark baselines, with no unexplained output/tokenizer mismatch.

### Milestone 2 — CPU inference throughput
1. Add reusable per-layer scratch/workspace buffers to avoid repeated allocations.
2. Add efficient SIMD-dispatched Q4_K/Q6_K kernels, then Q4_0/Q4_1/Q5/Q8 paths as needed.
3. Optimize the output projection; profile whether vocabulary projection dominates decode.
4. Implement multi-token prompt prefill and a more efficient attention path.
5. Add a benchmark script that compares the same prompt and generation length.

**Exit criterion:** measurable decode-only improvement without loss of correctness.

### Milestone 3 — CUDA backend on NVIDIA
1. Add a CMake CUDA option and a backend abstraction while keeping CPU as the always-available fallback.
2. Implement quantized GPU dot/matvec kernels and device-resident weights.
3. Keep KV cache and intermediate activations on-device to avoid transfers every operation.
4. Add device discovery, VRAM budgeting, explicit fallback and logging.
5. Validate logits against the CPU/reference implementation.

**Exit criterion:** actual GPU kernels execute, verified by runtime telemetry and GPU utilization—not merely a CUDA build flag or a CLI mode.

### Milestone 4 — hybrid and loading policies
1. Add layer placement/offloading with an explicit VRAM budget and transfer plan.
2. Add memory-mapped GGUF access and configurable eager/lazy/resident strategies.
3. Measure cold startup, repeated inference, RAM usage and page-fault behavior.
4. Add an automatic policy only after CPU/GPU/hybrid modes are measurable and reliable.

**Exit criterion:** each mode demonstrably runs the expected layers on the selected compute device and stays within configured memory limits.

### Milestone 5 — application-grade engine
1. Stable session/model C++ API (and C ABI if the product needs it).
2. Streaming output and cancellation.
3. Sampling/stop sequences, chat templates and tokenizer parity.
4. HTTP API/server, request concurrency, continuous batching.
5. Tool calling, grammar-constrained output and speculative decoding.
6. Multimodal support in a separate subsystem if Brahma Desk needs image/audio/video.

## Strategic recommendation

Do not try to implement every llama.cpp feature at once. First make Qwen2.5 text inference correct, memory-efficient, and measurably fast. The most consequential gaps for the user's stated target are:
1. exact tokenizer/logit parity and context safety;
2. optimized CPU quantized kernels and multi-token prompt prefill;
3. optimized output projection;
4. native CUDA execution with VRAM-resident weights;
5. hybrid placement and mmap only after CPU/GPU paths have measured baselines.

Keep Brahma Engine independent of llama.cpp/GGML as required. Its public implementation can be studied as an architectural reference, but avoid copying code without checking individual file licenses, attribution requirements, and the project's own license policy.
