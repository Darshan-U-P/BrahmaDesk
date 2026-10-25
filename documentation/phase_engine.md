Recommended roadmap

Phase 1 — Establish correctness and profiling
Start here
- Compare tokenizer IDs and model logits against a trusted reference.
- Measure model-load time separately from inference.
- Profile the time spent in matrix projections, attention, normalization and output projection.
- Confirm OpenMP/thread configuration and add repeatable benchmarks.




Phase 2 — Optimize CPU inference
- Reuse scratch buffers instead of repeatedly allocating vectors.
- Optimize Q4_K/Q6_K and other high-impact quantized kernels with SIMD.
- Implement genuine multi-token prompt prefill.
- Optimize the vocabulary/output projection and attention path.




Phase 3 — Implement a real CUDA backend
- Execute actual inference kernels on your NVIDIA GPU.
- Keep weights, activations and KV cache on the GPU where practical.
- Track VRAM use and validate GPU results against CPU/reference results.




Phase 4 — Hybrid execution and application integration
- Offload selected layers to the GPU when VRAM is insufficient for the full model.
- Add memory-mapped model loading.
- Add streaming output and a stable API for Brahma Desk.