#pragma once
#include <cstddef>
#include <span>
#include <vector>

namespace brahma {

enum class RopeLayout { Interleaved, NeoX };

// Row-major dense matrix multiplication: A[M,K] * B[K,N] -> C[M,N].
[[nodiscard]] std::vector<float> matmul_f32(
    std::span<const float> a, std::size_t m, std::size_t k,
    std::span<const float> b, std::size_t n);

// Apply adjacent-pair RoPE to [sequence, head_dim] rows, in-place.
// head_dim must be even. Position indices start at position_offset.
void rope_f32(
    std::span<float> values,
    std::size_t sequence_length,
    std::size_t head_dim,
    std::size_t position_offset = 0,
    float theta = 10000.0f,
    RopeLayout layout = RopeLayout::Interleaved);

// Single-sequence causal multi-head attention.
// q/k/v and output use flattened [sequence, heads, head_dim] layout.
// scale == 0 selects 1/sqrt(head_dim); otherwise scale is used directly.
void causal_attention_f32(
    std::span<const float> q,
    std::span<const float> k,
    std::span<const float> v,
    std::span<float> output,
    std::size_t sequence_length,
    std::size_t heads,
    std::size_t head_dim,
    float scale = 0.0f);

// SwiGLU activation: output[i] = SiLU(gate[i]) * up[i].
void swiglu_f32(
    std::span<const float> gate,
    std::span<const float> up,
    std::span<float> output);

// RMSNorm over the last dimension of each row.
// input and output have shape [rows, width]; weight has shape [width].
void rms_norm_f32(
    std::span<const float> input,
    std::span<const float> weight,
    std::span<float> output,
    std::size_t rows,
    std::size_t width,
    float epsilon = 1.0e-5f);

} // namespace brahma
