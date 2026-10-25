#pragma once
#include <cstddef>
#include <span>
#include <vector>

namespace brahma {
struct Qwen2BlockWeights {
    std::span<const float> attn_norm;
    std::span<const float> q_weight;
    std::span<const float> q_bias;
    std::span<const float> k_weight;
    std::span<const float> k_bias;
    std::span<const float> v_weight;
    std::span<const float> v_bias;
    std::span<const float> attention_output_weight;
    std::span<const float> ffn_norm;
    std::span<const float> ffn_gate_weight;
    std::span<const float> ffn_up_weight;
    std::span<const float> ffn_down_weight;
};

// Executes one standard Qwen2 decoder block on row-major [sequence, hidden]
// float activations and already-dequantized row-major GGUF weights. Q/K/V
// projections use [hidden, projected_width] storage; attention supports GQA.
[[nodiscard]] std::vector<float> qwen2_transformer_block_f32(
    std::span<const float> input,
    std::size_t sequence_length,
    std::size_t hidden_size,
    std::size_t feed_forward_size,
    std::size_t attention_heads,
    std::size_t kv_heads,
    float rope_theta,
    float rms_norm_epsilon,
    const Qwen2BlockWeights& weights);
} // namespace brahma
