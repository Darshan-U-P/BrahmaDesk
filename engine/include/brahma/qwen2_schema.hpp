#pragma once
#include "brahma/gguf_reader.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace brahma {
struct Qwen2ModelConfig {
    std::uint64_t hidden_size{};
    std::uint64_t feed_forward_size{};
    std::uint64_t block_count{};
    std::uint64_t attention_heads{};
    std::uint64_t kv_heads{};
    std::uint64_t vocabulary_size{};
    double rope_theta{10000.0};
    double rms_norm_epsilon{1.0e-6};
};

// Validate a standard dense Qwen2 GGUF tensor schema, including every block's
// required tensor names/shapes and key architecture metadata. Throws on mismatch.
[[nodiscard]] Qwen2ModelConfig validate_qwen2_schema(const GgufModelInfo& model);
} // namespace brahma
