#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace brahma {
// Decode one GGML Q4_K block (256 values, 144 serialized bytes) to float.
void dequantize_q4_k_block(std::span<const std::uint8_t, 144> block,
                           std::span<float, 256> output);
[[nodiscard]] std::vector<float> dequantize_q4_k(std::span<const std::uint8_t> payload,
                                                 std::size_t element_count);

// Decode one GGML Q6_K block (256 values, 210 serialized bytes) to float.
void dequantize_q6_k_block(std::span<const std::uint8_t, 210> block,
                           std::span<float, 256> output);
[[nodiscard]] std::vector<float> dequantize_q6_k(std::span<const std::uint8_t> payload,
                                                 std::size_t element_count);

// Fused block dot products avoid materializing a temporary 256-float block.
[[nodiscard]] float dot_q4_k_block(std::span<const std::uint8_t, 144> block,
                                   std::span<const float, 256> input) noexcept;
[[nodiscard]] float dot_q6_k_block(std::span<const std::uint8_t, 210> block,
                                   std::span<const float, 256> input) noexcept;
} // namespace brahma
