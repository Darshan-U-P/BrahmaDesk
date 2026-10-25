#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace brahma {
// Decode one GGML Q4_K block (256 values, 144 serialized bytes) to float.
void dequantize_q4_k_block(std::span<const std::uint8_t, 144> block,
                           std::span<float, 256> output);
// Decode a contiguous Q4_K tensor payload. Element count must be a multiple of 256.
[[nodiscard]] std::vector<float> dequantize_q4_k(std::span<const std::uint8_t> payload,
                                                 std::size_t element_count);
} // namespace brahma
