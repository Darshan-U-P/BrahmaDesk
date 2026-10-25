#include "brahma/quantization.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
bool near(float a, float b, float tolerance = 2.0e-4f) {
    return std::fabs(a - b) <= tolerance * (1.0f + std::fabs(b));
}
int main() {
    try {
        std::array<std::uint8_t, 144> q4{};
        q4[0] = 0x00; q4[1] = 0x3c; // d = fp16(1)
        q4[2] = 0x00; q4[3] = 0x38; // dmin = fp16(0.5)
        for (std::size_t i = 4; i < q4.size(); ++i)
            q4[i] = static_cast<std::uint8_t>((i * 37u + 11u) & 0xffu);
        std::array<std::uint8_t, 210> q6{};
        q6[208] = 0x00; q6[209] = 0x3c; // d = fp16(1)
        for (std::size_t i = 0; i < 208; ++i)
            q6[i] = static_cast<std::uint8_t>((i * 19u + 7u) & 0xffu);
        std::array<float, 256> input{};
        for (std::size_t i = 0; i < input.size(); ++i)
            input[i] = static_cast<float>(static_cast<int>(i % 23) - 11) / 13.0f;

        std::array<float, 256> decoded{};
        brahma::dequantize_q4_k_block(q4, decoded);
        float expected_q4 = 0.0f;
        for (std::size_t i = 0; i < input.size(); ++i) expected_q4 += decoded[i] * input[i];
        const float actual_q4 = brahma::dot_q4_k_block(q4, input);
        require(near(actual_q4, expected_q4), "Q4_K fused dot differs from dequantize-then-dot");

        brahma::dequantize_q6_k_block(q6, decoded);
        float expected_q6 = 0.0f;
        for (std::size_t i = 0; i < input.size(); ++i) expected_q6 += decoded[i] * input[i];
        const float actual_q6 = brahma::dot_q6_k_block(q6, input);
        require(near(actual_q6, expected_q6), "Q6_K fused dot differs from dequantize-then-dot");

        std::cout << "Fused quantized dot-product tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
