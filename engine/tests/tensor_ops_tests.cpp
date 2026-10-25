#include "brahma/tensor_ops.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

void req(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
bool near(float a, float b, float tol = 1.0e-5f) {
    return std::fabs(a - b) <= tol;
}
int main() {
    try {
        const std::array<float, 6> a{1, 2, 3, 4, 5, 6}; // [2,3]
        const std::array<float, 6> b{7, 8, 9, 10, 11, 12}; // [3,2]
        const auto c = brahma::matmul_f32(a, 2, 3, b, 2);
        req(c.size() == 4, "matmul output size");
        req(near(c[0], 58) && near(c[1], 64), "matmul first row");
        req(near(c[2], 139) && near(c[3], 154), "matmul second row");

        const std::array<float, 4> input{3, 4, 0, 2};
        const std::array<float, 2> weight{1, 1};
        std::array<float, 4> output{};
        brahma::rms_norm_f32(input, weight, output, 2, 2, 1.0e-5f);
        const float s0 = 1.0f / std::sqrt((9.0f + 16.0f) / 2.0f + 1.0e-5f);
        const float s1 = 1.0f / std::sqrt((0.0f + 4.0f) / 2.0f + 1.0e-5f);
        req(near(output[0], 3.0f*s0) && near(output[1], 4.0f*s0), "RMSNorm row 0");
        req(near(output[2], 0.0f) && near(output[3], 2.0f*s1), "RMSNorm row 1");

        bool rejected = false;
        try { (void)brahma::matmul_f32(a, 2, 2, b, 2); }
        catch (const std::invalid_argument&) { rejected = true; }
        req(rejected, "matmul shape mismatch rejection");

        rejected = false;
        try { brahma::rms_norm_f32(input, weight, output, 2, 2, 0.0f); }
        catch (const std::invalid_argument&) { rejected = true; }
        req(rejected, "RMSNorm invalid epsilon rejection");

        const std::array<float, 3> add_a{1.0f, -2.0f, 3.5f};
        const std::array<float, 3> add_b{4.0f, 2.0f, -0.5f};
        std::array<float, 3> add_out{};
        brahma::add_f32(add_a, add_b, add_out);
        req(near(add_out[0], 5.0f) && near(add_out[1], 0.0f) &&
            near(add_out[2], 3.0f), "elementwise addition");

        // SwiGLU: SiLU(gate) multiplied elementwise by the up projection.
        const std::array<float, 3> gate{-1.0f, 0.0f, 1.0f};
        const std::array<float, 3> up{2.0f, 3.0f, 4.0f};
        std::array<float, 3> gated{};
        brahma::swiglu_f32(gate, up, gated);
        req(near(gated[0], (-1.0f / (1.0f + std::exp(1.0f))) * 2.0f),
            "SwiGLU negative gate");
        req(near(gated[1], 0.0f), "SwiGLU zero gate");
        req(near(gated[2], (1.0f / (1.0f + std::exp(-1.0f))) * 4.0f),
            "SwiGLU positive gate");

        // RoPE: position zero is identity; position one rotates [1,0] by one radian.
        std::array<float, 4> rope_values{1, 0, 1, 0};
        brahma::rope_f32(rope_values, 2, 2);
        req(near(rope_values[0], 1) && near(rope_values[1], 0), "RoPE position zero");
        req(near(rope_values[2], std::cos(1.0f)) &&
            near(rope_values[3], std::sin(1.0f)), "RoPE position one");

        // NeoX RoPE pairs the first and second halves of the head dimension.
        std::array<float, 4> neox_values{1, 0, 0, 1};
        brahma::rope_f32(neox_values, 1, 4, 1, 10000.0f, brahma::RopeLayout::NeoX);
        req(near(neox_values[0], std::cos(1.0f)) &&
            near(neox_values[2], std::sin(1.0f)), "NeoX RoPE first half-pair");
        req(near(neox_values[1], -std::sin(0.01f)) &&
            near(neox_values[3], std::cos(0.01f)), "NeoX RoPE second half-pair");

        // Zero Q/K gives uniform attention over only the causal prefix.
        const std::array<float, 3> q{0, 0, 0};
        const std::array<float, 3> k{0, 0, 0};
        const std::array<float, 3> v{2, 4, 8};
        std::array<float, 3> attn{};
        brahma::causal_attention_f32(q, k, v, attn, 3, 1, 1);
        req(near(attn[0], 2.0f), "causal attention token 0");
        req(near(attn[1], 3.0f), "causal attention token 1");
        req(near(attn[2], 14.0f/3.0f), "causal attention token 2");

        rejected = false;
        try {
            std::array<float, 3> odd_rope{1, 2, 3};
            brahma::rope_f32(odd_rope, 1, 3);
        } catch (const std::invalid_argument&) { rejected = true; }
        req(rejected, "RoPE odd head dimension rejection");

        rejected = false;
        try {
            brahma::causal_attention_f32(q, k, v, attn, 3, 1, 2);
        } catch (const std::invalid_argument&) { rejected = true; }
        req(rejected, "attention shape mismatch rejection");

        std::cout << "CPU tensor operation tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
