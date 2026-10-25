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

        std::cout << "CPU tensor operation tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
