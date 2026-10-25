#include "brahma/tensor_ops.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace brahma {
namespace {
std::size_t checked_product(std::size_t a, std::size_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::overflow_error(what);
    }
    return a * b;
}
}

std::vector<float> matmul_f32(
    std::span<const float> a, std::size_t m, std::size_t k,
    std::span<const float> b, std::size_t n) {
    const auto a_size = checked_product(m, k, "matmul A shape overflow");
    const auto b_size = checked_product(k, n, "matmul B shape overflow");
    const auto c_size = checked_product(m, n, "matmul output shape overflow");
    if (a.size() != a_size || b.size() != b_size) {
        throw std::invalid_argument("matmul input size does not match shape");
    }
    std::vector<float> c(c_size, 0.0f);
    for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t inner = 0; inner < k; ++inner) {
            const float av = a[row * k + inner];
            for (std::size_t col = 0; col < n; ++col) {
                c[row * n + col] += av * b[inner * n + col];
            }
        }
    }
    return c;
}

void rms_norm_f32(
    std::span<const float> input,
    std::span<const float> weight,
    std::span<float> output,
    std::size_t rows,
    std::size_t width,
    float epsilon) {
    const auto total = checked_product(rows, width, "RMSNorm shape overflow");
    if (width == 0 || input.size() != total || output.size() != total ||
        weight.size() != width) {
        throw std::invalid_argument("RMSNorm input, weight, or output shape mismatch");
    }
    if (!(epsilon > 0.0f) || !std::isfinite(epsilon)) {
        throw std::invalid_argument("RMSNorm epsilon must be finite and positive");
    }
    for (std::size_t row = 0; row < rows; ++row) {
        const auto offset = row * width;
        double sum_squares = 0.0;
        for (std::size_t col = 0; col < width; ++col) {
            const double x = static_cast<double>(input[offset + col]);
            sum_squares += x * x;
        }
        const float mean_square = static_cast<float>(sum_squares / static_cast<double>(width));
        const float scale = 1.0f / std::sqrt(mean_square + epsilon);
        for (std::size_t col = 0; col < width; ++col) {
            output[offset + col] = input[offset + col] * scale * weight[col];
        }
    }
}
} // namespace brahma
