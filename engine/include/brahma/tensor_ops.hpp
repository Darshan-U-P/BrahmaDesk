#pragma once
#include <cstddef>
#include <span>
#include <vector>

namespace brahma {

// Row-major dense matrix multiplication: A[M,K] * B[K,N] -> C[M,N].
[[nodiscard]] std::vector<float> matmul_f32(
    std::span<const float> a, std::size_t m, std::size_t k,
    std::span<const float> b, std::size_t n);

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
