#include "brahma/tensor_ops.hpp"
#include <cmath>
#include <algorithm>
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

void rope_f32(
    std::span<float> values,
    std::size_t sequence_length,
    std::size_t head_dim,
    std::size_t position_offset,
    float theta,\n    RopeLayout layout) {
    const auto total = checked_product(sequence_length, head_dim, "RoPE shape overflow");
    if (head_dim == 0 || head_dim % 2 != 0 || values.size() != total) {
        throw std::invalid_argument("RoPE requires an even head dimension and matching input size");
    }
    if (!(theta > 0.0f) || !std::isfinite(theta)) {
        throw std::invalid_argument("RoPE theta must be finite and positive");
    }
    for (std::size_t pos = 0; pos < sequence_length; ++pos) {
        if (position_offset > std::numeric_limits<std::size_t>::max() - pos) {
            throw std::overflow_error("RoPE position overflow");
        }
        const auto absolute_pos = position_offset + pos;
        const auto row = pos * head_dim;
        for (std::size_t pair = 0; pair < head_dim / 2; ++pair) {
            const float exponent = static_cast<float>(2 * pair) / static_cast<float>(head_dim);
            const float angle = static_cast<float>(absolute_pos) /
                std::pow(theta, exponent);
            const float cs = std::cos(angle);
            const float sn = std::sin(angle);
            const auto i0 = layout == RopeLayout::NeoX ? row + pair : row + pair * 2;
            const auto i1 = layout == RopeLayout::NeoX
                ? row + pair + head_dim / 2 : row + pair * 2 + 1;
            const float x0 = values[i0];
            const float x1 = values[i1];
            values[i0] = x0 * cs - x1 * sn;
            values[i1] = x0 * sn + x1 * cs;
        }
    }
}

void causal_attention_f32(
    std::span<const float> q,
    std::span<const float> k,
    std::span<const float> v,
    std::span<float> output,
    std::size_t sequence_length,
    std::size_t heads,
    std::size_t head_dim,
    float scale) {
    const auto tokens_heads = checked_product(sequence_length, heads, "attention shape overflow");
    const auto total = checked_product(tokens_heads, head_dim, "attention shape overflow");
    if (sequence_length == 0 || heads == 0 || head_dim == 0 ||
        q.size() != total || k.size() != total || v.size() != total || output.size() != total) {
        throw std::invalid_argument("Attention input/output shape mismatch or empty dimensions");
    }
    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    } else if (!std::isfinite(scale)) {
        throw std::invalid_argument("Attention scale must be finite");
    }
    std::vector<float> scores(sequence_length);
    for (std::size_t head = 0; head < heads; ++head) {
        for (std::size_t query_pos = 0; query_pos < sequence_length; ++query_pos) {
            float max_score = -std::numeric_limits<float>::infinity();
            for (std::size_t key_pos = 0; key_pos <= query_pos; ++key_pos) {
                float dot = 0.0f;
                const auto q_offset = (query_pos * heads + head) * head_dim;
                const auto k_offset = (key_pos * heads + head) * head_dim;
                for (std::size_t d = 0; d < head_dim; ++d) {
                    dot += q[q_offset + d] * k[k_offset + d];
                }
                scores[key_pos] = dot * scale;
                max_score = std::max(max_score, scores[key_pos]);
            }
            double denominator = 0.0;
            for (std::size_t key_pos = 0; key_pos <= query_pos; ++key_pos) {
                scores[key_pos] = std::exp(scores[key_pos] - max_score);
                denominator += static_cast<double>(scores[key_pos]);
            }
            const auto out_offset = (query_pos * heads + head) * head_dim;
            for (std::size_t d = 0; d < head_dim; ++d) {
                double weighted = 0.0;
                for (std::size_t key_pos = 0; key_pos <= query_pos; ++key_pos) {
                    const auto v_offset = (key_pos * heads + head) * head_dim;
                    weighted += static_cast<double>(scores[key_pos]) * v[v_offset + d];
                }
                output[out_offset + d] = static_cast<float>(weighted / denominator);
            }
        }
    }
}
} // namespace brahma
