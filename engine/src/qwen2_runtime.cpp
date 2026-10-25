#include "brahma/qwen2_runtime.hpp"
#include "brahma/qwen2_tokenizer.hpp"
#include "brahma/quantization.hpp"
#include "brahma/tensor_ops.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#ifdef BRAHMA_ENGINE_USE_OPENMP
#include <omp.h>
#endif

namespace brahma {
namespace {
float fp16_to_float(std::uint16_t h) noexcept {
    const std::uint32_t sign = std::uint32_t(h & 0x8000u) << 16;
    std::uint32_t exp = (h >> 10) & 0x1fu;
    std::uint32_t mant = h & 0x03ffu;
    std::uint32_t bits{};
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            int e = -14;
            while ((mant & 0x0400u) == 0) { mant <<= 1; --e; }
            mant &= 0x03ffu;
            bits = sign | (std::uint32_t(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) bits = sign | 0x7f800000u | (mant << 13);
    else bits = sign | ((exp + 112u) << 23) | (mant << 13);
    return std::bit_cast<float>(bits);
}

std::vector<float> decode_tensor(const LoadedTensor& tensor) {
    const auto n64 = tensor.layout.elements;
    if (n64 > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("Tensor is too large for this process: " + tensor.info.name);
    const auto n = static_cast<std::size_t>(n64);
    switch (tensor.info.ggml_type) {
        case 0: {
            if (tensor.payload.size() != n * sizeof(float)) throw std::runtime_error("Invalid F32 payload: " + tensor.info.name);
            std::vector<float> out(n);
            for (std::size_t i = 0; i < n; ++i) {
                const auto p = tensor.payload.data() + i * 4;
                const std::uint32_t bits = std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
                    (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
                out[i] = std::bit_cast<float>(bits);
            }
            return out;
        }
        case 1: {
            if (tensor.payload.size() != n * 2) throw std::runtime_error("Invalid F16 payload: " + tensor.info.name);
            std::vector<float> out(n);
            for (std::size_t i = 0; i < n; ++i) {
                const auto p = tensor.payload.data() + i * 2;
                out[i] = fp16_to_float(std::uint16_t(p[0]) | (std::uint16_t(p[1]) << 8));
            }
            return out;
        }
        case 2: case 3: case 6: case 7: case 8: {
            const auto type = tensor.info.ggml_type;
            const std::size_t block_bytes = type == 2 ? 18 : type == 3 ? 20 :
                type == 6 ? 22 : type == 7 ? 24 : 34;
            const std::size_t values_per_block = 32;
            if (n % values_per_block || tensor.payload.size() != (n / values_per_block) * block_bytes)
                throw std::runtime_error("Invalid quantized block payload: " + tensor.info.name);
            std::vector<float> out(n);
            for (std::size_t b = 0; b < n / values_per_block; ++b) {
                const auto* p = tensor.payload.data() + b * block_bytes;
                const float d = fp16_to_float(std::uint16_t(p[0]) | (std::uint16_t(p[1]) << 8));
                if (type == 2) {
                    for (std::size_t j = 0; j < 16; ++j) {
                        out[b*32+j] = (float(int(p[2+j] & 0x0f)-8))*d;
                        out[b*32+j+16] = (float(int(p[2+j] >> 4)-8))*d;
                    }
                } else if (type == 3) {
                    const float m = fp16_to_float(std::uint16_t(p[2]) | (std::uint16_t(p[3]) << 8));
                    for (std::size_t j = 0; j < 16; ++j) {
                        out[b*32+j] = float(p[4+j] & 0x0f)*d+m;
                        out[b*32+j+16] = float(p[4+j] >> 4)*d+m;
                    }
                } else if (type == 6 || type == 7) {
                    const std::size_t qh_off = type == 6 ? 2 : 4;
                    const std::size_t qs_off = type == 6 ? 6 : 8;
                    const float m = type == 7 ? fp16_to_float(std::uint16_t(p[2]) | (std::uint16_t(p[3]) << 8)) : 0.0f;
                    std::uint32_t qh = std::uint32_t(p[qh_off]) | (std::uint32_t(p[qh_off+1]) << 8) |
                        (std::uint32_t(p[qh_off+2]) << 16) | (std::uint32_t(p[qh_off+3]) << 24);
                    for (std::size_t j = 0; j < 16; ++j) {
                        const int q0 = int(p[qs_off+j] & 0x0f) | (int((qh >> j) & 1u) << 4);
                        const int q1 = int(p[qs_off+j] >> 4) | (int((qh >> (j+16)) & 1u) << 4);
                        out[b*32+j] = (type == 6 ? float(q0-16)*d : float(q0)*d+m);
                        out[b*32+j+16] = (type == 6 ? float(q1-16)*d : float(q1)*d+m);
                    }
                } else {
                    for (std::size_t j = 0; j < 32; ++j)
                        out[b*32+j] = d * float(static_cast<std::int8_t>(p[2+j]));
                }
            }
            return out;
        }
        case 12: return dequantize_q4_k(tensor.payload, n);
        case 14: return dequantize_q6_k(tensor.payload, n);
        default:
            throw std::runtime_error("Native inference does not yet decode GGML type " +
                std::to_string(tensor.info.ggml_type) + " for tensor " + tensor.info.name);
    }
}

void add_bias(std::vector<float>& values, std::span<const float> bias) {
    if (values.size() != bias.size()) throw std::runtime_error("Projection bias shape mismatch");
    for (std::size_t i = 0; i < values.size(); ++i) values[i] += bias[i];
}

std::vector<float> quantized_linear_vector(std::span<const float> input,
        const LoadedTensor& tensor, std::size_t output_width) {
    const auto type = tensor.info.ggml_type;
    const std::size_t block_values = 256;
    const std::size_t block_bytes = type == 12 ? 144 : 210;
    if ((type != 12 && type != 14) || input.empty() ||
        input.size() % block_values != 0 ||
        tensor.layout.elements != input.size() * output_width ||
        tensor.payload.size() != (tensor.layout.elements / block_values) * block_bytes) {
        throw std::runtime_error("Quantized projection shape or payload mismatch: " + tensor.info.name);
    }
    const std::size_t blocks_per_row = input.size() / block_values;
    std::vector<float> output(output_width, 0.0f);
#ifdef BRAHMA_ENGINE_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t row_i = 0; row_i < static_cast<std::int64_t>(output_width); ++row_i) {
        const auto row = static_cast<std::size_t>(row_i);
        float sum = 0.0f;
        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            const auto block_index = row * blocks_per_row + b;
            const auto* p = tensor.payload.data() + block_index * block_bytes;
            const auto input_offset = b * block_values;
            const std::span<const float, 256> input_block(input.data() + input_offset, 256);
            if (type == 12)
                sum += dot_q4_k_block(std::span<const std::uint8_t, 144>(p, 144), input_block);
            else
                sum += dot_q6_k_block(std::span<const std::uint8_t, 210>(p, 210), input_block);
        }
        output[row] = sum;
    }
    return output;
}

std::vector<float> linear_vector(std::span<const float> input, std::span<const float> weight,
                                 std::size_t output_width) {
    if (input.empty() || output_width == 0 || weight.size() != input.size() * output_width)
        throw std::runtime_error("Linear projection weight shape mismatch");
    std::vector<float> output(output_width, 0.0f);
    // GGUF stores the first tensor dimension contiguously. For a [input, output]
    // GGUF matrix this is output-major, with input_width contiguous per output.
    // Independent output rows can be evaluated in parallel; float accumulation
    // allows MSVC's optimized Release build to vectorize the inner dot product.
#ifdef BRAHMA_ENGINE_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t row_i = 0; row_i < static_cast<std::int64_t>(output_width); ++row_i) {
        const auto row = static_cast<std::size_t>(row_i);
        const auto offset = row * input.size();
        float sum = 0.0f;
        for (std::size_t col = 0; col < input.size(); ++col)
            sum += weight[offset + col] * input[col];
        output[row] = sum;
    }
    return output;
}

void apply_rope(std::vector<float>& values, std::size_t heads, std::size_t head_dim,
                std::size_t position, float theta) {
    if (values.size() != heads * head_dim) throw std::runtime_error("RoPE projection shape mismatch");
    for (std::size_t head = 0; head < heads; ++head) {
        auto row = std::span<float>(values.data() + head * head_dim, head_dim);
        rope_f32(row, 1, head_dim, position, theta, RopeLayout::NeoX);
    }
}
} // namespace

struct Qwen2Runtime::Impl {
    std::filesystem::path path;
    TensorRegistry registry;
    Qwen2ModelConfig cfg;
    Qwen2Tokenizer tokenizer;
    std::unordered_map<std::string, LoadedTensor> raw_tensors;
    std::unordered_map<std::string, std::vector<float>> weights;
    std::vector<std::vector<float>> key_cache;
    std::vector<std::vector<float>> value_cache;
    GenerationStats last_stats;
    double projection_seconds{};
    double normalization_seconds{};
    double attention_seconds{};

    explicit Impl(const std::filesystem::path& p)
        : path(p), registry(GgufReader::inspect(p), p), cfg(validate_qwen2_schema(registry.model())), tokenizer(registry.model()) {
        if (cfg.hidden_size > std::numeric_limits<std::size_t>::max() ||
            cfg.feed_forward_size > std::numeric_limits<std::size_t>::max() ||
            cfg.vocabulary_size > std::numeric_limits<std::size_t>::max() ||
            cfg.block_count > std::numeric_limits<std::size_t>::max())
            throw std::runtime_error("Qwen2 dimensions exceed addressable memory");
        key_cache.resize(static_cast<std::size_t>(cfg.block_count));
        value_cache.resize(static_cast<std::size_t>(cfg.block_count));
    }

    const LoadedTensor& raw(const std::string& name) {
        auto it = raw_tensors.find(name);
        if (it != raw_tensors.end()) return it->second;
        return raw_tensors.emplace(name, registry.load(name)).first->second;
    }

    const std::vector<float>& get(const std::string& name) {
        auto it = weights.find(name);
        if (it != weights.end()) return it->second;
        const auto& tensor = raw(name);
        auto decoded = decode_tensor(tensor);
        return weights.emplace(name, std::move(decoded)).first->second;
    }

    std::vector<float> embedding_row(const std::string& name, std::size_t row,
                                     std::size_t row_width) {
        const auto& tensor = raw(name);
        if (row_width == 0 || tensor.layout.elements % row_width != 0 ||
            row >= tensor.layout.elements / row_width)
            throw std::runtime_error("Embedding row shape mismatch: " + name);
        const auto type = tensor.info.ggml_type;
        if (type == 12 || type == 14) {
            constexpr std::size_t block_values = 256;
            const std::size_t block_bytes = type == 12 ? 144 : 210;
            if (row_width % block_values != 0)
                throw std::runtime_error("Quantized embedding row is not block-aligned: " + name);
            const auto blocks_per_row = row_width / block_values;
            std::vector<float> result(row_width);
            std::array<float, 256> decoded{};
            for (std::size_t b = 0; b < blocks_per_row; ++b) {
                const auto block_index = row * blocks_per_row + b;
                const auto* p = tensor.payload.data() + block_index * block_bytes;
                if (type == 12)
                    dequantize_q4_k_block(std::span<const std::uint8_t, 144>(p, 144), decoded);
                else
                    dequantize_q6_k_block(std::span<const std::uint8_t, 210>(p, 210), decoded);
                std::copy(decoded.begin(), decoded.end(), result.begin() + static_cast<std::ptrdiff_t>(b * block_values));
            }
            return result;
        }
        const auto& all = get(name);
        const auto begin = all.begin() + static_cast<std::ptrdiff_t>(row * row_width);
        return std::vector<float>(begin, begin + static_cast<std::ptrdiff_t>(row_width));
    }

    std::vector<float> norm(std::span<const float> input, const std::string& name) {
        const auto op_start = std::chrono::steady_clock::now();
        const auto& w = get(name);
        std::vector<float> out(input.size());
        rms_norm_f32(input, w, out, 1, input.size(), static_cast<float>(cfg.rms_norm_epsilon));
        normalization_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - op_start).count();
        return out;
    }

    std::vector<float> project(std::span<const float> input, const std::string& name,
                               std::size_t output_width) {
        const auto op_start = std::chrono::steady_clock::now();
        const auto& tensor = raw(name);
        auto output = (tensor.info.ggml_type == 12 || tensor.info.ggml_type == 14)
            ? quantized_linear_vector(input, tensor, output_width)
            : linear_vector(input, get(name), output_width);
        projection_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - op_start).count();
        return output;
    }

    std::vector<float> step(std::uint32_t token, std::size_t position) {
        const auto hidden = static_cast<std::size_t>(cfg.hidden_size);
        const auto ffn = static_cast<std::size_t>(cfg.feed_forward_size);
        const auto heads = static_cast<std::size_t>(cfg.attention_heads);
        const auto kv_heads = static_cast<std::size_t>(cfg.kv_heads);
        const auto dim = hidden / heads;
        const auto kv_width = kv_heads * dim;
        if (token >= cfg.vocabulary_size) throw std::runtime_error("Token ID is outside the model vocabulary");

        const auto& embedding_tensor = raw("token_embd.weight");
        if (embedding_tensor.layout.elements != static_cast<std::size_t>(cfg.vocabulary_size) * hidden)
            throw std::runtime_error("Token embedding shape mismatch");
        std::vector<float> x = embedding_row("token_embd.weight", static_cast<std::size_t>(token), hidden);

        for (std::size_t layer = 0; layer < static_cast<std::size_t>(cfg.block_count); ++layer) {
            const auto p = "blk." + std::to_string(layer) + ".";
            const auto residual = x;
            auto n = norm(x, p + "attn_norm.weight");
            auto q = project(n, p + "attn_q.weight", hidden);
            auto k = project(n, p + "attn_k.weight", kv_width);
            auto v = project(n, p + "attn_v.weight", kv_width);
            add_bias(q, get(p + "attn_q.bias"));
            add_bias(k, get(p + "attn_k.bias"));
            add_bias(v, get(p + "attn_v.bias"));
            apply_rope(q, heads, dim, position, static_cast<float>(cfg.rope_theta));
            apply_rope(k, kv_heads, dim, position, static_cast<float>(cfg.rope_theta));

            auto& kc = key_cache[layer];
            auto& vc = value_cache[layer];
            kc.insert(kc.end(), k.begin(), k.end());
            vc.insert(vc.end(), v.begin(), v.end());
            const auto context_tokens = position + 1;
            if (kc.size() != context_tokens * kv_width || vc.size() != context_tokens * kv_width)
                throw std::runtime_error("KV cache position mismatch");

            const auto attention_start = std::chrono::steady_clock::now();
            std::vector<float> attended(hidden, 0.0f);
            const auto group_size = heads / kv_heads;
            const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
            std::vector<double> scores(context_tokens);
            for (std::size_t h = 0; h < heads; ++h) {
                const auto kvh = h / group_size;
                double max_score = -std::numeric_limits<double>::infinity();
                for (std::size_t t = 0; t < context_tokens; ++t) {
                    double dot = 0.0;
                    const auto qoff = h * dim, koff = t * kv_width + kvh * dim;
                    for (std::size_t d = 0; d < dim; ++d) dot += double(q[qoff + d]) * kc[koff + d];
                    scores[t] = dot * scale;
                    max_score = std::max(max_score, scores[t]);
                }
                double denom = 0.0;
                for (auto& s : scores) { s = std::exp(s - max_score); denom += s; }
                for (std::size_t d = 0; d < dim; ++d) {
                    double sum = 0.0;
                    for (std::size_t t = 0; t < context_tokens; ++t)
                        sum += scores[t] * vc[t * kv_width + kvh * dim + d];
                    attended[h * dim + d] = static_cast<float>(sum / denom);
                }
            }
            attention_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - attention_start).count();
            auto attn_out = project(attended, p + "attn_output.weight", hidden);
            x.resize(hidden);
            for (std::size_t i = 0; i < hidden; ++i) x[i] = residual[i] + attn_out[i];

            const auto residual_ffn = x;
            n = norm(x, p + "ffn_norm.weight");
            auto gate = project(n, p + "ffn_gate.weight", ffn);
            auto up = project(n, p + "ffn_up.weight", ffn);
            std::vector<float> gated(ffn);
            swiglu_f32(gate, up, gated);
            auto down = project(gated, p + "ffn_down.weight", hidden);
            for (std::size_t i = 0; i < hidden; ++i) x[i] = residual_ffn[i] + down[i];
        }

        auto final_norm = norm(x, "output_norm.weight");
        const auto output_name = registry.contains("output.weight") ? "output.weight" : "token_embd.weight";
        auto logits = project(final_norm, output_name, static_cast<std::size_t>(cfg.vocabulary_size));
        if (registry.contains("output.bias")) add_bias(logits, get("output.bias"));
        return logits;
    }
};

Qwen2Runtime::Qwen2Runtime(const std::filesystem::path& model_path)
    : impl_(std::make_unique<Impl>(model_path)) {}
Qwen2Runtime::Qwen2Runtime(Qwen2Runtime&&) noexcept = default;
Qwen2Runtime& Qwen2Runtime::operator=(Qwen2Runtime&&) noexcept = default;
Qwen2Runtime::~Qwen2Runtime() = default;

const Qwen2ModelConfig& Qwen2Runtime::config() const noexcept { return impl_->cfg; }
const GenerationStats& Qwen2Runtime::last_generation_stats() const noexcept { return impl_->last_stats; }
std::vector<std::uint32_t> Qwen2Runtime::tokenize(const std::string& text) const { return impl_->tokenizer.encode(text); }
const char* Qwen2Runtime::parallel_backend() noexcept {
#ifdef BRAHMA_ENGINE_USE_OPENMP
    return "OpenMP";
#else
    return "serial (OpenMP not found by CMake)";
#endif
}

std::vector<float> Qwen2Runtime::logits(std::span<const std::uint32_t> token_ids) {
    if (token_ids.empty()) throw std::invalid_argument("At least one prompt token ID is required");
    for (auto& cache : impl_->key_cache) cache.clear();
    for (auto& cache : impl_->value_cache) cache.clear();
    std::vector<float> result;
    for (std::size_t i = 0; i < token_ids.size(); ++i) result = impl_->step(token_ids[i], i);
    return result;
}

std::uint32_t Qwen2Runtime::greedy_next_token(std::span<const std::uint32_t> token_ids) {
    const auto values = logits(token_ids);
    const auto it = std::max_element(values.begin(), values.end());
    if (it == values.end()) throw std::runtime_error("Model returned empty logits");
    return static_cast<std::uint32_t>(std::distance(values.begin(), it));
}

std::string Qwen2Runtime::generate(const std::string& prompt, std::size_t max_new_tokens) {
    impl_->last_stats = {};
    impl_->projection_seconds = 0.0;
    impl_->normalization_seconds = 0.0;
    impl_->attention_seconds = 0.0;
    if (max_new_tokens == 0) return {};
    const std::string formatted =
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    auto ids = impl_->tokenizer.encode(formatted);
    impl_->last_stats.prompt_tokens = ids.size();
    for (auto& cache : impl_->key_cache) cache.clear();
    for (auto& cache : impl_->value_cache) cache.clear();
    std::vector<float> current_logits;
    const auto prefill_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < ids.size(); ++i) current_logits = impl_->step(ids[i], i);
    const auto prefill_end = std::chrono::steady_clock::now();
    impl_->last_stats.prefill_seconds =
        std::chrono::duration<double>(prefill_end - prefill_start).count();
    std::vector<std::uint32_t> generated;
    generated.reserve(max_new_tokens);
    const auto decode_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < max_new_tokens; ++i) {
        const auto it = std::max_element(current_logits.begin(), current_logits.end());
        if (it == current_logits.end()) throw std::runtime_error("Model returned empty logits");
        const auto next = static_cast<std::uint32_t>(std::distance(current_logits.begin(), it));
        if (impl_->tokenizer.has_eos_token() && next == impl_->tokenizer.eos_token_id()) break;
        ids.push_back(next);
        generated.push_back(next);
        if (i + 1 < max_new_tokens) current_logits = impl_->step(next, ids.size() - 1);
    }
    const auto decode_end = std::chrono::steady_clock::now();
    impl_->last_stats.generated_tokens = generated.size();
    impl_->last_stats.decode_seconds =
        std::chrono::duration<double>(decode_end - decode_start).count();
    impl_->last_stats.projection_seconds = impl_->projection_seconds;
    impl_->last_stats.normalization_seconds = impl_->normalization_seconds;
    impl_->last_stats.attention_seconds = impl_->attention_seconds;
    return impl_->tokenizer.decode(generated, true);
}

} // namespace brahma
