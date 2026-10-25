#include "brahma/qwen2_runtime.hpp"
#include "brahma/qwen2_tokenizer.hpp"
#include "brahma/quantization.hpp"
#include "brahma/tensor_ops.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

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

std::vector<float> linear_vector(std::span<const float> input, std::span<const float> weight,
                                 std::size_t output_width) {
    if (input.empty() || output_width == 0 || weight.size() != input.size() * output_width)
        throw std::runtime_error("Linear projection weight shape mismatch");
    std::vector<float> output(output_width, 0.0f);
    // GGUF stores the first tensor dimension contiguously. For a [input, output]
    // GGUF matrix this is output-major, with input_width contiguous per output.
    for (std::size_t row = 0; row < output_width; ++row) {
        const auto offset = row * input.size();
        double sum = 0.0;
        for (std::size_t col = 0; col < input.size(); ++col)
            sum += static_cast<double>(weight[offset + col]) * input[col];
        output[row] = static_cast<float>(sum);
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
    std::unordered_map<std::string, std::vector<float>> weights;
    std::vector<std::vector<float>> key_cache;
    std::vector<std::vector<float>> value_cache;

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

    const std::vector<float>& get(const std::string& name) {
        auto it = weights.find(name);
        if (it != weights.end()) return it->second;
        auto loaded = registry.load(name);
        auto decoded = decode_tensor(loaded);
        return weights.emplace(name, std::move(decoded)).first->second;
    }

    std::vector<float> norm(std::span<const float> input, const std::string& name) {
        const auto& w = get(name);
        std::vector<float> out(input.size());
        rms_norm_f32(input, w, out, 1, input.size(), static_cast<float>(cfg.rms_norm_epsilon));
        return out;
    }

    std::vector<float> project(std::span<const float> input, const std::string& name,
                               std::size_t output_width) {
        return linear_vector(input, get(name), output_width);
    }

    std::vector<float> step(std::uint32_t token, std::size_t position) {
        const auto hidden = static_cast<std::size_t>(cfg.hidden_size);
        const auto ffn = static_cast<std::size_t>(cfg.feed_forward_size);
        const auto heads = static_cast<std::size_t>(cfg.attention_heads);
        const auto kv_heads = static_cast<std::size_t>(cfg.kv_heads);
        const auto dim = hidden / heads;
        const auto kv_width = kv_heads * dim;
        if (token >= cfg.vocabulary_size) throw std::runtime_error("Token ID is outside the model vocabulary");

        const auto& embedding = get("token_embd.weight");
        if (embedding.size() != static_cast<std::size_t>(cfg.vocabulary_size) * hidden)
            throw std::runtime_error("Token embedding shape mismatch");
        std::vector<float> x(embedding.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(token) * hidden),
                             embedding.begin() + static_cast<std::ptrdiff_t>((static_cast<std::size_t>(token) + 1) * hidden));

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
    if (max_new_tokens == 0) return {};
    const std::string formatted =
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    auto ids = impl_->tokenizer.encode(formatted);
    for (auto& cache : impl_->key_cache) cache.clear();
    for (auto& cache : impl_->value_cache) cache.clear();
    std::vector<float> current_logits;
    for (std::size_t i = 0; i < ids.size(); ++i) current_logits = impl_->step(ids[i], i);
    std::vector<std::uint32_t> generated;
    generated.reserve(max_new_tokens);
    for (std::size_t i = 0; i < max_new_tokens; ++i) {
        const auto it = std::max_element(current_logits.begin(), current_logits.end());
        if (it == current_logits.end()) throw std::runtime_error("Model returned empty logits");
        const auto next = static_cast<std::uint32_t>(std::distance(current_logits.begin(), it));
        if (impl_->tokenizer.has_eos_token() && next == impl_->tokenizer.eos_token_id()) break;
        ids.push_back(next);
        generated.push_back(next);
        if (i + 1 < max_new_tokens) current_logits = impl_->step(next, ids.size() - 1);
    }
    return impl_->tokenizer.decode(generated, true);
}

} // namespace brahma
