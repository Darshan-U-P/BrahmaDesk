#pragma once
#include "brahma/qwen2_schema.hpp"
#include "brahma/tensor_registry.hpp"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace brahma {

// Native CPU Qwen2 inference runtime. Weights are decoded from GGUF directly;
// this class has no llama.cpp/GGML dependency.
class Qwen2Runtime {
public:
    explicit Qwen2Runtime(const std::filesystem::path& model_path);
    Qwen2Runtime(Qwen2Runtime&&) noexcept;
    Qwen2Runtime& operator=(Qwen2Runtime&&) noexcept;
    ~Qwen2Runtime();

    Qwen2Runtime(const Qwen2Runtime&) = delete;
    Qwen2Runtime& operator=(const Qwen2Runtime&) = delete;

    [[nodiscard]] const Qwen2ModelConfig& config() const noexcept;
    // Replays the supplied token IDs from position zero and returns next-token logits.
    [[nodiscard]] std::vector<float> logits(std::span<const std::uint32_t> token_ids);
    [[nodiscard]] std::uint32_t greedy_next_token(std::span<const std::uint32_t> token_ids);
    // Greedy Qwen2.5-Instruct generation using the tokenizer embedded in GGUF.
    [[nodiscard]] std::string generate(const std::string& prompt, std::size_t max_new_tokens = 32);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace brahma
