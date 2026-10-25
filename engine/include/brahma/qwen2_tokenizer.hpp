#pragma once
#include "brahma/gguf_reader.hpp"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace brahma {

// GGUF byte-level BPE tokenizer for Qwen2-family vocabularies.
// The pre-tokenizer is deliberately self-contained and does not use Python.
class Qwen2Tokenizer {
public:
    explicit Qwen2Tokenizer(const GgufModelInfo& model);
    [[nodiscard]] std::vector<std::uint32_t> encode(const std::string& text) const;
    [[nodiscard]] std::string decode(const std::vector<std::uint32_t>& ids,
                                     bool skip_special_tokens = true) const;
    [[nodiscard]] std::uint32_t eos_token_id() const noexcept { return eos_id_; }
    [[nodiscard]] bool has_eos_token() const noexcept { return has_eos_; }

private:
    std::vector<std::string> tokens_;
    std::unordered_map<std::string, std::uint32_t> token_ids_;
    std::unordered_map<std::string, std::uint32_t> merge_ranks_;
    std::vector<std::string> special_tokens_;
    std::uint32_t eos_id_{};
    bool has_eos_{};
};

} // namespace brahma
