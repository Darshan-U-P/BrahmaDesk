#include "brahma/qwen2_tokenizer.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    try {
        brahma::GgufModelInfo model;
        model.metadata.emplace("tokenizer.ggml.tokens",
            std::vector<std::string>{"h", "i", "hi", "<|im_start|>"});
        model.metadata.emplace("tokenizer.ggml.merges", std::vector<std::string>{"h i"});
        model.metadata.emplace("tokenizer.ggml.eos_token_id", std::uint64_t(3));
        const brahma::Qwen2Tokenizer tokenizer(model);
        const auto ids = tokenizer.encode("hi");
        if (ids.size() != 1 || ids[0] != 2) throw std::runtime_error("BPE merge rank failed");
        if (tokenizer.decode(ids) != "hi") throw std::runtime_error("Byte-level decode failed");
        const auto special = tokenizer.encode("<|im_start|>hi");
        if (special.size() != 2 || special[0] != 3 || special[1] != 2)
            throw std::runtime_error("Special token splitting failed");
        if (tokenizer.eos_token_id() != 3 || !tokenizer.has_eos_token())
            throw std::runtime_error("EOS metadata parsing failed");
        std::cout << "Qwen2 tokenizer tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
