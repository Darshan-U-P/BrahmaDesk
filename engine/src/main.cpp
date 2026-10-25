#include "brahma/gguf_reader.hpp"
#include "brahma/qwen2_schema.hpp"
#include "brahma/tensor_registry.hpp"
#include "brahma/tensor_validator.hpp"

#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
void print_usage() {
    std::cerr
        << "Usage:\n"
        << "  brahma-engine <model.gguf>\n"
        << "  brahma-engine <model.gguf> --list-tensors\n"
        << "  brahma-engine <model.gguf> --validate-qwen2\n"
        << "Brahma Engine Core: GGUF inspection and tensor payload validation; inference is not implemented yet.\n";
}

void print_tensor_list(const brahma::TensorRegistry& registry) {
    std::cout << "Tensor count: " << registry.model().tensors.size() << "\n";
    for (const auto& tensor : registry.model().tensors) {
        const auto& layout = registry.layout(tensor.name);
        std::cout << tensor.name << " | shape=[";
        for (std::size_t i = 0; i < tensor.dimensions.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << tensor.dimensions[i];
        }
        std::cout << "] | type=" << brahma::ggml_type_name(tensor.ggml_type)
                  << " (" << tensor.ggml_type << ")"
                  << " | elements=" << layout.elements
                  << " | bytes=" << layout.bytes
                  << " | offset=" << layout.absolute_offset << "\n";
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        print_usage();
        return 2;
    }

    const std::string mode = argc == 3 ? argv[2] : "";
    if (!mode.empty() && mode != "--list-tensors" && mode != "--validate-qwen2") {
        print_usage();
        return 2;
    }

    try {
        auto model = brahma::GgufReader::inspect(argv[1]);
        brahma::TensorRegistry registry(std::move(model), argv[1]);

        if (mode == "--list-tensors") {
            print_tensor_list(registry);
            return 0;
        }
        if (mode == "--validate-qwen2") {
            const auto config = brahma::validate_qwen2_schema(registry.model());
            std::cout << "Qwen2 schema validation: PASS\n"
                      << "Hidden size: " << config.hidden_size << "\n"
                      << "Feed-forward size: " << config.feed_forward_size << "\n"
                      << "Transformer blocks: " << config.block_count << "\n"
                      << "Attention heads: " << config.attention_heads << "\n"
                      << "KV heads: " << config.kv_heads << "\n"
                      << "Vocabulary size: " << config.vocabulary_size << "\n"
                      << "RoPE theta: " << config.rope_theta << "\n"
                      << "RMSNorm epsilon: " << config.rms_norm_epsilon << "\n";
            return 0;
        }

        std::uint64_t total = 0;
        for (const auto& tensor : registry.model().tensors) {
            const auto bytes = registry.layout(tensor.name).bytes;
            if (bytes > UINT64_MAX - total) {
                throw std::runtime_error("Total tensor byte count overflow");
            }
            total += bytes;
        }

        const auto& info = registry.model();
        std::cout << "Brahma Engine Core 0.3\n"
                  << "GGUF version: " << info.version << "\n"
                  << "Architecture: " << (info.architecture().empty() ? "(not declared)" : info.architecture()) << "\n"
                  << "Metadata entries: " << info.metadata.size() << "\n"
                  << "Tensor descriptors: " << info.tensors.size() << "\n"
                  << "Tensor data offset: " << info.tensor_data_offset << "\n"
                  << "Alignment: " << info.alignment << "\n"
                  << "Vocabulary entries: " << info.vocabulary_size() << "\n"
                  << "Payload validation: PASS\n"
                  << "Validated tensor payload bytes: " << total << "\n"
                  << "Inference: NOT IMPLEMENTED\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Brahma Engine error: " << error.what() << "\n";
        return 1;
    }
}
