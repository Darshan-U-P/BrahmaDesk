#include "brahma/gguf_reader.hpp"
#include "brahma/qwen2_schema.hpp"
#include "brahma/qwen2_runtime.hpp"
#include "brahma/tensor_registry.hpp"
#include "brahma/tensor_validator.hpp"

#include <cstdint>
#include <chrono>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
#include <utility>

namespace {
void print_usage() {
    std::cerr
        << "Usage:\n"
        << "  brahma-engine <model.gguf>\n"
        << "  brahma-engine <model.gguf> --list-tensors\n"
        << "  brahma-engine <model.gguf> --validate-qwen2\n"
        << "  brahma-engine <model.gguf> --generate \"prompt\" [max-new-tokens]\n"
        << "  brahma-engine <model.gguf> --tokenize \"text\"\n"
        << "  brahma-engine <model.gguf> --logits \"id,id,id\"\n"
        << "Native Qwen2 CPU inference supports F32, F16, Q4_K and Q6_K tensors.\n";
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
    if (argc < 2 || argc > 5) { print_usage(); return 2; }
    const std::string mode = argc >= 3 ? argv[2] : "";
    if (mode == "--generate" || mode == "--tokenize" || mode == "--logits") {
        if ((mode == "--generate" && argc != 4 && argc != 5) || ((mode == "--tokenize" || mode == "--logits") && argc != 4)) { print_usage(); return 2; }
    } else if ((argc != 2 && argc != 3) ||
               (!mode.empty() && mode != "--list-tensors" && mode != "--validate-qwen2")) {
        print_usage(); return 2;
    }

    try {
        if (mode == "--tokenize" || mode == "--logits") {
            brahma::Qwen2Runtime runtime(argv[1]);
            if (mode == "--logits") {
                std::vector<std::uint32_t> ids;
                std::stringstream input(argv[3]);
                std::string part;
                while (std::getline(input, part, ',')) {
                    if (part.empty()) throw std::runtime_error("Empty token ID in --logits list");
                    const auto value = std::stoull(part);
                    if (value > UINT32_MAX) throw std::runtime_error("Token ID exceeds uint32 range");
                    ids.push_back(static_cast<std::uint32_t>(value));
                }
                const auto values = runtime.logits(ids);
                std::cout.precision(9);
                for (std::size_t i = 0; i < values.size(); ++i) std::cout << i << ',' << values[i] << '\n';
                return 0;
            }
            const auto ids = runtime.tokenize(argv[3]);
            std::cout << "Tokenizer IDs (" << ids.size() << " tokens):";
            for (const auto id : ids) std::cout << ' ' << id;
            std::cout << "\nBackend: " << brahma::Qwen2Runtime::parallel_backend() << "\n";
            return 0;
        }
        if (mode == "--generate") {
            const auto parsed = argc == 5 ? std::stoull(argv[4]) : 32ULL;
            if (parsed > 4096) throw std::runtime_error("max-new-tokens must not exceed 4096");
            const auto load_start = std::chrono::steady_clock::now();
            brahma::Qwen2Runtime runtime(argv[1]);
            const double load_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count();
            const auto text = runtime.generate(argv[3], static_cast<std::size_t>(parsed));
            std::cout << text << "\n";
            const auto& stats = runtime.last_generation_stats();
            std::cerr << "Performance stats:\n"
                      << "Model construction: " << load_seconds << " s (includes GGUF inspection; weights are loaded lazily)\n"
                      << "Parallel backend: " << brahma::Qwen2Runtime::parallel_backend() << "\n"
                      << "Prompt tokens: " << stats.prompt_tokens << "\n"
                      << "Generated tokens: " << stats.generated_tokens << "\n"
                      << "Prompt prefill: " << stats.prefill_seconds << " s";
            if (stats.prompt_tokens > 0 && stats.prefill_seconds > 0.0)
                std::cerr << " (" << static_cast<double>(stats.prompt_tokens) / stats.prefill_seconds
                          << " prompt tokens/s)";
            std::cerr << "\n"
                      << "Decode loop: " << stats.decode_seconds << " s\n"
                      << "Operator totals across prefill + decode (overlap-free CPU wall time):\n"
                      << "  Projections: " << stats.projection_seconds << " s\n"
                      << "  RMSNorm: " << stats.normalization_seconds << " s\n"
                      << "  Attention core: " << stats.attention_seconds << " s";
            if (stats.generated_tokens > 0 && stats.decode_seconds > 0.0)
                std::cerr << " (" << static_cast<double>(stats.generated_tokens) / stats.decode_seconds
                          << " generated tokens/s)";
            std::cerr << "\n";
            return 0;
        }
        if (mode == "--list-tensors") {
            print_tensor_list(registry);
            return 0;
        }
        auto model = brahma::GgufReader::inspect(argv[1]);
        brahma::TensorRegistry registry(std::move(model), argv[1]);

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
            if (bytes > UINT64_MAX - total) throw std::runtime_error("Total tensor byte count overflow");
            total += bytes;
        }
        const auto& info = registry.model();
        std::cout << "Brahma Engine Core 0.4\n"
                  << "GGUF version: " << info.version << "\n"
                  << "Architecture: " << (info.architecture().empty() ? "(not declared)" : info.architecture()) << "\n"
                  << "Metadata entries: " << info.metadata.size() << "\n"
                  << "Tensor descriptors: " << info.tensors.size() << "\n"
                  << "Tensor data offset: " << info.tensor_data_offset << "\n"
                  << "Alignment: " << info.alignment << "\n"
                  << "Vocabulary entries: " << info.vocabulary_size() << "\n"
                  << "Payload validation: PASS\n"
                  << "Validated tensor payload bytes: " << total << "\n"
                  << "Inference: use --generate for native Qwen2 generation\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Brahma Engine error: " << error.what() << "\n";
        return 1;
    }
}
