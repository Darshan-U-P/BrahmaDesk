#include "llama.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

struct Options {
    std::string model_path;
    std::string prompt = "Hello from Brahma Engine.";
    int max_tokens = 64;
    uint32_t context_size = 2048;
    uint32_t batch_size = 512;
    int32_t threads = 0;
    int32_t gpu_layers = 0;
    float temperature = 0.0f;
    int32_t top_k = 40;
    float top_p = 0.95f;
};

void print_usage(std::ostream& output) {
    output << "Brahma Engine 0.1.0\n"
           << "Usage: brahma-engine --model <path> [options]\n"
           << "       brahma-engine <model.gguf> [prompt] [max-tokens]\n\n"
           << "Options:\n"
           << "  --model <path>       GGUF model path (required)\n"
           << "  --prompt <text>      Input prompt (default: \"Hello from Brahma Engine.\")\n"
           << "  --max-tokens <n>     Maximum generated tokens (default: 64; range: 1-4096)\n"
           << "  --ctx-size <n>       Context window (default: 2048)\n"
           << "  --batch-size <n>     Prompt processing batch (default: 512)\n"
           << "  --threads <n>        CPU inference threads (default: backend)\n"
           << "  --gpu-layers <n>     Layers to offload to GPU (default: 0)\n"
           << "  --temperature <n>    Sampling temperature (default: 0, greedy)\n"
           << "  --top-k <n>          Top-K sampling (default: 40)\n"
           << "  --top-p <n>          Top-P sampling (default: 0.95)\n"
           << "  --help               Print this help\n";
}

bool parse_integer(std::string_view text, int64_t& value) {
    if (text.empty()) {
        return false;
    }

    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto result = std::from_chars(begin, end, value);
    return result.ec == std::errc{} && result.ptr == end;
}

bool parse_float(std::string_view text, float& value) {
    if (text.empty()) {
        return false;
    }

    try {
        std::size_t parsed = 0;
        value = std::stof(std::string(text), &parsed);
        return parsed == text.size() && std::isfinite(value);
    } catch (const std::exception&) {
        return false;
    }
}

bool parse_options(int argc, char** argv, Options& options, bool& help_requested) {
    bool model_set = false;
    bool prompt_set = false;
    bool max_tokens_set = false;
    int positional_count = 0;

    auto set_integer = [](std::string_view value, int64_t min, int64_t max,
                          int64_t& output) {
        return parse_integer(value, output) && output >= min && output <= max;
    };

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--help") {
            print_usage(std::cout);
            help_requested = true;
            return false;
        }

        if (arg == "--model" || arg == "--prompt" || arg == "--max-tokens" ||
            arg == "--ctx-size" || arg == "--batch-size" ||
            arg == "--threads" || arg == "--gpu-layers" ||
            arg == "--temperature" || arg == "--top-k" || arg == "--top-p") {
            if (i + 1 >= argc) {
                std::cerr << "Brahma Engine: missing value for " << arg << ".\n";
                return false;
            }

            const std::string_view value(argv[++i]);
            int64_t integer = 0;
            if (arg == "--model") {
                if (model_set || value.empty()) {
                    std::cerr << "Brahma Engine: --model must be provided once with a non-empty path.\n";
                    return false;
                }
                options.model_path.assign(value);
                model_set = true;
            } else if (arg == "--prompt") {
                if (prompt_set) {
                    std::cerr << "Brahma Engine: --prompt may only be provided once.\n";
                    return false;
                }
                options.prompt.assign(value);
                prompt_set = true;
            } else if (arg == "--max-tokens") {
                if (max_tokens_set ||
                    !set_integer(value, 1, 4096, integer)) {
                    std::cerr << "Brahma Engine: --max-tokens must be an integer between 1 and 4096.\n";
                    return false;
                }
                options.max_tokens = static_cast<int>(integer);
                max_tokens_set = true;
            } else if (arg == "--ctx-size") {
                if (!set_integer(value, 1, std::numeric_limits<uint32_t>::max(), integer)) {
                    std::cerr << "Brahma Engine: --ctx-size must be a positive 32-bit integer.\n";
                    return false;
                }
                options.context_size = static_cast<uint32_t>(integer);
            } else if (arg == "--batch-size") {
                if (!set_integer(value, 1, std::numeric_limits<int32_t>::max(), integer)) {
                    std::cerr << "Brahma Engine: --batch-size must be a positive 32-bit integer.\n";
                    return false;
                }
                options.batch_size = static_cast<uint32_t>(integer);
            } else if (arg == "--threads") {
                if (!set_integer(value, 1, std::numeric_limits<int32_t>::max(), integer)) {
                    std::cerr << "Brahma Engine: --threads must be a positive 32-bit integer.\n";
                    return false;
                }
                options.threads = static_cast<int32_t>(integer);
            } else if (arg == "--gpu-layers") {
                if (!set_integer(value, -1, std::numeric_limits<int32_t>::max(), integer)) {
                    std::cerr << "Brahma Engine: --gpu-layers must be -1 or a non-negative 32-bit integer.\n";
                    return false;
                }
                options.gpu_layers = static_cast<int32_t>(integer);
            } else if (arg == "--temperature") {
                if (!parse_float(value, options.temperature) || options.temperature < 0.0f) {
                    std::cerr << "Brahma Engine: --temperature must be a finite non-negative number.\n";
                    return false;
                }
            } else if (arg == "--top-k") {
                if (!set_integer(value, 0, std::numeric_limits<int32_t>::max(), integer)) {
                    std::cerr << "Brahma Engine: --top-k must be a non-negative 32-bit integer.\n";
                    return false;
                }
                options.top_k = static_cast<int32_t>(integer);
            } else if (arg == "--top-p") {
                if (!parse_float(value, options.top_p) ||
                    options.top_p <= 0.0f || options.top_p > 1.0f) {
                    std::cerr << "Brahma Engine: --top-p must be greater than 0 and at most 1.\n";
                    return false;
                }
            }
            continue;
        }

        if (!arg.empty() && arg.front() == '-') {
            std::cerr << "Brahma Engine: unknown option " << arg << ".\n";
            return false;
        }

        if (positional_count == 0 && !model_set) {
            options.model_path.assign(arg);
            model_set = true;
        } else if (positional_count <= 1 && !prompt_set) {
            options.prompt.assign(arg);
            prompt_set = true;
        } else if (positional_count <= 2 && !max_tokens_set) {
            int64_t integer = 0;
            if (!set_integer(arg, 1, 4096, integer)) {
                std::cerr << "Brahma Engine: positional max-tokens must be an integer between 1 and 4096.\n";
                return false;
            }
            options.max_tokens = static_cast<int>(integer);
            max_tokens_set = true;
        } else {
            std::cerr << "Brahma Engine: unexpected positional argument " << arg << ".\n";
            return false;
        }
        ++positional_count;
    }

    if (!model_set) {
        std::cerr << "Brahma Engine: a model path is required.\n";
        print_usage(std::cerr);
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    bool help_requested = false;
    if (!parse_options(argc, argv, options, help_requested)) {
        return help_requested ? 0 : 2;
    }

    llama_backend_init();

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = options.gpu_layers;

    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(options.model_path.c_str(), model_params),
        llama_model_free);

    if (!model) {
        std::cerr << "Brahma Engine: model loading failed.\n";
        llama_backend_free();
        return 1;
    }

    auto context_params = llama_context_default_params();
    context_params.n_ctx = options.context_size;
    context_params.n_batch = options.batch_size;
    if (options.threads > 0) {
        context_params.n_threads = options.threads;
        context_params.n_threads_batch = options.threads;
    }

    std::unique_ptr<llama_context, decltype(&llama_free)> context(
        llama_init_from_model(model.get(), context_params), llama_free);

    if (!context) {
        std::cerr << "Brahma Engine: context creation failed.\n";
        llama_backend_free();
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model.get());

    std::vector<llama_token> tokens(options.prompt.size() + 128);
    int32_t token_count = llama_tokenize(
        vocab, options.prompt.c_str(), static_cast<int32_t>(options.prompt.size()),
        tokens.data(), static_cast<int32_t>(tokens.size()), true, true);

    if (token_count < 0) {
        tokens.resize(static_cast<std::size_t>(-token_count));
        token_count = llama_tokenize(
            vocab, options.prompt.c_str(), static_cast<int32_t>(options.prompt.size()),
            tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
    }

    if (token_count <= 0) {
        std::cerr << "Brahma Engine: prompt tokenization failed.\n";
        llama_backend_free();
        return 1;
    }

    tokens.resize(static_cast<std::size_t>(token_count));

    if (tokens.size() >= context_params.n_ctx) {
        std::cerr << "Brahma Engine: prompt exceeds the configured context.\n";
        llama_backend_free();
        return 1;
    }

    for (std::size_t offset = 0; offset < tokens.size();) {
        const std::size_t count = std::min(
            static_cast<std::size_t>(context_params.n_batch), tokens.size() - offset);
        if (llama_decode(context.get(), llama_batch_get_one(
                tokens.data() + offset, static_cast<int32_t>(count))) != 0) {
            std::cerr << "Brahma Engine: prompt evaluation failed.\n";
            llama_backend_free();
            return 1;
        }
        offset += count;
    }

    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
        llama_sampler_chain_init(llama_sampler_chain_default_params()),
        llama_sampler_free);

    if (!sampler) {
        std::cerr << "Brahma Engine: sampler initialization failed.\n";
        llama_backend_free();
        return 1;
    }

    if (options.temperature == 0.0f) {
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(options.top_k));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(options.top_p, 1));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(options.temperature));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    const std::size_t generation_limit = std::min(
        static_cast<std::size_t>(options.max_tokens),
        static_cast<std::size_t>(context_params.n_ctx) - tokens.size());

    std::cout << "Brahma Engine 0.1.0 | llama.cpp backend\n"
              << "Model: " << options.model_path << "\n\n";

    for (std::size_t i = 0; i < generation_limit; ++i) {
        const llama_token token = llama_sampler_sample(sampler.get(), context.get(), -1);

        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        char piece[256];
        const int32_t piece_size = llama_token_to_piece(
            vocab, token, piece, sizeof(piece), 0, false);

        if (piece_size > 0) {
            std::cout.write(piece, piece_size);
            std::cout.flush();
        }

        if (i + 1 < generation_limit) {
            llama_token next = token;
            if (llama_decode(context.get(), llama_batch_get_one(&next, 1)) != 0) {
                std::cerr << "\nBrahma Engine: generation stopped after a decode error.\n";
                llama_backend_free();
                return 1;
            }
        }
    }

    std::cout << '\n';
    llama_backend_free();
    return 0;
}
