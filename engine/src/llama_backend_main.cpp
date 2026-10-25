#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "Brahma Engine 0.1.0\n"
                  << "Usage: brahma-engine <model.gguf> [prompt] [max-tokens]\n";
        return 2;
    }

    const std::string model_path = argv[1];
    const std::string prompt =
        argc >= 3 ? argv[2] : "Hello from Brahma Engine.";
    const int max_tokens = argc >= 4 ? std::stoi(argv[3]) : 64;

    if (max_tokens < 1 || max_tokens > 4096) {
        std::cerr << "max-tokens must be between 1 and 4096.\n";
        return 2;
    }

    llama_backend_init();

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(model_path.c_str(), model_params),
        llama_model_free);

    if (!model) {
        std::cerr << "Brahma Engine: model loading failed.\n";
        llama_backend_free();
        return 1;
    }

    auto context_params = llama_context_default_params();
    context_params.n_ctx = 2048;
    context_params.n_batch = 512;

    std::unique_ptr<llama_context, decltype(&llama_free)> context(
        llama_init_from_model(model.get(), context_params), llama_free);

    if (!context) {
        std::cerr << "Brahma Engine: context creation failed.\n";
        llama_backend_free();
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model.get());

    std::vector<llama_token> tokens(prompt.size() + 128);
    int32_t token_count = llama_tokenize(
        vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
        tokens.data(), static_cast<int32_t>(tokens.size()), true, true);

    if (token_count < 0) {
        tokens.resize(static_cast<std::size_t>(-token_count));
        token_count = llama_tokenize(
            vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
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

    if (llama_decode(context.get(),
            llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()))) != 0) {
        std::cerr << "Brahma Engine: prompt evaluation failed.\n";
        llama_backend_free();
        return 1;
    }

    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
        llama_sampler_chain_init(llama_sampler_chain_default_params()),
        llama_sampler_free);

    if (!sampler) {
        std::cerr << "Brahma Engine: sampler initialization failed.\n";
        llama_backend_free();
        return 1;
    }

    llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());

    std::cout << "Brahma Engine 0.1.0 | CPU inference\n"
              << "Model: " << model_path << "\n\n";

    for (int i = 0; i < max_tokens; ++i) {
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

        if (i + 1 < max_tokens) {
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
