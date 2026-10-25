
#include "llama.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kContextSize = 2048;
constexpr std::uint32_t kBatchSize = 512;
constexpr int kDefaultMaxTokens = 64;
constexpr int kMaxAllowedTokens = 4096;

bool format_chat_prompt(
    const llama_model* model,
    const std::string& user_prompt,
    std::string& formatted_prompt) {

    const char* tmpl = llama_model_chat_template(model, nullptr);

    if (tmpl == nullptr || *tmpl == '\0') {
        std::cerr
            << "Brahma Engine: model does not provide a chat template.\n";
        return false;
    }

    llama_chat_message message{
        "user",
        user_prompt.c_str()
    };

    // Start with a modest buffer. The API returns the required size
    // when the provided buffer is too small.
    std::vector<char> buffer(user_prompt.size() * 4 + 1024);

    int32_t length = llama_chat_apply_template(
        tmpl,
        &message,
        1,
        true,
        buffer.data(),
        static_cast<int32_t>(buffer.size()));

    if (length < 0) {
        std::cerr
            << "Brahma Engine: chat-template formatting failed.\n";
        return false;
    }

    if (static_cast<std::size_t>(length) >= buffer.size()) {
        // Allocate space for the reported output and its terminating
        // null byte, then apply the template again.
        buffer.resize(static_cast<std::size_t>(length) + 1);

        length = llama_chat_apply_template(
            tmpl,
            &message,
            1,
            true,
            buffer.data(),
            static_cast<int32_t>(buffer.size()));

        if (length < 0 ||
            static_cast<std::size_t>(length) >= buffer.size()) {
            std::cerr
                << "Brahma Engine: chat-template formatting failed "
                   "after resizing the buffer.\n";
            return false;
        }
    }

    formatted_prompt.assign(
        buffer.data(),
        static_cast<std::size_t>(length));

    return !formatted_prompt.empty();
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr
            << "Brahma Engine 0.1.0\n"
            << "Usage: brahma-engine <model.gguf> [prompt] [max-tokens]\n";
        return 2;
    }

    const std::string model_path = argv[1];
    const std::string user_prompt =
        argc >= 3 ? argv[2] : "Hello from Brahma Engine.";

    int max_tokens = kDefaultMaxTokens;

    try {
        if (argc >= 4) {
            std::size_t parsed = 0;
            const int parsed_tokens = std::stoi(argv[3], &parsed);

            if (parsed != std::string(argv[3]).size()) {
                std::cerr << "Brahma Engine: invalid max-tokens value.\n";
                return 2;
            }

            max_tokens = parsed_tokens;
        }
    } catch (const std::exception&) {
        std::cerr << "Brahma Engine: invalid max-tokens value.\n";
        return 2;
    }

    if (max_tokens < 1 || max_tokens > kMaxAllowedTokens) {
        std::cerr << "max-tokens must be between 1 and "
                  << kMaxAllowedTokens << ".\n";
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
    context_params.n_ctx = kContextSize;
    context_params.n_batch = kBatchSize;

    std::unique_ptr<llama_context, decltype(&llama_free)> context(
        llama_init_from_model(model.get(), context_params),
        llama_free);

    if (!context) {
        std::cerr << "Brahma Engine: context creation failed.\n";
        llama_backend_free();
        return 1;
    }

    std::string formatted_prompt;

    if (!format_chat_prompt(
            model.get(),
            user_prompt,
            formatted_prompt)) {
        llama_backend_free();
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model.get());

    std::vector<llama_token> tokens(formatted_prompt.size() + 128);

    int32_t token_count = llama_tokenize(
        vocab,
        formatted_prompt.c_str(),
        static_cast<int32_t>(formatted_prompt.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        true,
        true);

    if (token_count < 0) {
        tokens.resize(static_cast<std::size_t>(-token_count));

        token_count = llama_tokenize(
            vocab,
            formatted_prompt.c_str(),
            static_cast<int32_t>(formatted_prompt.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            true,
            true);
    }

    if (token_count <= 0) {
        std::cerr << "Brahma Engine: prompt tokenization failed.\n";
        llama_backend_free();
        return 1;
    }

    tokens.resize(static_cast<std::size_t>(token_count));

    if (tokens.size() >= context_params.n_ctx) {
        std::cerr
            << "Brahma Engine: prompt exceeds the configured context.\n";
        llama_backend_free();
        return 1;
    }

    if (llama_decode(
            context.get(),
            llama_batch_get_one(
                tokens.data(),
                static_cast<int32_t>(tokens.size()))) != 0) {
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

    llama_sampler_chain_add(
        sampler.get(),
        llama_sampler_init_greedy());

    std::cout
        << "Brahma Engine 0.2.0 | llama.cpp CPU inference\n"
        << "Model: " << model_path << "\n\n";

    for (int i = 0; i < max_tokens; ++i) {
        const llama_token token =
            llama_sampler_sample(sampler.get(), context.get(), -1);

        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        char piece[256];

        const int32_t piece_size = llama_token_to_piece(
            vocab,
            token,
            piece,
            sizeof(piece),
            0,
            false);

        if (piece_size < 0) {
            std::cerr
                << "\nBrahma Engine: output token piece exceeded "
                   "the output buffer.\n";
            llama_backend_free();
            return 1;
        }

        if (piece_size > 0) {
            std::cout.write(piece, piece_size);
            std::cout.flush();
        }

        if (i + 1 < max_tokens) {
            llama_token next = token;

            if (llama_decode(
                    context.get(),
                    llama_batch_get_one(&next, 1)) != 0) {
                std::cerr
                    << "\nBrahma Engine: generation stopped after "
                       "a decode error.\n";
                llama_backend_free();
                return 1;
            }
        }
    }

    std::cout << '\n';
    llama_backend_free();
    return 0;
}
