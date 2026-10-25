#pragma once
#include "brahma/gguf_reader.hpp"
#include "brahma/tensor_validator.hpp"
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace brahma {
struct LoadedTensor {
    TensorInfo info;
    TensorLayout layout;
    // Raw serialized GGML bytes; quantized tensors remain quantized.
    std::vector<std::uint8_t> payload;
};

// Loads one named tensor from a GGUF file after validating tensor layouts.
// Throws if the name is absent, duplicated, unsupported, truncated, or invalid.
[[nodiscard]] LoadedTensor load_gguf_tensor(
    const GgufModelInfo& model,
    const std::filesystem::path& path,
    const std::string& tensor_name);
} // namespace brahma
