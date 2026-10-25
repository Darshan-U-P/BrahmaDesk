#pragma once
#include "brahma/gguf_reader.hpp"
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace brahma {
struct TensorLayout {
    std::uint64_t elements{};
    std::uint64_t bytes{};
    std::uint64_t absolute_offset{};
};

// Returns GGML block layout for types supported by the initial native CPU runtime.
// Unknown types fail closed instead of silently mis-sizing a tensor.
[[nodiscard]] TensorLayout tensor_layout(const TensorInfo& tensor,
                                         std::uint64_t data_offset,
                                         std::uint64_t file_size);
[[nodiscard]] std::vector<TensorLayout> validate_tensor_payloads(
    const GgufModelInfo& model, const std::filesystem::path& path);
[[nodiscard]] const char* ggml_type_name(std::uint32_t type) noexcept;
} // namespace brahma
