#pragma once
#include "brahma/gguf_reader.hpp"
#include "brahma/tensor_validator.hpp"
#include "brahma/tensor_loader.hpp"
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace brahma {
// Metadata/layout index for a GGUF file. Payloads are read only on demand.
class TensorRegistry {
public:
    TensorRegistry(GgufModelInfo model, std::filesystem::path path);

    [[nodiscard]] const GgufModelInfo& model() const noexcept { return model_; }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
    [[nodiscard]] bool contains(const std::string& name) const noexcept;
    [[nodiscard]] const TensorInfo& info(const std::string& name) const;
    [[nodiscard]] const TensorLayout& layout(const std::string& name) const;
    [[nodiscard]] std::vector<std::string> names() const;
    [[nodiscard]] LoadedTensor load(const std::string& name) const;

private:
    GgufModelInfo model_;
    std::filesystem::path path_;
    std::vector<TensorLayout> layouts_;
    std::unordered_map<std::string, std::size_t> indices_;
};
} // namespace brahma
