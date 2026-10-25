#include "brahma/tensor_registry.hpp"
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace brahma {
TensorRegistry::TensorRegistry(GgufModelInfo model, std::filesystem::path path)
    : model_(std::move(model)), path_(std::move(path)) {
    layouts_ = validate_tensor_payloads(model_, path_);
    if (layouts_.size() != model_.tensors.size()) {
        throw std::runtime_error("Tensor registry layout count mismatch");
    }
    indices_.reserve(model_.tensors.size());
    for (std::size_t i = 0; i < model_.tensors.size(); ++i) {
        const auto& name = model_.tensors[i].name;
        if (!indices_.emplace(name, i).second) {
            throw std::runtime_error("Duplicate tensor name in GGUF: " + name);
        }
    }
}

bool TensorRegistry::contains(const std::string& name) const noexcept {
    return indices_.find(name) != indices_.end();
}

const TensorInfo& TensorRegistry::info(const std::string& name) const {
    const auto it = indices_.find(name);
    if (it == indices_.end()) throw std::runtime_error("Tensor not found: " + name);
    return model_.tensors[it->second];
}

const TensorLayout& TensorRegistry::layout(const std::string& name) const {
    const auto it = indices_.find(name);
    if (it == indices_.end()) throw std::runtime_error("Tensor not found: " + name);
    return layouts_[it->second];
}

std::vector<std::string> TensorRegistry::names() const {
    std::vector<std::string> result;
    result.reserve(model_.tensors.size());
    for (const auto& tensor : model_.tensors) result.push_back(tensor.name);
    return result;
}

LoadedTensor TensorRegistry::load(const std::string& name) const {
    const auto it = indices_.find(name);
    if (it == indices_.end()) throw std::runtime_error("Tensor not found: " + name);
    const auto index = it->second;
    const auto& tensor_layout = layouts_[index];
    // Recheck the current file size and perform a bounded read of only this tensor.
    const auto current_size = std::filesystem::file_size(path_);
    if (tensor_layout.absolute_offset > current_size ||
        tensor_layout.bytes > current_size - tensor_layout.absolute_offset) {
        throw std::runtime_error("Tensor payload no longer fits file: " + name);
    }
    if (tensor_layout.bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        tensor_layout.bytes > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("Tensor payload is too large to load into memory: " + name);
    }
    std::ifstream file(path_, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open GGUF file: " + path_.string());
    if (tensor_layout.absolute_offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error("Tensor file offset is too large: " + name);
    }
    file.seekg(static_cast<std::streamoff>(tensor_layout.absolute_offset), std::ios::beg);
    if (!file) throw std::runtime_error("Cannot seek to tensor payload: " + name);
    LoadedTensor result{model_.tensors[index], tensor_layout,
        std::vector<std::uint8_t>(static_cast<std::size_t>(tensor_layout.bytes))};
    if (!result.payload.empty()) {
        file.read(reinterpret_cast<char*>(result.payload.data()),
                  static_cast<std::streamsize>(result.payload.size()));
        if (!file || static_cast<std::size_t>(file.gcount()) != result.payload.size()) {
            throw std::runtime_error("Failed to read complete tensor payload: " + name);
        }
    }
    return result;
}
} // namespace brahma
