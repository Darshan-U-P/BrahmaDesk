#include "brahma/tensor_loader.hpp"
#include <fstream>
#include <limits>
#include <stdexcept>

namespace brahma {
LoadedTensor load_gguf_tensor(
    const GgufModelInfo& model,
    const std::filesystem::path& path,
    const std::string& tensor_name) {
    const TensorInfo* selected = nullptr;
    std::size_t selected_index = 0;
    for (std::size_t i = 0; i < model.tensors.size(); ++i) {
        if (model.tensors[i].name == tensor_name) {
            if (selected) throw std::runtime_error("Duplicate tensor name: " + tensor_name);
            selected = &model.tensors[i];
            selected_index = i;
        }
    }
    if (!selected) throw std::runtime_error("Tensor not found: " + tensor_name);

    const auto layouts = validate_tensor_payloads(model, path);
    const auto& layout = layouts.at(selected_index);
    if (layout.bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        layout.bytes > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("Tensor payload is too large to load into memory: " + tensor_name);
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open GGUF file: " + path.string());
    if (layout.absolute_offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error("Tensor file offset is too large: " + tensor_name);
    }
    file.seekg(static_cast<std::streamoff>(layout.absolute_offset), std::ios::beg);
    if (!file) throw std::runtime_error("Cannot seek to tensor payload: " + tensor_name);

    LoadedTensor result{*selected, layout, std::vector<std::uint8_t>(static_cast<std::size_t>(layout.bytes))};
    if (!result.payload.empty()) {
        file.read(reinterpret_cast<char*>(result.payload.data()),
                  static_cast<std::streamsize>(result.payload.size()));
        if (!file || static_cast<std::size_t>(file.gcount()) != result.payload.size()) {
            throw std::runtime_error("Failed to read complete tensor payload: " + tensor_name);
        }
    }
    return result;
}
} // namespace brahma
