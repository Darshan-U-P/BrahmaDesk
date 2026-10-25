#include "brahma/tensor_validator.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace brahma {
namespace {
struct Block { std::uint64_t elements; std::uint64_t bytes; const char* name; };
Block block_for(std::uint32_t id) {
    switch (id) {
        case 0: return {1,4,"F32"}; case 1: return {1,2,"F16"};
        case 2: return {32,18,"Q4_0"}; case 3: return {32,20,"Q4_1"};
        case 6: return {32,22,"Q5_0"}; case 7: return {32,24,"Q5_1"};
        case 8: return {32,34,"Q8_0"};
        case 10: return {256,84,"Q2_K"}; case 11: return {256,110,"Q3_K"};
        case 12: return {256,144,"Q4_K"}; case 13: return {256,176,"Q5_K"};
        case 14: return {256,210,"Q6_K"}; case 15: return {256,292,"Q8_K"};
        case 24: return {1,1,"I8"}; case 25: return {1,2,"I16"};
        case 26: return {1,4,"I32"}; case 27: return {1,8,"I64"};
        case 28: return {1,8,"F64"};
        default: throw std::runtime_error("Unsupported GGML tensor type " + std::to_string(id) + " in native payload validator");
    }
}
std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, const char* what) {
    if (b && a > std::numeric_limits<std::uint64_t>::max() / b)
        throw std::runtime_error(std::string("Overflow computing ") + what);
    return a * b;
}
}
const char* ggml_type_name(std::uint32_t id) noexcept {
    switch (id) {
        case 0:return "F32"; case 1:return "F16"; case 2:return "Q4_0"; case 3:return "Q4_1";
        case 6:return "Q5_0"; case 7:return "Q5_1"; case 8:return "Q8_0";
        case 10:return "Q2_K"; case 11:return "Q3_K"; case 12:return "Q4_K";
        case 13:return "Q5_K"; case 14:return "Q6_K"; case 15:return "Q8_K";
        case 24:return "I8"; case 25:return "I16"; case 26:return "I32";
        case 27:return "I64"; case 28:return "F64"; default:return "UNKNOWN";
    }
}
TensorLayout tensor_layout(const TensorInfo& t, std::uint64_t data_offset, std::uint64_t file_size) {
    const auto type = block_for(t.ggml_type);
    if (t.dimensions.empty()) throw std::runtime_error("Tensor has no dimensions: " + t.name);
    std::uint64_t elements = 1;
    for (const auto dim : t.dimensions) {
        if (!dim) throw std::runtime_error("Zero tensor dimension: " + t.name);
        elements = checked_mul(elements, dim, "tensor element count");
    }
    if (elements % type.elements)
        throw std::runtime_error("Tensor element count is not divisible by quantization block size for " + t.name);
    const auto bytes = checked_mul(elements / type.elements, type.bytes, "tensor payload bytes");
    if (t.relative_offset > std::numeric_limits<std::uint64_t>::max() - data_offset)
        throw std::runtime_error("Tensor offset overflow: " + t.name);
    const auto absolute = data_offset + t.relative_offset;
    if (absolute > file_size || bytes > file_size - absolute)
        throw std::runtime_error("Tensor payload extends beyond file: " + t.name);
    return {elements, bytes, absolute};
}
std::vector<TensorLayout> validate_tensor_payloads(const GgufModelInfo& model, const std::filesystem::path& path) {
    const auto file_size = std::filesystem::file_size(path);
    if (model.tensor_data_offset > file_size) throw std::runtime_error("Tensor data offset exceeds file size");
    std::vector<TensorLayout> layouts;
    layouts.reserve(model.tensors.size());
    for (const auto& t : model.tensors) {
        if (model.alignment == 0 || t.relative_offset % model.alignment != 0)
            throw std::runtime_error("Tensor relative offset is not aligned: " + t.name);
        layouts.push_back(tensor_layout(t, model.tensor_data_offset, file_size));
    }
    std::vector<std::size_t> order(layouts.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](auto a, auto b) { return layouts[a].absolute_offset < layouts[b].absolute_offset; });
    for (std::size_t i = 1; i < order.size(); ++i) {
        const auto prev = order[i - 1], cur = order[i];
        if (layouts[prev].bytes > layouts[cur].absolute_offset - layouts[prev].absolute_offset)
            throw std::runtime_error("Tensor payloads overlap: " + model.tensors[cur].name);
    }
    return layouts;
}
} // namespace brahma
