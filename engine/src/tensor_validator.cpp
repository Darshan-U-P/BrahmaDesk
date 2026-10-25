#include "brahma/tensor_validator.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace brahma {
namespace {
struct Block { std::uint64_t elements; std::uint64_t bytes; const char* name; };
// GGML type IDs and serialized block sizes from ggml's public tensor type definitions.
constexpr Block kTypes[] = {
    {1, 4, "F32"}, {1, 2, "F16"}, {32, 18, "Q4_0"}, {32, 20, "Q4_1"},
    {32, 20, "Q4_2"}, {32, 20, "Q4_3"}, {32, 22, "Q5_0"}, {32, 24, "Q5_1"},
    {32, 34, "Q8_0"}, {1, 1, "I8"}, {256, 84, "Q2_K"}, {256, 110, "Q3_K"},
    {256, 144, "Q4_K"}, {256, 176, "Q5_K"}, {256, 210, "Q6_K"},
    {256, 292, "Q8_K"}, {1, 4, "I32"}, {1, 8, "I64"}, {1, 8, "F64"},
    {1, 2, "I16"}, {256, 66, "IQ2_XXS"}, {256, 74, "IQ2_XS"},
    {256,  98, "IQ3_XXS"}, {256, 98, "IQ1_S"}, {256, 50, "IQ4_NL"},
    {256,  110, "IQ3_S"}, {256, 82, "IQ2_S"}, {256, 136, "IQ4_XS"}
};
const Block* find_type(std::uint32_t id) {
    // Explicit switch keeps unknown/new GGML type IDs from being mistaken for valid layouts.
    switch (id) {
        case 0: return &kTypes[0]; case 1: return &kTypes[1];
        case 2: return &kTypes[2]; case 3: return &kTypes[3];
        case 6: return &kTypes[6]; case 7: return &kTypes[7];
        case 8: return &kTypes[8]; case 9: return &kTypes[9];
        case 10: return &kTypes[10]; case 11: return &kTypes[11];
        case 12: return &kTypes[12]; case 13: return &kTypes[13];
        case 14: return &kTypes[14]; case 15: return &kTypes[15];
        case 16: return &kTypes[16]; case 17: return &kTypes[17];
        case 18: return &kTypes[18]; case 19: return &kTypes[19];
        default: return nullptr;
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
        case 0:return "F32"; case 1:return "F16"; case 2:return "Q4_0";
        case 3:return "Q4_1"; case 6:return "Q5_0"; case 7:return "Q5_1";
        case 8:return "Q8_0"; case 9:return "I8"; case 10:return "Q2_K";
        case 11:return "Q3_K"; case 12:return "Q4_K"; case 13:return "Q5_K";
        case 14:return "Q6_K"; case 15:return "Q8_K"; case 16:return "I32";
        case 17:return "I64"; case 18:return "F64"; case 19:return "I16";
        default:return "UNKNOWN";
    }
}
TensorLayout tensor_layout(const TensorInfo& t, std::uint64_t data_offset, std::uint64_t file_size) {
    const auto* type = find_type(t.ggml_type);
    if (!type) throw std::runtime_error("Unsupported GGML tensor type " + std::to_string(t.ggml_type) + " for tensor " + t.name);
    if (t.dimensions.empty()) throw std::runtime_error("Tensor has no dimensions: " + t.name);
    std::uint64_t elements = 1;
    for (const auto dim : t.dimensions) {
        if (!dim) throw std::runtime_error("Zero tensor dimension: " + t.name);
        elements = checked_mul(elements, dim, "tensor element count");
    }
    if (elements % type->elements)
        throw std::runtime_error("Tensor element count is not divisible by quantization block size: " + t.name);
    const auto bytes = checked_mul(elements / type->elements, type->bytes, "tensor payload bytes");
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
