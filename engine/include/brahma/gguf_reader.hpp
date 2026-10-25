#pragma once
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <variant>
#include <vector>
namespace brahma {
using MetadataValue = std::variant<std::uint64_t, std::int64_t, double, bool, std::string,
 std::vector<std::uint64_t>, std::vector<std::int64_t>, std::vector<double>, std::vector<bool>, std::vector<std::string>>;
struct TensorInfo { std::string name; std::vector<std::uint64_t> dimensions; std::uint32_t ggml_type{}; std::uint64_t relative_offset{}; };
struct GgufModelInfo {
 std::uint32_t version{}; std::uint64_t alignment{32}; std::uint64_t tensor_data_offset{};
 std::map<std::string, MetadataValue> metadata; std::vector<TensorInfo> tensors;
 [[nodiscard]] std::string architecture() const;
 [[nodiscard]] std::uint64_t vocabulary_size() const;
};
class GgufReader {
public:
 // Reads GGUF metadata and tensor descriptors only; tensor payloads are not loaded.
 [[nodiscard]] static GgufModelInfo inspect(const std::filesystem::path& path);
};
} // namespace brahma
