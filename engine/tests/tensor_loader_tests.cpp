#include "brahma/tensor_loader.hpp"
#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

template<class T> void write_value(std::ofstream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
void write_string(std::ofstream& out, const std::string& value) {
    write_value(out, static_cast<std::uint64_t>(value.size()));
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
int main() {
    const auto path = std::filesystem::temp_directory_path() / "brahma_tensor_loader_test.gguf";
    try {
        {
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out) throw std::runtime_error("cannot create synthetic GGUF");
            out.write("GGUF", 4);
            write_value(out, std::uint32_t{3});
            write_value(out, std::uint64_t{1}); // tensors
            write_value(out, std::uint64_t{2}); // metadata
            write_string(out, "general.architecture");
            write_value(out, std::uint32_t{8}); // string
            write_string(out, "llama");
            write_string(out, "general.alignment");
            write_value(out, std::uint32_t{4}); // u32
            write_value(out, std::uint32_t{32});
            write_string(out, "test.weight");
            write_value(out, std::uint32_t{2}); // dimensions
            write_value(out, std::uint64_t{2});
            write_value(out, std::uint64_t{2});
            write_value(out, std::uint32_t{0}); // F32
            write_value(out, std::uint64_t{0}); // relative offset
            const auto pos = static_cast<std::uint64_t>(out.tellp());
            const auto padding = (32 - pos % 32) % 32;
            for (std::uint64_t i = 0; i < padding; ++i) out.put('\0');
            const std::array<float, 4> values{1.25f, -2.0f, 3.5f, 4.0f};
            out.write(reinterpret_cast<const char*>(values.data()), sizeof(values));
        }
        const auto model = brahma::GgufReader::inspect(path);
        const auto loaded = brahma::load_gguf_tensor(model, path, "test.weight");
        require(loaded.info.name == "test.weight", "loaded tensor name");
        require(loaded.layout.elements == 4 && loaded.layout.bytes == 16, "loaded tensor layout");
        require(loaded.payload.size() == 16, "loaded payload byte count");
        std::array<float, 4> values{};
        for (std::size_t i = 0; i < values.size(); ++i) {
            const auto b = i * sizeof(float);
            const std::uint32_t bits =
                std::uint32_t(loaded.payload[b]) |
                (std::uint32_t(loaded.payload[b + 1]) << 8) |
                (std::uint32_t(loaded.payload[b + 2]) << 16) |
                (std::uint32_t(loaded.payload[b + 3]) << 24);
            values[i] = std::bit_cast<float>(bits);
        }
        require(values[0] == 1.25f && values[1] == -2.0f &&
                values[2] == 3.5f && values[3] == 4.0f, "loaded payload contents");
        bool rejected = false;
        try { (void)brahma::load_gguf_tensor(model, path, "missing.weight"); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "missing tensor name rejection");
        std::filesystem::remove(path);
        std::cout << "GGUF tensor loader tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove(path);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
