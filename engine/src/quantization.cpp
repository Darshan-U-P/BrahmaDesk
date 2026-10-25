#include "brahma/quantization.hpp"
#include <bit>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <string>

namespace brahma {
namespace {
float half_to_float(std::uint16_t h) noexcept {
    const std::uint32_t sign = (std::uint32_t(h & 0x8000u)) << 16;
    std::uint32_t exp = (h >> 10) & 0x1fu;
    std::uint32_t mant = h & 0x03ffu;
    std::uint32_t bits{};
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            int e = -14;
            while ((mant & 0x0400u) == 0) { mant <<= 1; --e; }
            mant &= 0x03ffu;
            bits = sign | (std::uint32_t(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    return std::bit_cast<float>(bits);
}
std::uint16_t u16le(std::span<const std::uint8_t,144> b, std::size_t i) noexcept {
    return std::uint16_t(b[i]) | (std::uint16_t(b[i+1]) << 8);
}
void get_scale_min(std::span<const std::uint8_t,144> b, int j, int& scale, int& min) noexcept {
    const auto* s = b.data() + 4;
    if (j < 4) {
        scale = s[j] & 0x3f;
        min = s[j+4] & 0x3f;
    } else {
        scale = (s[j+4] & 0x0f) | ((s[j-4] >> 6) << 4);
        min = (s[j+4] >> 4) | ((s[j] >> 6) << 4);
    }
}
}
void dequantize_q4_k_block(std::span<const std::uint8_t,144> b, std::span<float,256> out) {
    const float d = half_to_float(u16le(b,0));
    const float dmin = half_to_float(u16le(b,2));
    const auto* qs = b.data()+16;
    std::size_t out_pos=0;
    for (int group=0; group<4; ++group) {
        int s1{},m1{},s2{},m2{};
        get_scale_min(b,group*2,s1,m1);
        get_scale_min(b,group*2+1,s2,m2);
        for (int i=0;i<32;++i) {
            const auto q=qs[std::size_t(group)*32+std::size_t(i)];
            out[out_pos++] = d*float(s1)*float(q & 0x0f) - dmin*float(m1);
        }
        for (int i=0;i<32;++i) {
            const auto q=qs[std::size_t(group)*32+std::size_t(i)];
            out[out_pos++] = d*float(s2)*float(q >> 4) - dmin*float(m2);
        }
    }
    if (out_pos != out.size()) throw std::runtime_error("Internal Q4_K block size mismatch");
}
std::vector<float> dequantize_q4_k(std::span<const std::uint8_t> payload, std::size_t n) {
    if (n % 256 != 0) throw std::runtime_error("Q4_K element count must be a multiple of 256");
    if (n/256 > std::numeric_limits<std::size_t>::max()/144 || payload.size() != (n/256)*144)
        throw std::runtime_error("Q4_K payload size does not match requested element count");
    std::vector<float> result(n);
    for (std::size_t block=0; block<n/256; ++block) {
        const auto bytes=std::span<const std::uint8_t,144>(payload.data()+block*144,144);
        const auto values=std::span<float,256>(result.data()+block*256,256);
        dequantize_q4_k_block(bytes,values);
    }
    return result;
}
} // namespace brahma
