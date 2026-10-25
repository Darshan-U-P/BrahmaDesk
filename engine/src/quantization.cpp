#include "brahma/quantization.hpp"
#include <bit>
#include <cmath>
#include <stdexcept>
#include <limits>

namespace brahma {
namespace {
float half_to_float(std::uint16_t h) noexcept {
    const std::uint32_t sign=(std::uint32_t(h&0x8000u))<<16;
    std::uint32_t exp=(h>>10)&0x1fu, mant=h&0x03ffu, bits{};
    if(exp==0){if(mant==0)bits=sign;else{int e=-14;while((mant&0x0400u)==0){mant<<=1;--e;}mant&=0x03ffu;bits=sign|(std::uint32_t(e+127)<<23)|(mant<<13);}}
    else if(exp==0x1fu) bits=sign|0x7f800000u|(mant<<13);
    else bits=sign|((exp+(127-15))<<23)|(mant<<13);
    return std::bit_cast<float>(bits);
}
std::uint16_t u16le(const std::uint8_t* b) noexcept { return std::uint16_t(b[0])|(std::uint16_t(b[1])<<8); }
void get_scale_min(std::span<const std::uint8_t,144> b,int j,int& scale,int& min) noexcept {
    const auto* s=b.data()+4;
    if(j<4){scale=s[j]&0x3f;min=s[j+4]&0x3f;}
    else{scale=(s[j+4]&0x0f)|((s[j-4]>>6)<<4);min=(s[j+4]>>4)|((s[j]>>6)<<4);}
}
template<std::size_t BlockBytes>
std::vector<float> dequantize_blocks(std::span<const std::uint8_t> payload,std::size_t n,
                                     std::size_t values_per_block,
                                     void(*decode)(std::span<const std::uint8_t,BlockBytes>,std::span<float,256>)) {
    if(n%values_per_block) throw std::runtime_error("Quantized element count must be divisible by block size");
    if(n/values_per_block>std::numeric_limits<std::size_t>::max()/BlockBytes ||
       payload.size()!=(n/values_per_block)*BlockBytes)
        throw std::runtime_error("Quantized payload size does not match element count");
    std::vector<float> out(n);
    for(std::size_t i=0;i<n/values_per_block;++i)
        decode(std::span<const std::uint8_t,BlockBytes>(payload.data()+i*BlockBytes,BlockBytes),
               std::span<float,256>(out.data()+i*256,256));
    return out;
}
}
void dequantize_q4_k_block(std::span<const std::uint8_t,144> b,std::span<float,256> out) {
    const float d=half_to_float(u16le(b.data())), dmin=half_to_float(u16le(b.data()+2));
    const auto* qs=b.data()+16; std::size_t pos=0;
    for(int group=0;group<4;++group){int s1{},m1{},s2{},m2{};get_scale_min(b,group*2,s1,m1);get_scale_min(b,group*2+1,s2,m2);
        for(int i=0;i<32;++i){const auto q=qs[std::size_t(group)*32+std::size_t(i)];out[pos++]=d*float(s1)*float(q&15)-dmin*float(m1);}
        for(int i=0;i<32;++i){const auto q=qs[std::size_t(group)*32+std::size_t(i)];out[pos++]=d*float(s2)*float(q>>4)-dmin*float(m2);}
    }
}
std::vector<float> dequantize_q4_k(std::span<const std::uint8_t> p,std::size_t n) {
    return dequantize_blocks<144>(p,n,256,&dequantize_q4_k_block);
}
void dequantize_q6_k_block(std::span<const std::uint8_t,210> b,std::span<float,256> out) {
    const auto* ql=b.data(); const auto* qh=b.data()+128; const auto* scales=reinterpret_cast<const std::int8_t*>(b.data()+192);
    const float d=half_to_float(u16le(b.data()+208));
    std::size_t out_base=0;
    for(std::size_t half=0;half<2;++half) {
        const auto* ql_half=ql+half*64; const auto* qh_half=qh+half*32; const auto* sc=scales+half*8;
        for(std::size_t l=0;l<32;++l) {
            const auto a=std::uint8_t((ql_half[l]&0x0f)|((qh_half[l]&0x03)<<4));
            const auto c=std::uint8_t((ql_half[l+32]&0x0f)|((qh_half[l]>>2&0x03)<<4));
            const auto e=std::uint8_t((ql_half[l]>>4)|((qh_half[l]>>4&0x03)<<4));
            const auto g=std::uint8_t((ql_half[l+32]>>4)|((qh_half[l]>>6)<<4));
            const auto group=l/16;
            out[out_base+l]=d*float(sc[group+0])*float(int(a)-32);
            out[out_base+l+32]=d*float(sc[group+2])*float(int(c)-32);
            out[out_base+l+64]=d*float(sc[group+4])*float(int(e)-32);
            out[out_base+l+96]=d*float(sc[group+6])*float(int(g)-32);
        }
        out_base+=128;
    }
}
std::vector<float> dequantize_q6_k(std::span<const std::uint8_t> p,std::size_t n) {
    return dequantize_blocks<210>(p,n,256,&dequantize_q6_k_block);
}

float dot_q4_k_block(std::span<const std::uint8_t,144> b,
                     std::span<const float,256> input) noexcept {
    const float d=half_to_float(u16le(b.data()));
    const float dmin=half_to_float(u16le(b.data()+2));
    const auto* qs=b.data()+16;
    float sum=0.0f;
    std::size_t input_pos=0;
    for(int group=0;group<4;++group) {
        int s1{},m1{},s2{},m2{};
        get_scale_min(b,group*2,s1,m1);
        get_scale_min(b,group*2+1,s2,m2);
        const auto* q=qs+std::size_t(group)*32;
        for(std::size_t i=0;i<32;++i) {
            sum += (d*float(s1)*float(q[i]&0x0f)-dmin*float(m1))*input[input_pos+i];
            sum += (d*float(s2)*float(q[i]>>4)-dmin*float(m2))*input[input_pos+32+i];
        }
        input_pos+=64;
    }
    return sum;
}

float dot_q6_k_block(std::span<const std::uint8_t,210> b,
                     std::span<const float,256> input) noexcept {
    const auto* ql=b.data();
    const auto* qh=b.data()+128;
    const auto* scales=reinterpret_cast<const std::int8_t*>(b.data()+192);
    const float d=half_to_float(u16le(b.data()+208));
    float sum=0.0f;
    for(std::size_t half=0;half<2;++half) {
        const auto* ql_half=ql+half*64;
        const auto* qh_half=qh+half*32;
        const auto* sc=scales+half*8;
        const std::size_t base=half*128;
        for(std::size_t l=0;l<32;++l) {
            const int a=int(std::uint8_t((ql_half[l]&0x0f)|((qh_half[l]&0x03)<<4)))-32;
            const int c=int(std::uint8_t((ql_half[l+32]&0x0f)|((qh_half[l]>>2&0x03)<<4)))-32;
            const int e=int(std::uint8_t((ql_half[l]>>4)|((qh_half[l]>>4&0x03)<<4)))-32;
            const int g=int(std::uint8_t((ql_half[l+32]>>4)|((qh_half[l]>>6)<<4)))-32;
            const auto group=l/16;
            sum += d*float(sc[group+0])*float(a)*input[base+l];
            sum += d*float(sc[group+2])*float(c)*input[base+l+32];
            sum += d*float(sc[group+4])*float(e)*input[base+l+64];
            sum += d*float(sc[group+6])*float(g)*input[base+l+96];
        }
    }
    return sum;
}

} // namespace brahma
