#include "brahma/qwen2_block.hpp"
#include "brahma/tensor_ops.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace brahma {
namespace {
std::size_t checked(std::size_t a,std::size_t b,const char* message) {
    if(a && b>std::numeric_limits<std::size_t>::max()/a) throw std::overflow_error(message);
    return a*b;
}
void add_bias(std::vector<float>& x,std::span<const float> bias,std::size_t rows,std::size_t width) {
    if(bias.size()!=width || x.size()!=checked(rows,width,"bias shape overflow"))
        throw std::invalid_argument("Qwen2 projection bias shape mismatch");
    for(std::size_t r=0;r<rows;++r) for(std::size_t c=0;c<width;++c) x[r*width+c]+=bias[c];
}
void apply_rope_heads(std::vector<float>& x,std::size_t seq,std::size_t heads,std::size_t dim,float theta) {
    for(std::size_t pos=0;pos<seq;++pos) for(std::size_t head=0;head<heads;++head) {
        auto row=std::span<float>(x.data()+(pos*heads+head)*dim,dim);
        rope_f32(row,1,dim,pos,theta,RopeLayout::NeoX);
    }
}
std::vector<float> gqa_attention(std::span<const float> q,std::span<const float> k,std::span<const float> v,
    std::size_t seq,std::size_t heads,std::size_t kv_heads,std::size_t dim) {
    const auto q_width=checked(heads,dim,"Qwen2 Q shape overflow");
    const auto kv_width=checked(kv_heads,dim,"Qwen2 KV shape overflow");
    if(!seq||!heads||!kv_heads||!dim||heads%kv_heads||q.size()!=checked(seq,q_width,"Qwen2 Q size overflow")||
       k.size()!=checked(seq,kv_width,"Qwen2 K size overflow")||v.size()!=k.size())
        throw std::invalid_argument("Qwen2 GQA input shape mismatch");
    std::vector<float> out(q.size()),scores(seq);
    const float scale=1.0f/std::sqrt(static_cast<float>(dim));
    const auto group_size=heads/kv_heads;
    for(std::size_t pos=0;pos<seq;++pos) for(std::size_t h=0;h<heads;++h) {
        const auto kvh=h/group_size;
        float max_score=-std::numeric_limits<float>::infinity();
        for(std::size_t key=0;key<=pos;++key) {
            float dot=0.0f;
            const auto qo=(pos*heads+h)*dim, ko=(key*kv_heads+kvh)*dim;
            for(std::size_t d=0;d<dim;++d) dot+=q[qo+d]*k[ko+d];
            scores[key]=dot*scale; max_score=std::max(max_score,scores[key]);
        }
        double denom=0.0;
        for(std::size_t key=0;key<=pos;++key){scores[key]=std::exp(scores[key]-max_score);denom+=scores[key];}
        const auto oo=(pos*heads+h)*dim;
        for(std::size_t d=0;d<dim;++d) {
            double sum=0.0;
            for(std::size_t key=0;key<=pos;++key) sum+=double(scores[key])*v[(key*kv_heads+kvh)*dim+d];
            out[oo+d]=static_cast<float>(sum/denom);
        }
    }
    return out;
}
}
std::vector<float> qwen2_transformer_block_f32(
    std::span<const float> input,std::size_t seq,std::size_t hidden,std::size_t ffn,
    std::size_t heads,std::size_t kv_heads,float theta,float epsilon,const Qwen2BlockWeights& w) {
    if(!seq||!hidden||!ffn||!heads||!kv_heads||hidden%heads||heads%kv_heads||hidden/heads%2)
        throw std::invalid_argument("Invalid Qwen2 block dimensions");
    if(input.size()!=checked(seq,hidden,"Qwen2 input size overflow")) throw std::invalid_argument("Qwen2 input shape mismatch");
    const auto dim=hidden/heads, kv_width=checked(kv_heads,dim,"Qwen2 KV width overflow");
    auto norm=std::vector<float>(input.size());
    rms_norm_f32(input,w.attn_norm,norm,seq,hidden,epsilon);
    auto q=matmul_f32(norm,seq,hidden,w.q_weight,hidden); add_bias(q,w.q_bias,seq,hidden);
    auto k=matmul_f32(norm,seq,hidden,w.k_weight,kv_width); add_bias(k,w.k_bias,seq,kv_width);
    auto v=matmul_f32(norm,seq,hidden,w.v_weight,kv_width); add_bias(v,w.v_bias,seq,kv_width);
    apply_rope_heads(q,seq,heads,dim,theta); apply_rope_heads(k,seq,kv_heads,dim,theta);
    auto attended=gqa_attention(q,k,v,seq,heads,kv_heads,dim);
    auto projected=matmul_f32(attended,seq,hidden,w.attention_output_weight,hidden);
    std::vector<float> residual(input.size());
    add_f32(input,projected,residual);
    rms_norm_f32(residual,w.ffn_norm,norm,seq,hidden,epsilon);
    auto gate=matmul_f32(norm,seq,hidden,w.ffn_gate_weight,ffn);
    auto up=matmul_f32(norm,seq,hidden,w.ffn_up_weight,ffn);
    std::vector<float> gated(gate.size());
    swiglu_f32(gate,up,gated);
    auto down=matmul_f32(gated,seq,ffn,w.ffn_down_weight,hidden);
    add_f32(residual,down,norm);
    return norm;
}
} // namespace brahma
