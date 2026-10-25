#include "brahma/qwen2_schema.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace brahma {
namespace {
std::uint64_t uint_meta(const GgufModelInfo& m, const std::string& key) {
    const auto it=m.metadata.find(key);
    if(it==m.metadata.end()) throw std::runtime_error("Missing Qwen2 metadata: "+key);
    if(const auto* v=std::get_if<std::uint64_t>(&it->second)) return *v;
    if(const auto* v=std::get_if<std::int64_t>(&it->second); v && *v>0) return static_cast<std::uint64_t>(*v);
    throw std::runtime_error("Qwen2 metadata must be a positive integer: "+key);
}
double float_meta(const GgufModelInfo& m, const std::string& key, double fallback) {
    const auto it=m.metadata.find(key);
    if(it==m.metadata.end()) return fallback;
    double v{};
    if(const auto* d=std::get_if<double>(&it->second)) v=*d;
    else if(const auto* u=std::get_if<std::uint64_t>(&it->second)) v=static_cast<double>(*u);
    else if(const auto* i=std::get_if<std::int64_t>(&it->second)) v=static_cast<double>(*i);
    else throw std::runtime_error("Qwen2 metadata must be numeric: "+key);
    if(!std::isfinite(v)||v<=0.0) throw std::runtime_error("Qwen2 metadata must be finite and positive: "+key);
    return v;
}
void expect_shape(const TensorInfo& t, std::initializer_list<std::uint64_t> dims) {
    if(t.dimensions.size()!=dims.size() || !std::equal(t.dimensions.begin(),t.dimensions.end(),dims.begin(),dims.end()))
        throw std::runtime_error("Qwen2 tensor shape mismatch: "+t.name);
}
}
Qwen2ModelConfig validate_qwen2_schema(const GgufModelInfo& model) {
    if(model.architecture()!="qwen2") throw std::runtime_error("Expected GGUF architecture qwen2");
    Qwen2ModelConfig c;
    c.hidden_size=uint_meta(model,"qwen2.embedding_length");
    c.feed_forward_size=uint_meta(model,"qwen2.feed_forward_length");
    c.block_count=uint_meta(model,"qwen2.block_count");
    c.attention_heads=uint_meta(model,"qwen2.attention.head_count");
    c.kv_heads=uint_meta(model,"qwen2.attention.head_count_kv");
    c.vocabulary_size=model.vocabulary_size();
    c.rope_theta=float_meta(model,"qwen2.rope.freq_base",10000.0);
    c.rms_norm_epsilon=float_meta(model,"qwen2.attention.layer_norm_rms_epsilon",1.0e-6);
    if(!c.hidden_size||!c.feed_forward_size||!c.block_count||!c.attention_heads||!c.kv_heads)
        throw std::runtime_error("Qwen2 dimensions must be nonzero");
    if(c.hidden_size%c.attention_heads!=0 || c.attention_heads%c.kv_heads!=0)
        throw std::runtime_error("Qwen2 hidden/head dimensions are inconsistent");
    if(c.vocabulary_size==0) throw std::runtime_error("Qwen2 tokenizer vocabulary is missing");
    if(c.block_count>1000) throw std::runtime_error("Qwen2 block count exceeds safety limit");
    const auto head_dim=c.hidden_size/c.attention_heads;
    std::unordered_map<std::string,const TensorInfo*> tensors;
    tensors.reserve(model.tensors.size());
    for(const auto& t:model.tensors) {
        if(!tensors.emplace(t.name,&t).second) throw std::runtime_error("Duplicate tensor name: "+t.name);
    }
    auto require=[&](const std::string& name,std::initializer_list<std::uint64_t> dims) {
        const auto it=tensors.find(name);
        if(it==tensors.end()) throw std::runtime_error("Missing required Qwen2 tensor: "+name);
        expect_shape(*it->second,dims);
    };
    require("token_embd.weight",{c.hidden_size,c.vocabulary_size});
    require("output_norm.weight",{c.hidden_size});
    for(std::uint64_t i=0;i<c.block_count;++i) {
        const auto p="blk."+std::to_string(i)+".";
        require(p+"attn_norm.weight",{c.hidden_size});
        require(p+"ffn_norm.weight",{c.hidden_size});
        require(p+"attn_q.weight",{c.hidden_size,c.hidden_size});
        require(p+"attn_q.bias",{c.hidden_size});
        require(p+"attn_k.weight",{c.hidden_size,c.kv_heads*head_dim});
        require(p+"attn_k.bias",{c.kv_heads*head_dim});
        require(p+"attn_v.weight",{c.hidden_size,c.kv_heads*head_dim});
        require(p+"attn_v.bias",{c.kv_heads*head_dim});
        require(p+"attn_output.weight",{c.hidden_size,c.hidden_size});
        require(p+"ffn_gate.weight",{c.hidden_size,c.feed_forward_size});
        require(p+"ffn_up.weight",{c.hidden_size,c.feed_forward_size});
        require(p+"ffn_down.weight",{c.feed_forward_size,c.hidden_size});
    }
    // Standard Qwen2 GGUF may tie output to token embeddings and omit output.weight.
    const auto out=tensors.find("output.weight");
    if(out!=tensors.end()) expect_shape(*out->second,{c.hidden_size,c.vocabulary_size});
    return c;
}
} // namespace brahma
