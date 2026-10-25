#include "brahma/qwen2_schema.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void req(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void add_meta(brahma::GgufModelInfo& m,const std::string& k,std::uint64_t v){m.metadata.emplace(k,v);}
void add_tensor(brahma::GgufModelInfo& m,std::string name,std::vector<std::uint64_t> dims){m.tensors.push_back({std::move(name),std::move(dims),0,0});}
}
int main(){
 try {
  brahma::GgufModelInfo m; m.version=3;
  m.metadata.emplace("general.architecture",std::string("qwen2"));
  add_meta(m,"qwen2.embedding_length",4); add_meta(m,"qwen2.feed_forward_length",8);
  add_meta(m,"qwen2.block_count",1); add_meta(m,"qwen2.attention.head_count",2);
  add_meta(m,"qwen2.attention.head_count_kv",1);
  m.metadata.emplace("tokenizer.ggml.tokens",std::vector<std::string>{"a","b","c","d","e"});
  add_tensor(m,"token_embd.weight",{4,5}); add_tensor(m,"output_norm.weight",{4});
  const std::string p="blk.0.";
  add_tensor(m,p+"attn_norm.weight",{4}); add_tensor(m,p+"ffn_norm.weight",{4});
  add_tensor(m,p+"attn_q.weight",{4,4}); add_tensor(m,p+"attn_q.bias",{4});
  add_tensor(m,p+"attn_k.weight",{4,2}); add_tensor(m,p+"attn_k.bias",{2});
  add_tensor(m,p+"attn_v.weight",{4,2}); add_tensor(m,p+"attn_v.bias",{2});
  add_tensor(m,p+"attn_output.weight",{4,4});
  add_tensor(m,p+"ffn_gate.weight",{4,8}); add_tensor(m,p+"ffn_up.weight",{4,8});
  add_tensor(m,p+"ffn_down.weight",{8,4});
  const auto config=brahma::validate_qwen2_schema(m);
  req(config.hidden_size==4 && config.feed_forward_size==8 && config.block_count==1,"config values");
  req(config.attention_heads==2 && config.kv_heads==1 && config.vocabulary_size==5,"attention/vocabulary values");
  bool rejected=false;
  auto missing=m; missing.tensors.pop_back();
  try{(void)brahma::validate_qwen2_schema(missing);}catch(const std::exception&){rejected=true;}
  req(rejected,"missing tensor rejection");
  rejected=false;
  auto wrong=m; wrong.tensors[2].dimensions={5};
  try{(void)brahma::validate_qwen2_schema(wrong);}catch(const std::exception&){rejected=true;}
  req(rejected,"wrong tensor shape rejection");
  std::cout<<"Qwen2 schema tests passed\n"; return 0;
 } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
