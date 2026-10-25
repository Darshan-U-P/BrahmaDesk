#include "brahma/qwen2_block.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
void req(bool v,const char* m){if(!v)throw std::runtime_error(m);}
bool near(float a,float b,float tol=1.0e-5f){return std::fabs(a-b)<=tol;}
}
int main(){
 try {
  const std::array<float,4> input{1.0f,2.0f,3.0f,4.0f};
  const std::array<float,4> ones{1,1,1,1}, zeros{0,0,0,0};
  const std::array<float,2> kv_zeros{0,0};
  const std::array<float,16> identity{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
  const std::array<float,8> kv_projection{1,0, 0,1, 1,0, 0,1};
  const std::array<float,16> zero_matrix{};
  brahma::Qwen2BlockWeights w{
   ones,identity,zeros,kv_projection,kv_zeros,kv_projection,kv_zeros,
   identity,ones,zero_matrix,zero_matrix,zero_matrix
  };
  const auto out=brahma::qwen2_transformer_block_f32(input,1,4,4,2,1,10000.0f,1.0e-5f,w);
  double ss=0;for(float x:input)ss+=double(x)*x;
  const float scale=1.0f/std::sqrt(float(ss/4.0)+1.0e-5f);
  req(out.size()==4,"block output size");
  for(std::size_t i=0;i<4;++i) req(near(out[i],input[i]+input[i]*scale),"single-token residual path");
  bool rejected=false;
  try{(void)brahma::qwen2_transformer_block_f32(input,1,4,4,3,1,10000.0f,1.0e-5f,w);}
  catch(const std::invalid_argument&){rejected=true;}
  req(rejected,"invalid head configuration rejection");
  std::cout<<"Qwen2 block tests passed\n";return 0;
 } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
