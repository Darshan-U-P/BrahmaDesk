#include "brahma/gguf_reader.hpp"
#include "brahma/tensor_validator.hpp"
#include "brahma/quantization.hpp"
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
template<class T> void wr(std::ofstream& o,T v){o.write(reinterpret_cast<const char*>(&v),sizeof(v));}
void str(std::ofstream& o,const std::string& s){wr(o,std::uint64_t(s.size()));o.write(s.data(),static_cast<std::streamsize>(s.size()));}
void req(bool b,const char* m){if(!b)throw std::runtime_error(m);}
int main(){
 auto p=std::filesystem::temp_directory_path()/"brahma_engine_minimal_test.gguf";
 try{
  std::ofstream o(p,std::ios::binary|std::ios::trunc);
  o.write("GGUF",4);wr(o,std::uint32_t(3));wr(o,std::uint64_t(1));wr(o,std::uint64_t(2));
  str(o,"general.architecture");wr(o,std::uint32_t(8));str(o,"llama");
  str(o,"general.alignment");wr(o,std::uint32_t(4));wr(o,std::uint32_t(32));
  str(o,"token_embd.weight");wr(o,std::uint32_t(2));wr(o,std::uint64_t(4));wr(o,std::uint64_t(8));wr(o,std::uint32_t(0));wr(o,std::uint64_t(0));
  auto pos=std::uint64_t(o.tellp());auto pad=(32-pos%32)%32;for(std::uint64_t i=0;i<pad;++i)o.put('\0');o.put('\0');o.close();
  auto m=brahma::GgufReader::inspect(p);
  req(m.version==3,"version");req(m.architecture()=="llama","architecture");req(m.tensors.size()==1,"tensor count");
  req(m.tensors[0].name=="token_embd.weight","tensor name");req(m.tensors[0].dimensions.size()==2&&m.tensors[0].dimensions[1]==8,"dimensions");
  req(m.alignment==32,"alignment");req(m.tensor_data_offset%32==0,"offset alignment");
  brahma::TensorInfo q4{"q4_test",{256},12,0};
  auto q4_layout=brahma::tensor_layout(q4,m.tensor_data_offset,m.tensor_data_offset+144);
  req(q4_layout.elements==256&&q4_layout.bytes==144,"Q4_K payload layout");
  bool rejected=false;
  try{brahma::TensorInfo bad{"truncated",{256},12,0};(void)brahma::tensor_layout(bad,m.tensor_data_offset,m.tensor_data_offset+143);}
  catch(const std::exception&){rejected=true;}
  req(rejected,"truncated payload rejection");
  rejected=false;
  try{brahma::TensorInfo bad{"unknown",{32},999,0};(void)brahma::tensor_layout(bad,m.tensor_data_offset,m.tensor_data_offset+4096);}
  catch(const std::exception&){rejected=true;}
  req(rejected,"unknown type rejection");
  // Non-uniform fixture exercises both scale/min decoding and nibble ordering.
  std::array<std::uint8_t,144> q4_block{};
  q4_block[0]=0x00;q4_block[1]=0x40; // fp16 d = 2.0
  q4_block[2]=0x00;q4_block[3]=0x38; // fp16 dmin = 0.5
  q4_block[4]=1;q4_block[5]=2;q4_block[6]=3;q4_block[7]=4;
  q4_block[8]=5;q4_block[9]=6;q4_block[10]=7;q4_block[11]=8;
  for(std::size_t i=16;i<144;++i)q4_block[i]=0x11; // low and high nibbles are both 1
  std::array<float,256> decoded{};
  brahma::dequantize_q4_k_block(q4_block,decoded);
  for(std::size_t i=0;i<32;++i)req(decoded[i]==-0.5f,"Q4_K scale/min group 0 low nibbles");
  for(std::size_t i=32;i<64;++i)req(decoded[i]==1.0f,"Q4_K scale/min group 0 high nibbles");
  for(std::size_t i=64;i<96;++i)req(decoded[i]==2.5f,"Q4_K scale/min group 1 low nibbles");
  for(std::size_t i=96;i<128;++i)req(decoded[i]==4.0f,"Q4_K scale/min group 1 high nibbles");
  for(std::size_t i=128;i<256;++i)req(decoded[i]==0.0f,"Q4_K zero-scale groups");
  std::vector<std::uint8_t> q4_payload(q4_block.begin(),q4_block.end());
  const auto decoded_vector=brahma::dequantize_q4_k(q4_payload,256);
  req(decoded_vector.size()==256&&decoded_vector[0]==-0.5f&&decoded_vector[127]==4.0f,
      "Q4_K tensor dequantization");
  rejected=false;
  try{(void)brahma::dequantize_q4_k(q4_payload,255);}catch(const std::exception&){rejected=true;}
  req(rejected,"Q4_K invalid element count rejection");
  std::filesystem::remove(p);std::cout<<"GGUF reader and tensor validator tests passed\n";return 0;
 }catch(const std::exception& e){std::filesystem::remove(p);std::cerr<<e.what()<<"\n";return 1;}
}
