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
  // Deterministic differential-style test: compare production decoder with a
  // scalar oracle following the documented Q4_K packed scale/min layout.
  std::array<std::uint8_t,144> q4_block{};
  q4_block[0]=0x00;q4_block[1]=0x40; // fp16 d = 2.0
  q4_block[2]=0x00;q4_block[3]=0x38; // fp16 dmin = 0.5
  // Give all eight 32-value groups distinct scales and minima, including
  // values above 15 so the packed high-bit fields are exercised.
  const std::array<int,8> expected_scales{17,18,19,20,21,22,23,24};
  const std::array<int,8> expected_mins{25,26,27,28,29,30,31,32};
  for(int j=0;j<4;++j){
    q4_block[4+std::size_t(j)]=static_cast<std::uint8_t>(expected_scales[std::size_t(j)]);
    q4_block[8+std::size_t(j)]=static_cast<std::uint8_t>(expected_mins[std::size_t(j)]);
  }
  for(int j=4;j<8;++j){
    const auto s=expected_scales[std::size_t(j)];
    const auto m=expected_mins[std::size_t(j)];
    q4_block[8+std::size_t(j)]=static_cast<std::uint8_t>(
      ((s&0x0f)|((m&0x0f)<<4)));
    q4_block[std::size_t(j)]=static_cast<std::uint8_t>(
      q4_block[std::size_t(j)]|((s>>4)<<6));
    q4_block[4+std::size_t(j)]=static_cast<std::uint8_t>(
      q4_block[4+std::size_t(j)]|((m>>4)<<6));
  }
  for(std::size_t i=16;i<144;++i)
    q4_block[i]=static_cast<std::uint8_t>(((i*13u+7u)&0x0fu)|((((i*7u+3u)&0x0fu)<<4)));
  std::array<float,256> decoded{};
  brahma::dequantize_q4_k_block(q4_block,decoded);
  // Independent scalar oracle: decode all eight packed scale/min entries,
  // then apply the serialized low/high nibble order for each 64-value stripe.
  std::array<int,8> scales{},mins{};
  for(int j=0;j<4;++j){
    scales[std::size_t(j)]=q4_block[4+std::size_t(j)]&0x3f;
    mins[std::size_t(j)]=q4_block[8+std::size_t(j)]&0x3f;
  }
  for(int j=4;j<8;++j){
    scales[std::size_t(j)]=(q4_block[8+std::size_t(j)]&0x0f)|
      ((q4_block[std::size_t(j)]>>6)<<4);
    mins[std::size_t(j)]=(q4_block[8+std::size_t(j)]>>4)|
      ((q4_block[4+std::size_t(j)]>>6)<<4);
  }
  for(int stripe=0;stripe<4;++stripe){
    for(int k=0;k<32;++k){
      const auto packed=q4_block[16+std::size_t(stripe)*32+std::size_t(k)];
      const float low=2.0f*float(scales[std::size_t(stripe*2)])*float(packed&0x0f)
        -0.5f*float(mins[std::size_t(stripe*2)]);
      const float high=2.0f*float(scales[std::size_t(stripe*2+1)])*float(packed>>4)
        -0.5f*float(mins[std::size_t(stripe*2+1)]);
      req(decoded[std::size_t(stripe)*64+std::size_t(k)]==low,
          "Q4_K scalar oracle low nibble");
      req(decoded[std::size_t(stripe)*64+32+std::size_t(k)]==high,
          "Q4_K scalar oracle high nibble");
    }
  }
  std::vector<std::uint8_t> q4_payload(q4_block.begin(),q4_block.end());
  const auto decoded_vector=brahma::dequantize_q4_k(q4_payload,256);
  req(decoded_vector.size()==256&&decoded_vector[0]==decoded[0]&&decoded_vector[255]==decoded[255],
      "Q4_K tensor dequantization");
  rejected=false;
  try{(void)brahma::dequantize_q4_k(q4_payload,255);}catch(const std::exception&){rejected=true;}
  req(rejected,"Q4_K invalid element count rejection");
  // Q6_K deterministic block test against a separately written scalar oracle.
  std::array<std::uint8_t,210> q6{};
  for(std::size_t i=0;i<128;++i) q6[i]=static_cast<std::uint8_t>((i*37u+11u)&0xffu);
  for(std::size_t i=128;i<192;++i) q6[i]=static_cast<std::uint8_t>((i*19u+5u)&0xffu);
  for(std::size_t i=0;i<16;++i) q6[192+i]=static_cast<std::uint8_t>(static_cast<std::int8_t>(int(i)-8));
  q6[208]=0x00; q6[209]=0x3c; // fp16 d = 1
  std::array<float,256> q6_decoded{};
  brahma::dequantize_q6_k_block(q6,q6_decoded);
  std::array<float,256> q6_oracle{};
  for(std::size_t half=0;half<2;++half) for(std::size_t l=0;l<32;++l) {
    const auto qlo0=q6[half*64+l], qlo1=q6[half*64+l+32], hi=q6[128+half*32+l];
    const int vals[4]={
      int((qlo0&0x0f)|((hi&0x03)<<4))-32,
      int((qlo1&0x0f)|(((hi>>2)&0x03)<<4))-32,
      int((qlo0>>4)|(((hi>>4)&0x03)<<4))-32,
      int((qlo1>>4)|((hi>>6)<<4))-32
    };
    const auto group=l/16, base=half*8;
    const std::size_t indices[4]={half*128+l,half*128+l+32,half*128+l+64,half*128+l+96};
    const std::size_t si[4]={base+group,base+group+2,base+group+4,base+group+6};
    for(int j=0;j<4;++j) q6_oracle[indices[j]]=float(static_cast<std::int8_t>(q6[192+si[j]]))*float(vals[j]);
  }
  for(std::size_t i=0;i<256;++i) req(q6_decoded[i]==q6_oracle[i],"Q6_K scalar oracle");
  const std::vector<std::uint8_t> q6_payload(q6.begin(),q6.end());
  const auto q6_vector=brahma::dequantize_q6_k(q6_payload,256);
  req(q6_vector.size()==256&&q6_vector.front()==q6_decoded.front()&&q6_vector.back()==q6_decoded.back(),"Q6_K tensor dequantization");
  rejected=false;
  try{(void)brahma::dequantize_q6_k(q6_payload,255);}catch(const std::exception&){rejected=true;}
  req(rejected,"Q6_K invalid element count rejection");
  std::filesystem::remove(p);std::cout<<"GGUF reader and tensor validator tests passed\n";return 0;
 }catch(const std::exception& e){std::filesystem::remove(p);std::cerr<<e.what()<<"\n";return 1;}
}
