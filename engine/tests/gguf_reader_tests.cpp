#include "brahma/gguf_reader.hpp"
#include "brahma/tensor_validator.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
template<class T>void wr(std::ofstream&o,T v){o.write(reinterpret_cast<const char*>(&v),sizeof(v));}
void str(std::ofstream&o,const std::string&s){wr(o,std::uint64_t(s.size()));o.write(s.data(),static_cast<std::streamsize>(s.size()));}
void req(bool b,const char*m){if(!b)throw std::runtime_error(m);}
int main(){
 auto p=std::filesystem::temp_directory_path()/"brahma_engine_minimal_test.gguf";
 try{
  std::ofstream o(p,std::ios::binary|std::ios::trunc);o.write("GGUF",4);wr(o,std::uint32_t(3));wr(o,std::uint64_t(1));wr(o,std::uint64_t(2));
  str(o,"general.architecture");wr(o,std::uint32_t(8));str(o,"llama");
  str(o,"general.alignment");wr(o,std::uint32_t(4));wr(o,std::uint32_t(32));
  str(o,"token_embd.weight");wr(o,std::uint32_t(2));wr(o,std::uint64_t(4));wr(o,std::uint64_t(8));wr(o,std::uint32_t(0));wr(o,std::uint64_t(0));
  auto pos=std::uint64_t(o.tellp());auto pad=(32-pos%32)%32;for(std::uint64_t i=0;i<pad;++i)o.put('\0');o.put('\0');o.close();
  auto m=brahma::GgufReader::inspect(p);req(m.version==3,"version");req(m.architecture()=="llama","architecture");req(m.tensors.size()==1,"tensor count");req(m.tensors[0].name=="token_embd.weight","tensor name");req(m.tensors[0].dimensions.size()==2&&m.tensors[0].dimensions[1]==8,"dimensions");req(m.alignment==32,"alignment");req(m.tensor_data_offset%32==0,"offset alignment");\n  TensorInfo q4{"q4_test",{256},12,0};auto q4_layout=tensor_layout(q4,m.tensor_data_offset,m.tensor_data_offset+144);req(q4_layout.elements==256&&q4_layout.bytes==144,"Q4_K payload layout");\n  bool rejected=false;try{TensorInfo bad{"truncated",{256},12,0};(void)tensor_layout(bad,m.tensor_data_offset,m.tensor_data_offset+143);}catch(const std::exception&){rejected=true;}req(rejected,"truncated payload rejection");\n  rejected=false;try{TensorInfo bad{"unknown",{32},999,0};(void)tensor_layout(bad,m.tensor_data_offset,m.tensor_data_offset+4096);}catch(const std::exception&){rejected=true;}req(rejected,"unknown type rejection");
  std::filesystem::remove(p);std::cout<<"GGUF reader tests passed\n";return 0;
 }catch(const std::exception&e){std::filesystem::remove(p);std::cerr<<e.what()<<"\n";return 1;}
}
