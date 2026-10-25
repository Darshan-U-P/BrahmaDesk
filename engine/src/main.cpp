#include "brahma/gguf_reader.hpp"
#include "brahma/tensor_validator.hpp"
#include <exception>
#include <iostream>
int main(int argc,char**argv){
 if(argc!=2){std::cerr<<"Usage: brahma-engine <model.gguf>\nBrahma Engine 0.2: GGUF inspection and tensor payload validation; inference is not implemented yet.\n";return 2;}
 try{
  auto m=brahma::GgufReader::inspect(argv[1]);
  auto layouts=brahma::validate_tensor_payloads(m,argv[1]);
  std::uint64_t total=0;
  for(const auto& t:layouts){if(t.bytes>UINT64_MAX-total)throw std::runtime_error("Total tensor byte count overflow");total+=t.bytes;}
  std::cout<<"Brahma Engine Core 0.2\nGGUF version: "<<m.version<<"\nArchitecture: "<<(m.architecture().empty()?"(not declared)":m.architecture())
   <<"\nMetadata entries: "<<m.metadata.size()<<"\nTensor descriptors: "<<m.tensors.size()<<"\nTensor data offset: "<<m.tensor_data_offset
   <<"\nAlignment: "<<m.alignment<<"\nVocabulary entries: "<<m.vocabulary_size()<<"\nPayload validation: PASS"
   <<"\nValidated tensor payload bytes: "<<total<<"\nInference: NOT IMPLEMENTED\n";
  return 0;
 }catch(const std::exception&e){std::cerr<<"Brahma Engine error: "<<e.what()<<"\n";return 1;}
}
