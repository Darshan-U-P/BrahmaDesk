#include "brahma/gguf_reader.hpp"
#include <exception>
#include <iostream>
int main(int argc,char**argv){
 if(argc!=2){std::cerr<<"Usage: brahma-engine <model.gguf>\nBrahma Engine 0.1: GGUF inspector; inference is not implemented yet.\n";return 2;}
 try{auto m=brahma::GgufReader::inspect(argv[1]);std::cout<<"Brahma Engine Core 0.1\nGGUF version: "<<m.version<<"\nArchitecture: "<<(m.architecture().empty()?"(not declared)":m.architecture())<<"\nMetadata entries: "<<m.metadata.size()<<"\nTensor descriptors: "<<m.tensors.size()<<"\nTensor data offset: "<<m.tensor_data_offset<<"\nAlignment: "<<m.alignment<<"\nVocabulary entries: "<<m.vocabulary_size()<<"\n";return 0;}
 catch(const std::exception&e){std::cerr<<"Brahma Engine error: "<<e.what()<<"\n";return 1;}
}
