#include "brahma/gguf_reader.hpp"
#include <array>
#include <fstream>
#include <stdexcept>
#include <type_traits>
namespace brahma { namespace {
constexpr std::uint64_t kMaxString=16ULL*1024*1024, kMaxEntries=10'000'000;
class Reader {
 std::ifstream in_; std::uint64_t size_{};
public:
 explicit Reader(const std::filesystem::path& p):in_(p,std::ios::binary){
  if(!in_) throw std::runtime_error("Cannot open GGUF file: "+p.string());
  in_.seekg(0,std::ios::end); auto e=in_.tellg(); if(e<0) throw std::runtime_error("Cannot determine file size");
  size_=static_cast<std::uint64_t>(e); in_.seekg(0,std::ios::beg);
 }
 template<class T>T read(){static_assert(std::is_trivially_copyable_v<T>); T v{}; in_.read(reinterpret_cast<char*>(&v),sizeof(T)); if(!in_) throw std::runtime_error("Unexpected end of GGUF file"); return v;}
 std::string string(){auto n=read<std::uint64_t>(); if(n>kMaxString||n>remaining()) throw std::runtime_error("Invalid GGUF string length"); std::string s(static_cast<std::size_t>(n),'\0'); if(n) in_.read(s.data(),static_cast<std::streamsize>(n)); if(!in_) throw std::runtime_error("Truncated GGUF string"); return s;}
 std::uint64_t remaining(){auto p=in_.tellg(); if(p<0||static_cast<std::uint64_t>(p)>size_) return 0; return size_-static_cast<std::uint64_t>(p);}
 std::uint64_t position(){auto p=in_.tellg(); if(p<0) throw std::runtime_error("Invalid file position"); return static_cast<std::uint64_t>(p);}
};
enum class Type:std::uint32_t{U8=0,I8=1,U16=2,I16=3,U32=4,I32=5,F32=6,Bool=7,String=8,Array=9,U64=10,I64=11,F64=12};
MetadataValue scalar(Reader&r,Type t){
 switch(t){
 case Type::U8:return std::uint64_t(r.read<std::uint8_t>()); case Type::I8:return std::int64_t(r.read<std::int8_t>());
 case Type::U16:return std::uint64_t(r.read<std::uint16_t>()); case Type::I16:return std::int64_t(r.read<std::int16_t>());
 case Type::U32:return std::uint64_t(r.read<std::uint32_t>()); case Type::I32:return std::int64_t(r.read<std::int32_t>());
 case Type::U64:return r.read<std::uint64_t>(); case Type::I64:return r.read<std::int64_t>();
 case Type::F32:return double(r.read<float>()); case Type::F64:return r.read<double>();
 case Type::Bool:{auto b=r.read<std::uint8_t>();if(b>1)throw std::runtime_error("Invalid GGUF boolean");return b!=0;}
 case Type::String:return r.string(); default:throw std::runtime_error("Invalid GGUF scalar type");
 }}
MetadataValue value(Reader&r,Type t){
 if(t!=Type::Array)return scalar(r,t);
 auto raw=r.read<std::uint32_t>(); if(raw>12||raw==9)throw std::runtime_error("Invalid GGUF array type");
 auto et=static_cast<Type>(raw); auto n=r.read<std::uint64_t>(); if(n>kMaxEntries)throw std::runtime_error("GGUF array too large");
 if(et==Type::String){std::vector<std::string> v;v.reserve(static_cast<std::size_t>(n));for(std::uint64_t i=0;i<n;++i)v.push_back(r.string());return v;}
 if(et==Type::Bool){std::vector<bool> v;v.reserve(static_cast<std::size_t>(n));for(std::uint64_t i=0;i<n;++i){auto b=r.read<std::uint8_t>();if(b>1)throw std::runtime_error("Invalid boolean array");v.push_back(b!=0);}return v;}
 if(et==Type::F32||et==Type::F64){std::vector<double> v;v.reserve(static_cast<std::size_t>(n));for(std::uint64_t i=0;i<n;++i)v.push_back(et==Type::F32?double(r.read<float>()):r.read<double>());return v;}
 if(et==Type::I8||et==Type::I16||et==Type::I32||et==Type::I64){std::vector<std::int64_t> v;v.reserve(static_cast<std::size_t>(n));for(std::uint64_t i=0;i<n;++i)v.push_back(std::get<std::int64_t>(scalar(r,et)));return v;}
 std::vector<std::uint64_t> v;v.reserve(static_cast<std::size_t>(n));for(std::uint64_t i=0;i<n;++i)v.push_back(std::get<std::uint64_t>(scalar(r,et)));return v;
}
}}
namespace brahma {
std::string GgufModelInfo::architecture()const{auto i=metadata.find("general.architecture");return i!=metadata.end()&&std::holds_alternative<std::string>(i->second)?std::get<std::string>(i->second):std::string{};}
std::uint64_t GgufModelInfo::vocabulary_size()const{for(auto k:{"tokenizer.ggml.tokens","tokenizer.tokens"}){auto i=metadata.find(k);if(i!=metadata.end()&&std::holds_alternative<std::vector<std::string>>(i->second))return std::get<std::vector<std::string>>(i->second).size();}return 0;}
GgufModelInfo GgufReader::inspect(const std::filesystem::path& path){
 Reader r(path);std::array<char,4> m{};for(auto&c:m)c=r.read<char>();if(m!=std::array<char,4>{'G','G','U','F'})throw std::runtime_error("Not a GGUF file");
 GgufModelInfo o;o.version=r.read<std::uint32_t>();if(o.version<2||o.version>3)throw std::runtime_error("Unsupported GGUF version "+std::to_string(o.version));
 auto nt=r.read<std::uint64_t>(), nk=r.read<std::uint64_t>();if(nt>kMaxEntries||nk>kMaxEntries)throw std::runtime_error("GGUF entry count exceeds safety limit");
 for(std::uint64_t i=0;i<nk;++i){auto k=r.string();auto t=r.read<std::uint32_t>();if(t>12)throw std::runtime_error("Unknown GGUF metadata type");if(!o.metadata.emplace(std::move(k),value(r,static_cast<Type>(t))).second)throw std::runtime_error("Duplicate GGUF metadata key");}
 auto a=o.metadata.find("general.alignment");if(a!=o.metadata.end()){if(!std::holds_alternative<std::uint64_t>(a->second))throw std::runtime_error("general.alignment must be unsigned");o.alignment=std::get<std::uint64_t>(a->second);}
 if(!o.alignment||o.alignment>4096||(o.alignment&(o.alignment-1)))throw std::runtime_error("Invalid GGUF alignment");
 o.tensors.reserve(static_cast<std::size_t>(nt));
 for(std::uint64_t i=0;i<nt;++i){TensorInfo t;t.name=r.string();auto nd=r.read<std::uint32_t>();if(!nd||nd>4)throw std::runtime_error("Invalid tensor dimension count");for(std::uint32_t d=0;d<nd;++d){auto n=r.read<std::uint64_t>();if(!n)throw std::runtime_error("Zero tensor dimension");t.dimensions.push_back(n);}t.ggml_type=r.read<std::uint32_t>();t.relative_offset=r.read<std::uint64_t>();o.tensors.push_back(std::move(t));}
 auto p=r.position();auto pad=(o.alignment-p%o.alignment)%o.alignment;o.tensor_data_offset=p+pad;
 // File may be metadata-only, so require the aligned payload start to be within or exactly at EOF.
 if(o.tensor_data_offset>p+r.remaining())throw std::runtime_error("Tensor data offset exceeds file size");
 return o;
}
} // namespace brahma
