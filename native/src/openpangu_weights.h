#pragma once
#include "openpangu_ops.h"
#include <filesystem>
#include <map>
#include <limits>
namespace openpangu {
struct WeightSlice {
 std::string name,shard,dtype,transform; uint64_t offset=0,bytes=0;std::vector<int64_t> shape;
};
struct LoadPlan {
 uint32_t rank=0,device=0;uint64_t stored=0,prepared=0;std::string metadata_sha256;
 std::vector<WeightSlice> weights;
 template<class T> static T number(std::istream& f){T n{};if(!f.read(reinterpret_cast<char*>(&n),sizeof(n)))throw std::runtime_error("truncated load plan");return n;}
 static std::string string(std::istream& f){auto n=number<uint32_t>(f);if(!n||n>4096)throw std::runtime_error("load plan string bound");std::string s(n,'\0');if(!f.read(s.data(),n)||s.find('\0')!=std::string::npos)throw std::runtime_error("invalid load plan string");return s;}
 explicit LoadPlan(const std::string& path){
  std::ifstream f(path,std::ios::binary);char magic[8];f.read(magic,8);if(!f||std::memcmp(magic,"IFLOAD01",8))throw std::runtime_error("load plan version");
  rank=number<uint32_t>(f);device=number<uint32_t>(f);auto count=number<uint32_t>(f);stored=number<uint64_t>(f);prepared=number<uint64_t>(f);metadata_sha256=string(f);
  if(rank>=8||device>1023||count==0||count>100000||metadata_sha256.size()!=64)throw std::runtime_error("load plan bounds");
  uint64_t sum=0,ready=0;std::map<std::string,bool> names;
  for(uint32_t i=0;i<count;++i){WeightSlice w;w.name=string(f);w.shard=string(f);w.dtype=string(f);w.transform=string(f);w.offset=number<uint64_t>(f);w.bytes=number<uint64_t>(f);auto dims=number<uint32_t>(f);
   if(dims==0||dims>4||w.bytes==0||w.bytes>(2ull<<30)||w.offset>uint64_t(INT64_MAX)-w.bytes||!names.emplace(w.name,true).second||std::filesystem::path(w.shard).filename()!=w.shard||w.shard=="."||w.shard=="..")throw std::runtime_error("load slice bounds/path/duplicate");
   uint64_t expected=w.dtype=="I8"?1:w.dtype=="BF16"?2:w.dtype=="F32"?4:0;if(!expected)throw std::runtime_error("load dtype");
   for(uint32_t d=0;d<dims;++d){auto n=number<uint64_t>(f);if(n==0||n>w.bytes/expected)throw std::runtime_error("load shape bounds");expected*=n;w.shape.push_back(n);}if(expected!=w.bytes)throw std::runtime_error("load shape bytes");
   bool cast=w.dtype=="BF16"&&(w.transform=="cast_fp32"||w.transform=="per_output_scale_to_fp32");sum+=w.bytes;ready+=w.bytes*(cast?2:1);weights.push_back(std::move(w));
  }
  if(f.peek()!=EOF||sum!=stored||ready!=prepared)throw std::runtime_error("load plan totals/trailing data");
 }
};
struct ResidentWeight {
 WeightSlice source;void* data=nullptr;uint64_t bytes=0;std::string dtype;
 explicit ResidentWeight(const WeightSlice& s):source(s),bytes(s.bytes),dtype(s.dtype){if(dtype=="BF16"&&(s.transform=="cast_fp32"||s.transform=="per_output_scale_to_fp32")){bytes*=2;dtype="F32";}checked(aclrtMalloc(&data,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"weight allocation");}
 ResidentWeight(const ResidentWeight&)=delete;
 ~ResidentWeight(){if(data)aclrtFree(data);}
};
// Raw canonical layout is retained. Operator-specific transposes/fusions are
// explicit executor preparation steps, never silently inferred by this loader.
struct ResidentWeights {
 std::map<std::string,std::unique_ptr<ResidentWeight>> tensors;
 uint64_t stored_bytes=0,prepared_bytes=0;
 ResidentWeights(const LoadPlan& plan,const std::string& directory,bool verify=false){
  auto root=std::filesystem::canonical(directory);std::vector<char> staging(4*1024*1024),roundtrip(8*1024*1024);std::vector<float> converted(staging.size()/2);
  // Validate every file range before allocating device weights.
  for(const auto&w:plan.weights){auto file=std::filesystem::canonical(root/w.shard);if(file.parent_path()!=root||!std::filesystem::is_regular_file(file)||std::filesystem::file_size(file)<w.offset+w.bytes)throw std::runtime_error("checkpoint shard path/range mismatch");}
  for(const auto&w:plan.weights){auto tensor=std::make_unique<ResidentWeight>(w);std::ifstream f(root/w.shard,std::ios::binary);f.seekg(w.offset);uint64_t done=0,device_offset=0;
   while(done<w.bytes){size_t n=std::min<uint64_t>(staging.size(),w.bytes-done);if(!f.read(staging.data(),n))throw std::runtime_error("short checkpoint read");const void* data=staging.data();size_t output=n;
    if(tensor->dtype!=w.dtype){for(size_t j=0;j<n/2;++j){uint16_t b;std::memcpy(&b,staging.data()+2*j,2);uint32_t bits=uint32_t(b)<<16;std::memcpy(&converted[j],&bits,4);if(!std::isfinite(converted[j]))throw std::runtime_error("nonfinite prepared weight");}data=converted.data();output=n*2;}
    auto dest=static_cast<char*>(tensor->data)+device_offset;checked(aclrtMemcpy(dest,output,data,output,ACL_MEMCPY_HOST_TO_DEVICE),"weight upload");
    if(verify){checked(aclrtMemcpy(roundtrip.data(),roundtrip.size(),dest,output,ACL_MEMCPY_DEVICE_TO_HOST),"weight readback");if(std::memcmp(roundtrip.data(),data,output))throw std::runtime_error("weight readback mismatch");}
    done+=n;device_offset+=output;
   }
   stored_bytes+=w.bytes;prepared_bytes+=tensor->bytes;tensors.emplace(w.name,std::move(tensor));
  }
 }
};
} // namespace openpangu
