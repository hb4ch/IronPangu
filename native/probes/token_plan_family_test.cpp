#include "token_execution_plan.h"
#include <filesystem>
#include <iostream>

// Reader-only binary compatibility tests; no device or checkpoint is required.
int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: token_plan_family_test TEMP_DIRECTORY");
    auto path = std::filesystem::path(argv[1]) / "family-reader-test.bin";
    if (std::filesystem::exists(path)) throw std::runtime_error("test file already exists");
    struct Cleanup { std::filesystem::path p; ~Cleanup() { std::filesystem::remove(p); } } cleanup{path};
    auto write = [&](int version, uint32_t maximum, uint32_t shared, const std::vector<uint32_t> &variants) {
      std::ofstream out(path, std::ios::binary);
      auto number = [&](uint32_t n) { for (int i=0;i<4;++i) out.put(char(n>>(8*i))); };
      auto text = [&](const std::string &s) { number(uint32_t(s.size())); out.write(s.data(),s.size()); };
      out << "IFTOKN0" << version;
      number(4096); number(148902); number(0); number(1);
      if(version>=3) number(2);
      if(version>=4) number(maximum);
      if(version==6) {number(shared);number(uint32_t(variants.size()));for(auto n:variants)number(n);}
      if(version==7) number(shared);
      number(0); number(151552); text("output");text("fixture");number(0);number(0);
      text("bf16");text("fixture");number(0);
    };
    for(int version:{2,3,4,5,6}) {
      write(version,16,1,{2,4,8,16});
      auto plan=inferfabric::execution::Plan::read(path.string());
      std::vector<uint32_t> expected=version<4?std::vector<uint32_t>{}:version<6?std::vector<uint32_t>{16}:std::vector<uint32_t>{2,4,8,16};
      if(plan.prefill_variants!=expected || plan.shared_pages!=(version>=5))throw std::runtime_error("legacy/family contract mismatch");
      for(uint32_t n=0;n<=32;++n) {
        bool expected_support=false;for(auto v:expected)if(v==n)expected_support=true;
        if(plan.supports_prefill(n)!=expected_support)throw std::runtime_error("undeclared shape accepted");
      }
    }
    write(6,32,1,{2,4,8,16,32});
    auto wide=inferfabric::execution::Plan::read(path.string());
    if(!wide.supports_prefill(32) || wide.supports_prefill(31) || wide.supports_prefill(64))throw std::runtime_error("wide exact-size contract mismatch");
    unsigned rejected=0;
    for(uint32_t shared:{0u,1u}) {
      write(7,4096,shared,{});
      auto dynamic=inferfabric::execution::Plan::read(path.string());
      if(!dynamic.uncaptured_prefill || dynamic.prefill_family || !dynamic.prefill_variants.empty() || dynamic.shared_pages!=bool(shared))
        throw std::runtime_error("uncaptured envelope confused with graph family");
      for(uint32_t count=0;count<=4097;++count)
        if(dynamic.supports_prefill(count)!=(count>=2 && count<=4096))
          throw std::runtime_error("uncaptured tail bounds");
    }
    for(auto spec:std::vector<std::pair<uint32_t,uint32_t>>{{32,0},{4096,2},{8192,0}}) {
      write(7,spec.first,spec.second,{});
      bool failed=false;
      try {(void)inferfabric::execution::Plan::read(path.string());}catch(const std::exception &){failed=true;}
      if(!failed)throw std::runtime_error("malformed uncaptured envelope accepted");
    }
    auto reject=[&](uint32_t maximum,uint32_t shared,const std::vector<uint32_t>& variants) {
      write(6,maximum,shared,variants);
      try { (void)inferfabric::execution::Plan::read(path.string()); }
      catch(const std::exception &) {++rejected;return;}
      throw std::runtime_error("malformed family accepted");
    };
    reject(16,1,{});reject(16,1,{1,16});reject(16,1,{3,16});reject(16,1,{2,2,16});
    reject(16,1,{4,2,16});reject(16,1,{2,4,8});reject(16,2,{16});
    reject(16,1,{2,4,8,16,32});reject(64,1,{64});reject(16,1,{0xffffffffu});
    std::cout << "five binary versions passed; " << rejected << " malformed families rejected\n";
  } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
