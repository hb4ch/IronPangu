#pragma once
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <set>
namespace inferfabric::prepared {
// Portable hardware-preparation output. Offsets are fixed, 512-byte aligned,
// disjoint physical slots; logical lifetimes may reuse a slot on one stream.
struct WorkspacePlan {
  uint64_t bytes=0;
  static WorkspacePlan read(const std::string &path,uint64_t program,uint64_t backend) {
    std::ifstream in(path);std::string magic,extra;uint64_t p=0,b=0;WorkspacePlan w;
    if(!(in>>magic>>p>>b>>w.bytes) || magic!="IFWORKSPACE1" || p!=program || b!=backend ||
       w.bytes>(uint64_t(1)<<40) || (in>>extra))
      throw std::runtime_error("shared workspace plan fingerprint/bounds mismatch");
    return w;
  }
};
struct ArenaLayout {
  struct Slot { uint64_t offset, bytes; };
  uint64_t fingerprint=0, backend=0, bytes=0;
  std::vector<Slot> slots;
  std::map<int64_t,std::set<size_t>> bindings;
  static uint64_t fingerprint_file(const std::string &path) {
    std::ifstream in(path,std::ios::binary);
    if(!in) throw std::runtime_error("arena program fingerprint open");
    uint64_t hash=14695981039346656037ull; char c;
    while(in.get(c)) { hash^=static_cast<unsigned char>(c);hash*=1099511628211ull; }
    if(!in.eof()) throw std::runtime_error("arena program fingerprint read");
    return hash;
  }
  static ArenaLayout read(const std::string &path,uint64_t expected, uint64_t expected_backend) {
    std::ifstream in(path);std::string magic;uint64_t count=0;ArenaLayout a;
    if(!(in>>magic>>a.fingerprint>>a.backend>>a.bytes>>count) || magic!="IFARENA1" ||
       a.fingerprint!=expected || a.backend!=expected_backend || !a.bytes || a.bytes>(uint64_t(1)<<40) ||
       !count || count>1000000) throw std::runtime_error("frozen arena header/program mismatch");
    uint64_t end=0;
    for(uint64_t i=0;i<count;++i) {
      Slot s{};
      if(!(in>>s.offset>>s.bytes) || !s.bytes || s.offset%512 || s.offset<end ||
         s.offset>a.bytes || s.bytes>a.bytes-s.offset)
        throw std::runtime_error("frozen arena slot overlap/bounds/alignment");
      end=s.offset+s.bytes;a.slots.push_back(s);
    }
    std::string tag;uint64_t mappings=0;
    if(!(in>>tag>>mappings) || tag!="bindings" || !mappings || mappings>10001)
      throw std::runtime_error("frozen arena binding header");
    for(uint64_t j=0;j<mappings;++j) {
      int64_t instruction;uint64_t n;
      if(!(in>>instruction>>n) || instruction < -1 || instruction>10000 || !n || n>count || a.bindings.count(instruction))
        throw std::runtime_error("frozen arena instruction binding");
      auto &ids=a.bindings[instruction];
      for(uint64_t k=0;k<n;++k) {uint64_t id;if(!(in>>id) || id>=count || !ids.insert(size_t(id)).second)throw std::runtime_error("frozen arena slot binding");}
    }
    std::string extra;if(in>>extra)throw std::runtime_error("frozen arena trailing data");
    if(end!=a.bytes)throw std::runtime_error("frozen arena extent mismatch");
    return a;
  }
};
}
