#pragma once
#include <algorithm>
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
  struct Envelope { uint64_t max_request_bytes; bool workspace; };
  std::map<int64_t,Envelope> envelopes;
  bool pooled=false;
  int64_t workspace_slot=-1;
  bool permits_slot(int64_t instruction,size_t slot) const {
    if(!pooled) {
      auto it=bindings.find(instruction);
      return it!=bindings.end() && it->second.count(slot);
    }
    auto it=envelopes.find(instruction);
    return it!=envelopes.end() && (int64_t(slot)==workspace_slot ? it->second.workspace : it->second.max_request_bytes>0);
  }
  bool permits_request(int64_t instruction,uint64_t bytes,bool workspace=false) const {
    if(!pooled)return true;
    auto it=envelopes.find(instruction);
    if(it==envelopes.end())return false;
    if(workspace)return instruction==-1 && it->second.workspace && workspace_slot>=0 && bytes==slots[size_t(workspace_slot)].bytes;
    return bytes>0 && bytes<=it->second.max_request_bytes;
  }
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
    if(!(in>>magic>>a.fingerprint>>a.backend>>a.bytes>>count) || (magic!="IFARENA1" && magic!="IFARENA2") ||
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
    if(magic=="IFARENA2") {
      a.pooled=true;
      if(!(in>>tag>>mappings>>a.workspace_slot) || tag!="pool" || !mappings || mappings>10001 || a.workspace_slot < -1 || a.workspace_slot>0)
        throw std::runtime_error("frozen pool header");
      uint64_t max_slot=0;
      for(size_t i=0;i<a.slots.size();++i)if(int64_t(i)!=a.workspace_slot)max_slot=std::max(max_slot,a.slots[i].bytes);
      for(uint64_t j=0;j<mappings;++j) {
        int64_t instruction;uint64_t bytes;int workspace;
        if(!(in>>instruction>>bytes>>workspace) || instruction < -1 || instruction>10000 || bytes>max_slot || (workspace!=0 && workspace!=1) || (workspace && a.workspace_slot<0) || (!bytes && !workspace) || a.envelopes.count(instruction))
          throw std::runtime_error("frozen pool request envelope");
        a.envelopes.emplace(instruction,Envelope{bytes,workspace==1});
      }
      if(a.workspace_slot>=0 && (!a.envelopes.count(-1) || !a.envelopes.at(-1).workspace))
        throw std::runtime_error("frozen pool workspace initialization");
      std::string extra;if(in>>extra)throw std::runtime_error("frozen arena trailing data");
      if(end!=a.bytes)throw std::runtime_error("frozen arena extent mismatch");
      return a;
    }
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
