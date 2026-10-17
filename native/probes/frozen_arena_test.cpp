#include "frozen_arena.h"
#include "bounded_buffers.h"
#include <cstdio>
#include <iostream>
#include <functional>
int main() {
  const char *path="frozen-arena-contract-test.txt";
  try {
    const std::string valid="IFARENA1 123 456 1028 3\n0 4\n512 128\n1024 4\nbindings 2\n-1 1 0\n7 2 1 2\n";
    auto load=[&](const std::string &text,uint64_t program=123,uint64_t backend=456) {
      {std::ofstream out(path);out<<text;}
      return inferfabric::prepared::ArenaLayout::read(path,program,backend);
    };
    auto a=load(valid);
    if(a.slots.size()!=3 || a.bytes!=1028 || a.bindings.at(7).size()!=2)
      throw std::runtime_error("valid layout changed");
    int rejected=0;
    auto reject=[&](auto fn){try{fn();}catch(const std::runtime_error&){++rejected;return;}throw std::logic_error("bad layout accepted");};
    reject([&]{load(valid,124);});
    reject([&]{load(valid,123,457);});
    auto replace=[&](const std::string &from,const std::string &to){auto text=valid;text.replace(text.find(from),from.size(),to);return text;};
    reject([&]{load(replace("512 128","0 128"));});
    reject([&]{load(replace("512 128","513 128"));});
    reject([&]{load(replace("1024 4","1024 8"));});
    reject([&]{load(replace("7 2 1 2","7 2 1 3"));});
    reject([&]{load(replace("7 2 1 2","7 2 1 1"));});
    reject([&]{load(replace("7 2 1 2","-1 2 1 2"));});
    reject([&]{load(valid+"extra");});
    reject([&]{load(replace("bindings 2","bindings 3"));});
    const std::string pool="IFARENA2 123 456 33555456 3\n0 1024\n1024 8388608\n8389632 25165824\npool 2 0\n-1 25165824 1\n8 8388608 0\n";
    auto pooled=load(pool);
    if(!pooled.permits_request(8,1083392) || !pooled.permits_slot(8,2) || pooled.permits_slot(8,0) ||
       pooled.permits_request(8,8388609) || pooled.permits_request(99,4) ||
       !pooled.permits_request(-1,1024,true) || pooled.permits_request(8,1024,true))
      throw std::runtime_error("pooled request bound confused with backing slot capacity");
    struct Block {size_t slot,bytes;};
    Block workspace_slot{0,1024}, larger_free{2,25165824};
    std::map<size_t,std::vector<Block *>> free{{1024,{&workspace_slot}},{25165824,{&larger_free}}};
    auto eligible=[&](Block *b){return pooled.permits_slot(8,b->slot) && int64_t(b->slot)!=pooled.workspace_slot;};
    auto *lease=inferfabric::prepared::take_bounded_buffer(free,1083392,eligible);
    if(lease!=&larger_free || inferfabric::prepared::take_bounded_buffer(free,1083392,eligible))
      throw std::runtime_error("shared pool lost larger backing or reused live lease");
    for(size_t rows=1;rows<=4096;++rows) {
      free[larger_free.bytes].push_back(&larger_free);
      if(!pooled.permits_request(8,rows*2048) || inferfabric::prepared::take_bounded_buffer(free,rows*2048,eligible)!=&larger_free)
        throw std::runtime_error("tail envelope cannot use oversized free backing");
    }
    reject([&]{load(pool+"extra");});
    auto invalid_pool=pool;invalid_pool.replace(invalid_pool.find("8 8388608 0"),11,"8 33554432 0");
    reject([&]{load(invalid_pool);});
    auto workspace=[&](const std::string &text,uint64_t program=123,uint64_t backend=456) {
      {std::ofstream out(path);out<<text;}
      return inferfabric::prepared::WorkspacePlan::read(path,program,backend);
    };
    if(workspace("IFWORKSPACE1 123 456 1024").bytes!=1024 || workspace("IFWORKSPACE1 123 456 0").bytes!=0)
      throw std::runtime_error("valid workspace plan changed");
    reject([&]{workspace("IFWORKSPACE1 123 456 1024",124);});
    reject([&]{workspace("IFWORKSPACE1 123 456 1024",123,457);});
    reject([&]{workspace("IFWORKSPACE1 123 456 1099511627777");});
    reject([&]{workspace("IFWORKSPACE1 123 456 -1");});
    reject([&]{workspace("IFWORKSPACE1 123 456 1024 extra");});
    reject([&]{workspace("IFWORKSPACE1 123 456");});
    std::remove(path);
    std::cout<<"frozen arena: valid layout and "<<rejected<<" negative contracts passed\n";
  } catch(const std::exception &e) {std::remove(path);std::cerr<<e.what()<<'\n';return 1;}
}
