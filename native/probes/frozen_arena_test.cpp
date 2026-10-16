#include "frozen_arena.h"
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
