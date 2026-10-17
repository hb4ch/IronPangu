#include "bounded_buffers.h"
#include <array>
#include <iostream>
#include <set>
#include <stdexcept>

namespace {
struct Block {size_t slot,bytes;};
using Free=std::map<size_t,std::vector<Block *>>;
void require(bool condition,const char *message) {
  if(!condition)throw std::runtime_error(message);
}
}
int main() {
  try {
    std::array<Block,5> blocks{{{0,64},{1,128},{2,128},{3,4096},{4,8192}}};
    Free free;
    for(auto &b:blocks)free[b.bytes].push_back(&b);
    auto all=[](Block *){return true;};
    auto *a=inferfabric::prepared::take_bounded_buffer(free,65,all);
    require(a==&blocks[2],"equal capacity LIFO");
    auto *b=inferfabric::prepared::take_bounded_buffer(free,65,all);
    require(b==&blocks[1] && b!=a,"live lease reused");
    free[a->bytes].push_back(a);
    require(inferfabric::prepared::take_bounded_buffer(free,127,all)==a,"released larger lease not reused");
    const auto before=free;
    require(!inferfabric::prepared::take_bounded_buffer(free,8193,all),"undersized slot returned");
    require(!inferfabric::prepared::take_bounded_buffer(free,0,all),"zero lease accepted");
    require(free==before,"failed selection changed ownership");
    auto only_large=[](Block *p){return p->slot==4;};
    require(inferfabric::prepared::take_bounded_buffer(free,1,only_large)==&blocks[4],"foreign instruction slot consumed");
    require(free[64].size()==1 && free[4096].size()==1,"skipped slots lost");

    // A previous tail can occupy the sampled address. The frozen plan must
    // permit an equivalent free slot, while the allocator still excludes the
    // live lease and does not invent storage or borrow another capacity class.
    Block sampled{10,25165824}, equivalent{11,25165824}, larger{12,33554432};
    Free changed_order{{equivalent.bytes,{&equivalent}}, {larger.bytes,{&larger}}};
    auto sampled_only=[](Block *p){return p->slot==10;};
    auto capacity_class=[](Block *p){return p->slot==10 || p->slot==11;};
    require(!inferfabric::prepared::take_bounded_buffer(changed_order,15728640,sampled_only),"sampled binding unexpectedly usable");
    require(inferfabric::prepared::take_bounded_buffer(changed_order,15728640,capacity_class)==&equivalent,"equivalent planned slot not reused");
    require(!inferfabric::prepared::take_bounded_buffer(changed_order,15728640,capacity_class),"live or foreign slot borrowed");
    changed_order[sampled.bytes].push_back(&sampled);
    require(inferfabric::prepared::take_bounded_buffer(changed_order,15728640,capacity_class)==&sampled,"released sampled slot unavailable");

    // Every valid query tail fits a maximum-query temporary. Other instructions'
    // buffers and live persistent/decode-output leases must remain unavailable.
    for(size_t queries=1;queries<=4096;++queries) {
      Block persistent{0,4096*576*2}, first{1,4096*576*2}, second{2,4096*576*2};
      Block foreign{3,queries*576*2};
      Free arena{{first.bytes,{&first,&second}}};
      arena[foreign.bytes].push_back(&foreign);
      auto allowed=[](Block *p){return p->slot==1 || p->slot==2;};
      auto *x=inferfabric::prepared::take_bounded_buffer(arena,queries*576*2,allowed);
      auto *y=inferfabric::prepared::take_bounded_buffer(arena,queries*576*2,allowed);
      require(x && y && x!=y && x!=&persistent && y!=&persistent,"tail lease overlap");
      require(!inferfabric::prepared::take_bounded_buffer(arena,queries*576*2,allowed),"exhaustion borrowed foreign storage");
      arena[x->bytes].push_back(x);
      require(inferfabric::prepared::take_bounded_buffer(arena,queries*576*2,allowed)==x,"tail release capacity changed");
      require(arena[foreign.bytes].size()==1 && arena[foreign.bytes][0]==&foreign,"foreign tail storage modified");
    }
    std::cout<<"bounded leases: all 4096 tails, ownership, reuse and exhaustion passed\n";
  } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
