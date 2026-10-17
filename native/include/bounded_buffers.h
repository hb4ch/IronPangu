#pragma once
#include <cstddef>
#include <map>
#include <vector>

namespace inferfabric::prepared {
// Smallest sufficient capacity, preserving LIFO among eligible equal-size
// slots. Eligibility is supplied by the frozen instruction binding table;
// insufficient or foreign slots remain untouched. Never allocates device memory.
template<class Block, class Eligible>
Block *take_bounded_buffer(std::map<size_t,std::vector<Block *>> &free,
                           size_t bytes, Eligible eligible) {
  if(!bytes)return nullptr;
  for(auto bin=free.lower_bound(bytes);bin!=free.end();++bin) {
    auto &slots=bin->second;
    for(size_t i=slots.size();i>0;--i) {
      auto *block=slots[i-1];
      if(!eligible(block))continue;
      slots.erase(slots.begin()+static_cast<std::ptrdiff_t>(i-1));
      return block;
    }
  }
  return nullptr;
}
}
