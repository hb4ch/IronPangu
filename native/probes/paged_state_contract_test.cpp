#include "../include/paged_state_contract.h"
#include "../src/page_binding.h"
#include <cassert>
#include <iostream>
#include <numeric>

template<class F> void rejects(F action) {
  bool rejected=false;
  try {action();} catch(const std::runtime_error &) {rejected=true;}
  if(!rejected)throw std::runtime_error("invalid paged state contract accepted");
}
int main() {
  using namespace inferfabric::execution;
  Plan plan;
  for(auto context:{1u,511u,512u,513u,2048u,65536u}) {
    plan.context=context;
    for(const auto &op:{"cache_c8_norm_rope_append","index_cache_append","cache_norm_rope_append"}) {
      Node node;node.op=op;node.attrs["page_tokens"]="64";
      auto contract=paged_state_contract(plan,node);
      uint32_t page=node.op=="index_cache_append"?64:node.op=="cache_c8_norm_rope_append"?32:16;
      uint32_t capacity=page==16?512:((context+page-1)/page)*page;
      assert(contract.page_tokens==page && contract.capacity_tokens==capacity);
      auto count=contract.table_entries();
      contract.validate_count(count,false);contract.validate_count(1,true);
      rejects([&]{contract.validate_count(0,true);});
      rejects([&]{contract.validate_count(count+1,true);});
      if(count>1)rejects([&]{contract.validate_count(count-1,false);});
      inferfabric::state::ResidentPages pages(page,capacity,2,true);
      pages.release(0);pages.release(1);
      std::vector<int32_t> ids(count);
      // Borrow every page from the other initial partition in reverse order.
      for(uint32_t i=0;i<count;++i)ids[i]=int32_t(2*count-1-i);
      pages.bind(ids);
      assert(pages.append(capacity-1)==uint64_t(count)*page+page-1);
      pages.select(1);
      rejects([&]{pages.bind({ids[0]});});
      pages.select(0);
      auto duplicate=ids;
      if(count>1){duplicate.back()=duplicate.front();rejects([&]{pages.bind(duplicate);});}
      assert(pages.append(0)==uint64_t(ids[0])*page);
      pages.release(0);pages.select(1);pages.bind(ids);
      assert(pages.append(capacity-1)==uint64_t(count)*page+page-1);
      rejects([&]{pages.append(capacity);});
    }
  }
  Node invalid;invalid.op="mome";
  rejects([&]{paged_state_contract(plan,invalid);});
  invalid.op="index_cache_append";invalid.attrs["page_tokens"]="0";
  rejects([&]{paged_state_contract(plan,invalid);});
  invalid.attrs["page_tokens"]="64";plan.context=UINT32_MAX;
  rejects([&]{paged_state_contract(plan,invalid);});
  plan.context=0;rejects([&]{paged_state_contract(plan,invalid);});
  std::cout<<"planned binding capacities and ownership through 65536 tokens passed; model context gate unchanged\n";
}
