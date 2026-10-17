#pragma once
#include "token_execution_plan.h"
#include "sliding_window_layout.h"
#include <limits>

namespace inferfabric::execution {
// Registered storage layouts, independent of model names. These bounds must
// agree with the serving manifest and the corresponding device-state allocator.
struct PagedStateContract {
  uint32_t page_tokens, capacity_tokens;
  uint32_t logical_tokens=0;
  uint32_t table_entries() const { return capacity_tokens / page_tokens; }
  void validate_count(uint32_t count, bool shared) const {
    if (!count || count > table_entries() || (!shared && count != table_entries()))
      throw std::runtime_error("page binding count exceeds planned state capacity/policy");
  }
};
inline bool is_paged_state(const Node &node) {
  return node.op == "index_cache_append" || node.op == "cache_append" ||
         node.op == "cache_norm_rope_append" || node.op == "cache_c8_norm_rope_append";
}
inline PagedStateContract paged_state_contract(const Plan &plan, const Node &node) {
  if (!plan.context || !is_paged_state(node))
    throw std::runtime_error("unregistered paged state");
  uint32_t page_tokens = 16, capacity = 512;
  if (node.op == "index_cache_append") {
    page_tokens = node.integer("page_tokens");
    if (page_tokens != 64) throw std::runtime_error("unsupported index page layout");
    capacity = plan.context;
  } else if (node.op == "cache_c8_norm_rope_append") {
    page_tokens = 32;
    capacity = plan.context;
  }
  else {
    uint32_t window=0;
    for(const auto &consumer:plan.nodes)if(consumer.op=="latent_attention" && consumer.inputs.size()>1 &&
        plan.nodes.at(consumer.inputs[1]).name==node.name) {
      auto next=consumer.integer("window");
      if(window && window!=next)throw std::runtime_error("cache consumers disagree on window");
      window=next;
    }
    if(!window)throw std::runtime_error("paged cache has no attention window");
    inferfabric::state::SlidingWindowLayout ring(plan.context,window,plan.prefill_tokens,16);
    return {16,ring.physical_pages*16,plan.context};
  }
  uint64_t rounded = ((uint64_t(capacity) + page_tokens - 1) / page_tokens) * page_tokens;
  if (rounded > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("paged state capacity overflow");
  return {page_tokens, uint32_t(rounded)};
}
} // namespace inferfabric::execution
