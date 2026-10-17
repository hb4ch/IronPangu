#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace inferfabric::state {
// Page-aligned retention for a chunk whose cache writes precede its queries.
// The table keeps absolute logical page numbers; physical pages are recycled.
struct SlidingWindowLayout {
  uint32_t context, window, chunk, page_tokens, physical_pages;
  SlidingWindowLayout(uint32_t c,uint32_t w,uint32_t q,uint32_t p)
      :context(c),window(w),chunk(q),page_tokens(p),physical_pages(0) {
    if(!c || !w || !q || !p || q>c)
      throw std::runtime_error("invalid sliding window layout");
    // A window plus the later queries in a chunk can straddle an extra page.
    uint64_t retained=uint64_t(w)+q-1;
    uint64_t pages=(retained+p-2)/p+1;
    uint64_t full=(uint64_t(c)+p-1)/p;
    if(pages>full)pages=full;
    if(pages*p>std::numeric_limits<uint32_t>::max())
      throw std::runtime_error("sliding window capacity overflow");
    physical_pages=uint32_t(pages);
  }
  uint32_t table_entries() const {return uint32_t((uint64_t(context)+page_tokens-1)/page_tokens);}
  uint32_t physical_page(uint32_t logical_page) const {
    if(logical_page>=table_entries())throw std::runtime_error("sliding page bounds");
    return logical_page%physical_pages;
  }
  uint32_t append(uint32_t absolute_position) const {
    if(absolute_position>=context)throw std::runtime_error("sliding append bounds");
    return physical_page(absolute_position/page_tokens)*page_tokens+absolute_position%page_tokens;
  }
};
}
