#pragma once
#include "openpangu_ops.h"
#include "page_binding.h"
namespace pangu {
struct MomeState {
  openpangu::DeviceTensor input, weight, state, starts, computed, out;
  explicit MomeState(int channels, unsigned residents=1)
      : input({1, channels}, ACL_BF16, 2), weight({3, channels}, ACL_BF16, 2),
        state({residents, 2, channels}, ACL_BF16, 2), starts({2}, ACL_INT32, 4),
        computed({1}, ACL_INT32, 4),
        out({1, channels}, ACL_BF16, 2) {
    starts.upload(std::vector<int32_t>{0, 1});
    state.upload(std::vector<uint16_t>(size_t(residents) * 2 * channels, 0));
  }
};
struct AttentionState {
  openpangu::DeviceTensor query, key, value, indices, table, qlen, klen, sink,
      sinkvalue, out;
  unsigned cache_bits, cache_capacity, physical_capacity;
  bool packed_c8;
  unsigned logical_context;
  inferfabric::state::ResidentPages page_binding;
  std::unique_ptr<openpangu::DeviceTensor> scales;
  explicit AttentionState(unsigned bits=16, unsigned capacity=512, bool packed=false, unsigned residents=1, bool shared=false, unsigned context=0)
      : query({1, 6, 576}, ACL_BF16, 2), key(packed?std::vector<int64_t>{residents*capacity/32,32,1,656}:(bits==16?std::vector<int64_t>{residents*capacity/16,16,1,576}:std::vector<int64_t>{int64_t(capacity)*576*bits/8}), packed?ACL_INT8:(bits==16?ACL_BF16:ACL_UINT8), bits==16?2:1),
        value({1, 16, 1, 512}, ACL_BF16, 2), indices({1, 1, 512}, ACL_INT32, 4),
        table({1, packed ? capacity/32 : (context?(context+15)/16:capacity/16)}, ACL_INT32, 4), qlen({1}, ACL_INT32, 4),
        klen({1}, ACL_INT32, 4), sink({128, 1, 576}, ACL_BF16, 2),
        sinkvalue({128, 1, 512}, ACL_BF16, 2), out({1, 6, 512}, ACL_BF16, 2), cache_bits(bits), cache_capacity(capacity), physical_capacity(cache_capacity*residents), packed_c8(packed),logical_context(context),
        page_binding(packed?32:16,cache_capacity,residents,shared) {
    if(residents>1 && bits!=16 && !packed)throw std::runtime_error("resident row-quantized cache unsupported");
    if(bits!=16 && !packed)scales=std::make_unique<openpangu::DeviceTensor>(std::vector<int64_t>{capacity},ACL_FLOAT,4);
    std::vector<int32_t> blocks(table.bytes/sizeof(int32_t));
    for (size_t i = 0; i < blocks.size(); ++i)
      blocks[i] = int32_t(i%(capacity/(packed?32:16)));
    table.upload(blocks);
    qlen.upload(std::vector<int32_t>{1});
    value.upload(std::vector<uint16_t>(16 * 512, 0));
    sinkvalue.upload(std::vector<uint16_t>(128 * 512, 0));
  }
  uint64_t append(uint32_t position) const {
    if(logical_context && position>=logical_context)throw std::runtime_error("cache absolute position bounds");
    return page_binding.append(logical_context?position%cache_capacity:position);
  }
  std::vector<int32_t> blocks() const {
    return page_binding.table(table.bytes/sizeof(int32_t),logical_context!=0);
  }
};
} // namespace pangu
