#pragma once
#include "openpangu_ops.h"
namespace pangu {
struct MomeState {
  openpangu::DeviceTensor input, weight, state, starts, slots, computed, out;
  explicit MomeState(int channels)
      : input({1, channels}, ACL_BF16, 2), weight({3, channels}, ACL_BF16, 2),
        state({1, 2, channels}, ACL_BF16, 2), starts({2}, ACL_INT32, 4),
        slots({1}, ACL_INT32, 4), computed({1}, ACL_INT32, 4),
        out({1, channels}, ACL_BF16, 2) {
    starts.upload(std::vector<int32_t>{0, 1});
    slots.upload(std::vector<int32_t>{0});
    state.upload(std::vector<uint16_t>(2 * channels, 0));
  }
};
struct AttentionState {
  openpangu::DeviceTensor query, key, value, indices, table, qlen, klen, sink,
      sinkvalue, out;
  unsigned cache_bits, cache_capacity;
  std::unique_ptr<openpangu::DeviceTensor> scales;
  explicit AttentionState(unsigned bits=16, unsigned capacity=512)
      : query({1, 6, 576}, ACL_BF16, 2), key(bits==16?std::vector<int64_t>{32,16,1,576}:std::vector<int64_t>{int64_t(capacity)*576*bits/8}, bits==16?ACL_BF16:ACL_UINT8, bits==16?2:1),
        value({1, 16, 1, 512}, ACL_BF16, 2), indices({1, 1, 512}, ACL_INT32, 4),
        table({1, 32}, ACL_INT32, 4), qlen({1}, ACL_INT32, 4),
        klen({1}, ACL_INT32, 4), sink({128, 1, 576}, ACL_BF16, 2),
        sinkvalue({128, 1, 512}, ACL_BF16, 2), out({1, 6, 512}, ACL_BF16, 2), cache_bits(bits), cache_capacity(bits==16?512:capacity) {
    if(bits!=16)scales=std::make_unique<openpangu::DeviceTensor>(std::vector<int64_t>{capacity},ACL_FLOAT,4);
    std::vector<int32_t> blocks(32);
    for (int i = 0; i < 32; ++i)
      blocks[i] = i;
    table.upload(blocks);
    qlen.upload(std::vector<int32_t>{1});
    value.upload(std::vector<uint16_t>(16 * 512, 0));
    sinkvalue.upload(std::vector<uint16_t>(128 * 512, 0));
  }
};
} // namespace pangu
