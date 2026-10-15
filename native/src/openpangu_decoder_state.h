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
  AttentionState()
      : query({1, 6, 576}, ACL_BF16, 2), key({32, 16, 1, 576}, ACL_BF16, 2),
        value({1, 16, 1, 512}, ACL_BF16, 2), indices({1, 1, 512}, ACL_INT32, 4),
        table({1, 32}, ACL_INT32, 4), qlen({1}, ACL_INT32, 4),
        klen({1}, ACL_INT32, 4), sink({128, 1, 576}, ACL_BF16, 2),
        sinkvalue({128, 1, 512}, ACL_BF16, 2), out({1, 6, 512}, ACL_BF16, 2) {
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
