#ifndef IF_TEST_ABI_VERSION
#define IF_TEST_ABI_VERSION 1
#endif
#include "inferfabric_kernel_v1.h"
#include <new>
struct State {
  float *input, *output;
};
int32_t prepare(const IFKernelTensorV1 *x, uint32_t ni,
                const IFKernelTensorV1 *y, uint32_t no,
                IFKernelPreparedV1 *out) {
  *out = {};
  if (ni != 1 || no != 1 || x[0].dtype != 3 || y[0].dtype != 3 ||
      x[0].shape[0] != 2 || y[0].shape[0] != 2)
    return 1;
  auto s = new (std::nothrow)
      State{static_cast<float *>(x[0].data), static_cast<float *>(y[0].data)};
  if (!s)
    return 2;
  out->state = s;
  out->workspace_bytes = 64;
  out->workspace_alignment = 64;
  return 0;
}
int32_t enqueue(void *state, void *, uint64_t, void *) {
  auto s = static_cast<State *>(state);
  for (int i = 0; i < 2; ++i)
    s->output[i] = s->input[i] * 2;
  return 0;
}
void release(void *state) { delete static_cast<State *>(state); }
extern "C" const IFKernelV1 *inferfabric_kernel_v1(uint32_t version) {
  static IFKernelV1 api{sizeof(IFKernelV1),
                        IF_TEST_ABI_VERSION,
                        "test.scale.v1",
                        "cpu",
                        prepare,
                        enqueue,
                        release};
  return version == 1 ? &api : nullptr;
}
