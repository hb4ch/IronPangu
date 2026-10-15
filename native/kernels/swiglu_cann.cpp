// Editable compiled implementation. A custom Ascend C host launcher can expose
// the same contract without changing the model definition or decoder call site.
#include "inferfabric_kernel_v1.h"
#include <acl/acl.h>
#include <aclnnop/aclnn_swi_glu.h>
#include <memory>
namespace {
struct State {
  aclTensor *input = nullptr, *output = nullptr;
  aclOpExecutor *executor = nullptr;
  ~State() {
    if (input)
      aclDestroyTensor(input);
    if (output)
      aclDestroyTensor(output);
  }
};
int32_t prepare(const IFKernelTensorV1 *inputs, uint32_t ni,
                const IFKernelTensorV1 *outputs, uint32_t no,
                IFKernelPreparedV1 *out) {
  if (!out)
    return -1;
  *out = {};
  try {
    if (ni != 1 || no != 1 || inputs[0].dtype != 2 || outputs[0].dtype != 2 ||
        inputs[0].rank != 2 || outputs[0].rank != 2 ||
        inputs[0].shape[0] != outputs[0].shape[0] || inputs[0].shape[1] % 2 ||
        inputs[0].shape[1] / 2 != outputs[0].shape[1])
      return -2;
    auto s = std::make_unique<State>();
    const auto &x = inputs[0], &y = outputs[0];
    s->input = aclCreateTensor(x.shape, 2, ACL_BF16, x.strides, 0,
                               ACL_FORMAT_ND, x.shape, 2, x.data);
    s->output = aclCreateTensor(y.shape, 2, ACL_BF16, y.strides, 0,
                                ACL_FORMAT_ND, y.shape, 2, y.data);
    if (!s->input || !s->output)
      return -3;
    uint64_t workspace = 0;
    int status = aclnnSwiGluGetWorkspaceSize(s->input, -1, s->output,
                                             &workspace, &s->executor);
    if (status)
      return status;
    out->workspace_bytes = workspace;
    out->workspace_alignment = 64;
    out->state = s.release();
    return 0;
  } catch (...) {
    return -4;
  }
}
int32_t enqueue(void *state, void *workspace, uint64_t bytes, void *stream) {
  return aclnnSwiGlu(workspace, bytes, static_cast<State *>(state)->executor,
                     static_cast<aclrtStream>(stream));
}
void release(void *state) { delete static_cast<State *>(state); }
const IFKernelV1 api{sizeof(IFKernelV1),
                     1,
                     "inferfabric.swiglu.bf16.v1",
                     "ascend",
                     prepare,
                     enqueue,
                     release};
} // namespace
extern "C" const IFKernelV1 *inferfabric_kernel_v1(uint32_t version) {
  return version == 1 ? &api : nullptr;
}
