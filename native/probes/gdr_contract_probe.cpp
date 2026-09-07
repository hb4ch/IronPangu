// Read-only kernel-contract qualification: prepares metadata, never launches
// GDR.
#include <acl/acl.h>
#include <aclnnop/aclnn_recurrent_gated_delta_rule.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
static void check(int status) {
  if (status)
    throw std::runtime_error("ACL status " + std::to_string(status));
}
static int query(aclDataType state_dtype) {
  std::vector<void *> buffers;
  std::vector<aclTensor *> tensors;
  aclOpExecutor *executor = nullptr;
  auto cleanup = [&] {
    if (executor)
      aclDestroyAclOpExecutor(executor);
    for (auto *t : tensors)
      aclDestroyTensor(t);
    for (auto *p : buffers)
      aclrtFree(p);
  };
  try {
    auto tensor = [&](std::vector<int64_t> shape, aclDataType dtype) {
      uint64_t count = 1;
      for (auto d : shape)
        count *= d;
      uint64_t bytes = count * (dtype == ACL_BF16 ? 2 : 4);
      void *ptr = nullptr;
      check(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
      buffers.push_back(ptr);
      std::vector<unsigned char> zero(bytes, 0);
      check(aclrtMemcpy(ptr, bytes, zero.data(), bytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
      std::vector<int64_t> stride(shape.size(), 1);
      for (size_t i = shape.size() - 1; i > 0; --i)
        stride[i - 1] = stride[i] * shape[i];
      auto *t =
          aclCreateTensor(shape.data(), shape.size(), dtype, stride.data(), 0,
                          ACL_FORMAT_ND, shape.data(), shape.size(), ptr);
      if (!t)
        throw std::runtime_error("tensor creation");
      tensors.push_back(t);
      return t;
    };
    auto *q = tensor({1, 1, 128}, ACL_BF16);
    auto *k = tensor({1, 1, 128}, ACL_BF16);
    auto *v = tensor({1, 1, 128}, ACL_BF16);
    auto *beta = tensor({1, 1}, ACL_BF16);
    auto *state = tensor({1, 1, 128, 128}, state_dtype);
    auto *length = tensor({1}, ACL_INT32);
    int32_t one = 1;
    check(aclrtMemcpy(buffers.back(), 4, &one, 4, ACL_MEMCPY_HOST_TO_DEVICE));
    auto *index = tensor({1}, ACL_INT32);
    auto *g = tensor({1, 1}, ACL_FLOAT);
    auto *out = tensor({1, 1, 128}, ACL_BF16);
    uint64_t workspace = 0;
    int status = aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(
        q, k, v, beta, state, length, index, g, nullptr, nullptr,
        0.08838834764831845f, out, &workspace, &executor);
    if (status == 0)
      check(aclSetAclOpExecutorRepeatable(executor));
    cleanup();
    return status;
  } catch (...) {
    cleanup();
    throw;
  }
}
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  aclrtContext context = nullptr;
  try {
    check(aclInit(nullptr));
    check(aclrtCreateContext(&context, std::atoi(argv[1])));
    int fp32 = query(ACL_FLOAT);
    int bf16 = query(ACL_BF16);
    check(aclrtDestroyContext(context));
    context = nullptr;
    check(aclFinalize());
    std::printf("{\"fp32_workspace_status\":%d,\"bf16_workspace_status\":%d,"
                "\"kernel_executed\":false}\n",
                fp32, bf16);
    return bf16 == 0 ? 0 : 1;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    if (context)
      aclrtDestroyContext(context);
    aclFinalize();
    return 1;
  }
}
