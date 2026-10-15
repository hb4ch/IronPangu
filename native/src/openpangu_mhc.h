#pragma once
#include "openpangu_ops.h"
#include <aclnn_ai_infra_manifold_constrained_hyper_connection_post.h>
#include <aclnn_ai_infra_manifold_constrained_hyper_connection_pre.h>
#include <aclnn_manifold_constrained_hyper_connection_sinkhorn_enhance.h>
#include <dlfcn.h>
namespace pangu {
// Both vendor packages have libcust_opapi.so as SONAME. Resolve training
// symbols from its explicit handle rather than relying on the inference
// library's scope.
struct TrainingApi {
  void *handle = nullptr;
  template <class T> T symbol(const char *name) {
    void *address = dlsym(handle, name);
    if (!address)
      throw std::runtime_error(std::string("missing training op: ") + name);
    return reinterpret_cast<T>(address);
  }
  decltype(&aclnnAiInfraManifoldConstrainedHyperConnectionPreGetWorkspaceSize)
      pre_size;
  decltype(&aclnnAiInfraManifoldConstrainedHyperConnectionPre) pre;
  decltype(&aclnnManifoldConstrainedHyperConnectionSinkhornEnhanceGetWorkspaceSize)
      sink_size;
  decltype(&aclnnManifoldConstrainedHyperConnectionSinkhornEnhance) sink;
  decltype(&aclnnAiInfraManifoldConstrainedHyperConnectionPostGetWorkspaceSize)
      post_size;
  decltype(&aclnnAiInfraManifoldConstrainedHyperConnectionPost) post;
  TrainingApi() {
    handle = dlopen(OMNI_TRAINING_OPAPI_PATH, RTLD_NOW | RTLD_LOCAL);
    if (!handle)
      throw std::runtime_error(dlerror());
    pre_size = symbol<decltype(pre_size)>(
        "aclnnAiInfraManifoldConstrainedHyperConnectionPreGetWorkspaceSize");
    pre = symbol<decltype(pre)>(
        "aclnnAiInfraManifoldConstrainedHyperConnectionPre");
    sink_size =
        symbol<decltype(sink_size)>("aclnnManifoldConstrainedHyperConnectionSin"
                                    "khornEnhanceGetWorkspaceSize");
    sink = symbol<decltype(sink)>(
        "aclnnManifoldConstrainedHyperConnectionSinkhornEnhance");
    post_size = symbol<decltype(post_size)>(
        "aclnnAiInfraManifoldConstrainedHyperConnectionPostGetWorkspaceSize");
    post = symbol<decltype(post)>(
        "aclnnAiInfraManifoldConstrainedHyperConnectionPost");
  }
  ~TrainingApi() {
    if (handle)
      dlclose(handle);
  }
};
inline TrainingApi &training_api() {
  static TrainingApi api;
  return api;
}
struct MhcState {
  openpangu::DeviceTensor x, phi, alpha, beta, input, post, res, rms, mix,
      pre_weights, sink, branch, out;
  MhcState()
      : x({1, 4, 2560}, ACL_BF16, 2), phi({24, 10240}, ACL_FLOAT, 4),
        alpha({3}, ACL_FLOAT, 4), beta({24}, ACL_FLOAT, 4),
        input({1, 2560}, ACL_BF16, 2), post({1, 4}, ACL_FLOAT, 4),
        res({1, 4, 4}, ACL_FLOAT, 4), rms({1}, ACL_FLOAT, 4),
        mix({1, 24}, ACL_FLOAT, 4), pre_weights({1, 4}, ACL_FLOAT, 4),
        sink({1, 4, 4}, ACL_FLOAT, 4), branch({1, 2560}, ACL_BF16, 2),
        out({1, 4, 2560}, ACL_BF16, 2) {}
};
} // namespace pangu
