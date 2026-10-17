#pragma once
#include "resident_tensor.h"

namespace inferfabric::resident {
// Preparation-only layout materialization of immutable parameters. Sources must
// remain immutable and alive for this cache's lifetime. Never use for
// activations. Both the copy and subsequent consumers run on the owning
// prepared stream.
class PreparedWeightViews {
  prepared::Context *owner_ = prepared::active;
  std::map<aclTensor *, std::unique_ptr<openpangu::DeviceTensor>> views_;

public:
  aclTensor *get(void *pointer, int64_t storage_elements,
                 const std::vector<int64_t> &shape,
                 const std::vector<int64_t> &strides, aclDataType dtype) {
    if (!owner_ || owner_ != prepared::active)
      throw std::runtime_error("prepared weight view context mismatch");
    auto *source = owner_->strided_descriptor(pointer, storage_elements, shape,
                                              strides, dtype, true);
    auto found = views_.find(source);
    if (found != views_.end())
      return found->second->tensor;
    auto output = std::make_unique<openpangu::DeviceTensor>(
        shape, dtype, element_bytes(dtype));
    uint64_t bytes = 0;
    aclOpExecutor *executor = nullptr;
    prepared::check(prepared::query("immutable_layout_copy", false,
                                    aclnnCastGetWorkspaceSize, source, dtype,
                                    output->tensor, &bytes, &executor),
                    "prepare immutable weight view");
    prepared::check(
        aclnnCast(owner_->workspace(bytes), bytes, executor, owner_->stream),
        "materialize immutable weight view");
    auto *result = output->tensor;
    views_.emplace(source, std::move(output));
    return result;
  }
};
} // namespace inferfabric::resident
