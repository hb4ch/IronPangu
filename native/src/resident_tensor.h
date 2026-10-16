#pragma once
// Device-only value type: deliberately has no host iterator, read or upload
// API.
#include "openpangu_ops.h"
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_argmax.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_index_select.h>
#include <aclnnop/aclnn_logsoftmax.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_sub.h>
#include <aclnnop/aclnn_topk.h>
namespace inferfabric::resident {
inline size_t element_bytes(aclDataType t) {
  switch (t) {
  case ACL_BF16:
    return 2;
  case ACL_INT8:
    return 1;
  case ACL_FLOAT:
  case ACL_INT32:
    return 4;
  case ACL_INT64:
    return 8;
  default:
    throw std::runtime_error("unsupported resident dtype");
  }
}
class Tensor {
  std::shared_ptr<openpangu::DeviceTensor> storage_;
  size_t count_ = 0;
  aclDataType dtype_ = ACL_FLOAT;

public:
  Tensor() = default;
  explicit Tensor(size_t count, aclDataType dtype = ACL_FLOAT)
      : count_(count), dtype_(dtype) {
    if (!count || count > INT64_MAX / element_bytes(dtype))
      throw std::runtime_error("resident tensor bounds");
    storage_ = std::make_shared<openpangu::DeviceTensor>(
        std::vector<int64_t>{int64_t(count)}, dtype, element_bytes(dtype));
  }
  size_t size() const { return count_; }
  size_t bytes() const { return count_ * element_bytes(dtype_); }
  aclDataType dtype() const { return dtype_; }
  void *data() const {
    if (!storage_)
      throw std::runtime_error("empty resident value");
    return storage_->data;
  }
  aclTensor *tensor() const {
    if (!storage_)
      throw std::runtime_error("empty resident value");
    return storage_->tensor;
  }
};
// Synchronous conservative execution makes ownership explicit. No host access
// to tensor contents is available here. Preparation/arena reuse can replace
// allocation without changing the value or device-only operator contracts.
inline void copy(void *dst, const void *src, size_t bytes) {
  openpangu::checked(
      aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE),
      "resident device copy");
}
inline Tensor from_device(const void *src, size_t count, aclDataType dtype) {
  Tensor out(count, dtype);
  copy(out.data(), src, out.bytes());
  return out;
}
inline Tensor slice(const Tensor &x, size_t start, size_t count) {
  if (start > x.size() || count > x.size() - start)
    throw std::runtime_error("resident slice bounds");
  return from_device(static_cast<const char *>(x.data()) +
                         start * element_bytes(x.dtype()),
                     count, x.dtype());
}
inline Tensor concat(const Tensor &a, const Tensor &b) {
  if (a.dtype() != b.dtype() || a.size() > SIZE_MAX - b.size())
    throw std::runtime_error("resident concat contract");
  Tensor out(a.size() + b.size(), a.dtype());
  copy(out.data(), a.data(), a.bytes());
  copy(static_cast<char *>(out.data()) + a.bytes(), b.data(), b.bytes());
  return out;
}
} // namespace inferfabric::resident
