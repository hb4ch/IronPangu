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
  std::shared_ptr<void> lease_;
  void *alias_data_ = nullptr;
  aclTensor *alias_tensor_ = nullptr;
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
  static Tensor alias(const openpangu::DeviceTensor &source,
                      aclDataType dtype) {
    if (!source.owner || source.owner != prepared::active ||
        source.type != dtype)
      throw std::runtime_error("resident alias ownership/dtype");
    Tensor out;
    out.count_ = source.bytes / element_bytes(dtype);
    out.dtype_ = dtype;
    out.lease_ = source.allocation;
    out.alias_data_ = source.data;
    out.alias_tensor_ =
        source.owner->descriptor(source.data, {int64_t(out.count_)}, dtype);
    return out;
  }
  Tensor view(size_t start, size_t count) const {
    if (!prepared::active || !count || start > count_ || count > count_ - start)
      throw std::runtime_error("resident view bounds/context");
    Tensor out = *this;
    out.count_ = count;
    out.alias_data_ =
        static_cast<char *>(data()) + start * element_bytes(dtype_);
    out.alias_tensor_ =
        prepared::active->descriptor(out.alias_data_, {int64_t(count)}, dtype_);
    return out;
  }
  size_t size() const { return count_; }
  size_t bytes() const { return count_ * element_bytes(dtype_); }
  aclDataType dtype() const { return dtype_; }
  void *data() const {
    if (alias_data_)
      return alias_data_;
    if (!storage_)
      throw std::runtime_error("empty resident value");
    return storage_->data;
  }
  aclTensor *tensor() const {
    if (alias_tensor_)
      return alias_tensor_;
    if (!storage_)
      throw std::runtime_error("empty resident value");
    return storage_->tensor;
  }
};
// Copies and recycling are ordered on the prepared context stream.
// No host access to tensor contents is available here.
inline void copy(void *dst, const void *src, size_t bytes) {
  if (prepared::active) {
    prepared::active->copy(dst, src, bytes);
    return;
  }
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
  if (prepared::active)
    return x.view(start, count);
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
