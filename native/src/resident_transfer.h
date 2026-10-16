#pragma once
#define IF_RESIDENT_TRANSFER_GUARD_ACTIVE 1
#include <acl/acl.h>
#include <array>
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <vector>
namespace inferfabric::resident {
// Transfer intent is explicit at the boundary. No activation role exists.
enum class TransferRole {
  None,
  ControlInput,
  ParameterPreparation,
  RequestOutput
};
struct Transfers {
  TransferRole role = TransferRole::None;
  std::array<uint64_t, 4> h2d_calls{}, h2d_bytes{}, d2h_calls{}, d2h_bytes{};
  uint64_t rejected = 0;
  void check(size_t bytes, aclrtMemcpyKind kind) {
    bool h2d = kind == ACL_MEMCPY_HOST_TO_DEVICE,
         d2h = kind == ACL_MEMCPY_DEVICE_TO_HOST;
    if (!h2d && !d2h) {
      if (kind != ACL_MEMCPY_DEVICE_TO_DEVICE)
        throw std::runtime_error("unsupported resident copy direction");
      return;
    }
    if (role == TransferRole::None ||
        (d2h && role == TransferRole::ControlInput) ||
        (h2d && role == TransferRole::RequestOutput)) {
      ++rejected;
      throw std::runtime_error(
          "implicit host transfer forbidden by device residency");
    }
    auto i = static_cast<size_t>(role);
    ++(h2d ? h2d_calls : d2h_calls)[i];
    (h2d ? h2d_bytes : d2h_bytes)[i] += bytes;
  }
  void write(std::ostream &out) const {
    const char *names[] = {"forbidden", "control_input",
                           "parameter_preparation", "request_output"};
    out << "{\"format\":\"inferfabric.transfer-audit.v1\",\"activation_host_"
           "transfers\":0,\"rejected\":"
        << rejected << ",\"boundaries\":[";
    for (size_t i = 1; i < 4; ++i) {
      if (i > 1)
        out << ',';
      out << "{\"role\":\"" << names[i] << "\",\"h2d_calls\":" << h2d_calls[i]
          << ",\"h2d_bytes\":" << h2d_bytes[i]
          << ",\"d2h_calls\":" << d2h_calls[i]
          << ",\"d2h_bytes\":" << d2h_bytes[i] << '}';
    }
    out << "]}\n";
  }
};
inline thread_local Transfers transfers;
struct TransferScope {
  TransferRole previous;
  explicit TransferScope(TransferRole role) : previous(transfers.role) {
    transfers.role = role;
  }
  ~TransferScope() { transfers.role = previous; }
  TransferScope(const TransferScope &) = delete;
};
inline aclError copy_checked(void *dst, size_t capacity, const void *src,
                             size_t bytes, aclrtMemcpyKind kind) {
  transfers.check(bytes, kind);
  return aclrtMemcpy(dst, capacity, src, bytes, kind);
}
inline aclError copy_async_checked(void *dst, size_t capacity, const void *src,
                                   size_t bytes, aclrtMemcpyKind kind,
                                   aclrtStream stream) {
  transfers.check(bytes, kind);
  return aclrtMemcpyAsync(dst, capacity, src, bytes, kind, stream);
}
// Helpers require an explicitly classified external input, never a Tensor.
template <class Tensor, class T>
void control_upload(Tensor &dst, const std::vector<T> &x) {
  TransferScope scope(TransferRole::ControlInput);
  dst.upload(x);
}
template <class Tensor, class T>
void parameter_upload(Tensor &dst, const std::vector<T> &x) {
  TransferScope scope(TransferRole::ParameterPreparation);
  dst.upload(x);
}
} // namespace inferfabric::resident
// Apply to this execution translation unit, including inline operator helpers.
// Backend libraries retain their own internal runtime metadata machinery.
#define aclrtMemcpy inferfabric::resident::copy_checked
#define aclrtMemcpyAsync inferfabric::resident::copy_async_checked
