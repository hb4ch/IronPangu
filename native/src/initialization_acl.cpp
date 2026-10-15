#include "initialization_acl.h"
#include <acl/acl.h>
#include <stdexcept>
namespace inferfabric::initialization {
namespace {
void check(aclError error) {
  if (error != ACL_SUCCESS)
    throw std::runtime_error("initialization ACL error " +
                             std::to_string(error));
}
class AclMemory final : public DeviceMemory {
public:
  void *allocate(uint64_t bytes) override {
    void *pointer = nullptr;
    check(aclrtMalloc(&pointer, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    return pointer;
  }
  void release(void *pointer) noexcept override { (void)aclrtFree(pointer); }
  void upload(void *destination, const void *source, size_t bytes) override {
    check(aclrtMemcpy(destination, bytes, source, bytes,
                      ACL_MEMCPY_HOST_TO_DEVICE));
  }
  void readback(void *destination, const void *source, size_t bytes) override {
    check(aclrtMemcpy(destination, bytes, source, bytes,
                      ACL_MEMCPY_DEVICE_TO_HOST));
  }
};
} // namespace
std::shared_ptr<DeviceMemory> acl_device_memory() {
  return std::make_shared<AclMemory>();
}
} // namespace inferfabric::initialization
