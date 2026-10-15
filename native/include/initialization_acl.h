#pragma once
#include "initialization_plan.h"
namespace inferfabric::initialization {
// Caller selects and retains the ACL device/context before initialization and
// until all returned parameters are destroyed. This adapter never owns aclInit.
std::shared_ptr<DeviceMemory> acl_device_memory();
} // namespace inferfabric::initialization
