#pragma once
#include "initialization_plan.h"
#include <acl/acl_prof.h>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <vector>

// Offline probe only: capture one completed serving batch after engine warmup.
// No device synchronization, tensor readback or execution-library changes.
class BatchProfile {
  aclprofConfig *config_ = nullptr;
  bool initialized_ = false, running_ = false, completed_ = false;
  static void checked(int status) {
    if (status) throw std::runtime_error("serving batch profiler failed: " + std::to_string(status));
  }
public:
  void begin(const std::string &bundle) {
    const char *directory = std::getenv("INFERFABRIC_BATCH_PROFILE_DIR");
    if (!directory || completed_) return;
    if (!std::filesystem::create_directory(directory))
      throw std::runtime_error("batch profile directory must be new");
    std::vector<uint32_t> devices;
    for (uint32_t rank = 0; std::filesystem::exists(bundle + "/rank-" + std::to_string(rank) + ".bin"); ++rank)
      devices.push_back(inferfabric::initialization::Plan::read(bundle + "/rank-" + std::to_string(rank) + ".bin").device);
    if (devices.empty()) throw std::runtime_error("profile has no rank devices");
    std::string path(directory);
    checked(aclprofInit(path.c_str(), path.size())); initialized_ = true;
    config_ = aclprofCreateConfig(devices.data(), devices.size(), ACL_AICORE_NONE, nullptr,
        ACL_PROF_ACL_API | ACL_PROF_TASK_TIME | ACL_PROF_RUNTIME_API |
        ACL_PROF_HCCL_TRACE | ACL_PROF_AICPU | ACL_PROF_GE_API_L0);
    if (!config_) throw std::runtime_error("batch profiler configuration failed");
    checked(aclprofStart(config_)); running_ = true;
  }
  bool finish() {
    if (!running_) return false;
    checked(aclprofStop(config_)); running_ = false; completed_ = true;
    checked(aclprofDestroyConfig(config_)); config_ = nullptr;
    checked(aclprofFinalize()); initialized_ = false;
    return true;
  }
  ~BatchProfile() {
    if (running_) aclprofStop(config_);
    if (config_) aclprofDestroyConfig(config_);
    if (initialized_) aclprofFinalize();
  }
};
