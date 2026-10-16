#pragma once
#include <acl/acl_prof.h>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <vector>
namespace inferfabric::profiling {
// Profiling-only rendezvous; disabled runs acquire no extra barriers.
class Capture {
  std::mutex mutex_;
  std::condition_variable cv_;
  size_t arrived_ = 0, generation_ = 0;
  size_t ranks_;
  aclprofConfig *config_ = nullptr;

public:
  std::string directory;
  explicit Capture(const std::vector<uint32_t> &devices)
      : ranks_(devices.size()) {
    const char *dir = std::getenv("INFERFABRIC_PROFILE_DIR");
    if (!dir)
      return;
    directory = dir;
    std::filesystem::create_directories(directory);
    if (std::getenv("INFERFABRIC_PROFILE_VENDOR")) {
      auto path = directory + "/vendor";
      std::filesystem::create_directories(path);
      if (aclprofInit(path.c_str(), path.size()))
        throw std::runtime_error("profiler init");
      auto ids = devices;
      config_ = aclprofCreateConfig(
          ids.data(), ids.size(), ACL_AICORE_PIPE_UTILIZATION, nullptr,
          ACL_PROF_ACL_API | ACL_PROF_TASK_TIME | ACL_PROF_AICORE_METRICS |
              ACL_PROF_RUNTIME_API | ACL_PROF_HCCL_TRACE | ACL_PROF_AICPU |
              ACL_PROF_GE_API_L0);
      if (!config_)
        throw std::runtime_error("profiler config");
    }
  }
  void rendezvous(bool start) {
    if (directory.empty())
      return;
    std::unique_lock<std::mutex> lock(mutex_);
    auto gen = generation_;
    if (++arrived_ == ranks_) {
      if (config_ && (start ? aclprofStart(config_) : aclprofStop(config_)))
        throw std::runtime_error("profiler start/stop");
      arrived_ = 0;
      ++generation_;
      cv_.notify_all();
    } else
      cv_.wait(lock, [&] { return generation_ != gen; });
  }
  void close() {
    if (config_) {
      aclprofDestroyConfig(config_);
      aclprofFinalize();
      config_ = nullptr;
    }
  }
  ~Capture() { close(); }
};
} // namespace inferfabric::profiling
