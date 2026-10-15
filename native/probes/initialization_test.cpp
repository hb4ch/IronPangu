#include "initialization_plan.h"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <unistd.h>
using namespace inferfabric::initialization;
void require(bool value) {
  if (!value)
    throw std::runtime_error("test assertion failed");
}
struct Memory : DeviceMemory {
  std::set<void *> allocations;
  int uploads = 0, fail_upload = -1;
  void *allocate(uint64_t bytes) override {
    auto p = new char[bytes];
    allocations.insert(p);
    return p;
  }
  void release(void *p) noexcept override {
    allocations.erase(p);
    delete[] static_cast<char *>(p);
  }
  void upload(void *d, const void *s, size_t n) override {
    if (++uploads == fail_upload)
      throw std::runtime_error("injected upload failure");
    std::memcpy(d, s, n);
  }
  void readback(void *d, const void *s, size_t n) override {
    std::memcpy(d, s, n);
  }
};
template <class F> void rejects(F f) {
  bool failed = false;
  try {
    f();
  } catch (const std::exception &) {
    failed = true;
  }
  require(failed);
}
int main(int argc, char **argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--validate") {
      auto p = Plan::read(argv[2]);
      std::cout << p.parameters.size() << " valid parameters\n";
      return 0;
    }
    auto root = std::filesystem::temp_directory_path() /
                ("inferfabric-init-test-" + std::to_string(getpid()));
    std::filesystem::create_directory(root);
    struct Cleanup {
      std::filesystem::path path;
      ~Cleanup() { std::filesystem::remove_all(path); }
    } cleanup{root};
    // Explicit little-endian BF16 [1,2,3,4], spanning multiple staging chunks.
    unsigned char values[] = {0x80, 0x3f, 0, 0x40, 0x40, 0x40, 0x80, 0x40};
    {
      std::ofstream f(root / "weights", std::ios::binary);
      f.write(reinterpret_cast<char *>(values), 8);
    }
    Plan p{
        17,
        22,
        std::string(64, 'a'),
        {{0, "raw", DType::Bf16, {4}}, {1, "cast", DType::F32, {4}}},
        {{0, "weights", 0, 8, DType::Bf16}, {1, "weights", 0, 8, DType::Bf16}}};
    auto memory = std::make_shared<Memory>();
    Options options;
    options.staging_bytes = 4;
    options.verify_upload = true;
    {
      auto store = Executor::execute(p, root.string(), memory, options);
      require(store.size() == 2 && store.source_bytes == 16 &&
              store.resident_bytes == 24);
      require(std::memcmp(store.at(0).data, values, 8) == 0);
      float expected[] = {1, 2, 3, 4};
      require(std::memcmp(store.at(1).data, expected, 16) == 0);
      require(memory->uploads == 4);
      rejects([&] { store.at(2); });
    }
    require(memory->allocations.empty());
    memory->uploads = 0;
    memory->fail_upload = 3;
    rejects([&] { Executor::execute(p, root.string(), memory, options); });
    require(memory->allocations.empty()); // Current and already published
                                          // internal allocations unwind.
    memory->fail_upload = -1;
    options.resident_budget = 23;
    memory->uploads = 0;
    rejects([&] { Executor::execute(p, root.string(), memory, options); });
    require(memory->allocations.empty() && memory->uploads == 0);
    options.resident_budget = 24;
    auto bad = p;
    bad.instructions[1].offset = 8;
    rejects([&] { Executor::execute(bad, root.string(), memory, options); });
    require(memory->allocations.empty() && memory->uploads == 0);
    bad = p;
    bad.instructions[1].destination = 0;
    rejects([&] { bad.validate(); });
    bad = p;
    bad.parameters[1].dtype = DType::I8;
    rejects([&] { bad.validate(); });
    bad = p;
    bad.instructions[0].shard = "../weights";
    rejects([&] { bad.validate(); });
    // Nonfinite conversion fails without publishing a partially initialized
    // store.
    {
      std::ofstream f(root / "weights", std::ios::binary);
      unsigned char inf[] = {0x80, 0x7f, 0, 0, 0, 0, 0, 0};
      f.write(reinterpret_cast<char *>(inf), 8);
    }
    rejects([&] { Executor::execute(p, root.string(), memory, options); });
    require(memory->allocations.empty());
    std::cout
        << "initialization executor: streaming copy/cast, generic rank, "
           "budget, bounds, producer validation, failure cleanup passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
