#include "pangu_acl.h"
#include <array>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
int main(int argc, char **argv) {
  if (argc == 2 && std::strcmp(argv[1], "--abi") == 0) {
    std::printf("{\"abi_version\":%u}\n", pangu_acl_abi_version());
    return pangu_acl_abi_version() == 2 ? 0 : 1;
  }
  if (argc != 2) {
    std::fprintf(stderr, "usage: acl_graph_probe DEVICE_ID\n");
    return 2;
  }
  pangu_acl_session *s = nullptr;
  uint64_t source = 0, destination = 0, graph = 0;
  auto ok = [](int status) {
    if (status)
      std::fprintf(stderr, "%s\n", pangu_acl_last_error());
    return status == 0;
  };
  char *end = nullptr;
  errno = 0;
  long device = std::strtol(argv[1], &end, 10);
  if (errno || !end || *end || device < 0 || device > INT_MAX)
    return 2;
  if (!ok(pangu_acl_open(static_cast<int>(device), &s)))
    return 1;
  bool passed = ok(pangu_acl_allocate(s, 64, &source)) &&
                ok(pangu_acl_allocate(s, 64, &destination)) &&
                ok(pangu_acl_capture_copy(s, source, destination, 64, &graph));
  for (int iteration = 0; passed && iteration < 3; ++iteration) {
    std::array<unsigned char, 64> input{}, output{};
    input.fill(static_cast<unsigned char>(iteration + 1));
    passed =
        ok(pangu_acl_write(s, source, 0, input.data(), input.size())) &&
        ok(pangu_acl_replay(s, graph)) &&
        ok(pangu_acl_read(s, destination, 0, output.data(), output.size())) &&
        input == output;
  }
  if (passed) {
    std::array<unsigned char, 65> too_large{};
    passed = pangu_acl_write(s, source, 0, too_large.data(),
                             too_large.size()) != 0 &&
             pangu_acl_replay(s, UINT64_MAX) != 0;
    int wrong_thread_status = 0;
    std::thread other([&] {
      uint64_t ignored = 0;
      wrong_thread_status = pangu_acl_allocate(s, 64, &ignored);
    });
    other.join();
    passed = passed && wrong_thread_status != 0;
    // Rejected calls must not invalidate a healthy graph.
    passed = ok(pangu_acl_replay(s, graph)) && passed;
  }
  passed = ok(pangu_acl_close(s)) && passed;
  std::printf("{\"graph_copy_mutation\":\"%s\"}\n",
              passed ? "passed" : "failed");
  return passed ? 0 : 1;
}
