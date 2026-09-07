#include "pangu_acl.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>
void checked(int code) {
  if (code)
    throw std::runtime_error(pangu_acl_last_error());
}
uint16_t bf(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  return uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
float fp(uint16_t value) {
  uint32_t bits = uint32_t(value) << 16;
  float out;
  std::memcpy(&out, &bits, 4);
  return out;
}
int main() {
  pangu_acl_session *bootstrap = nullptr;
  checked(pangu_acl_open(0, &bootstrap));
  std::vector<unsigned char> root(pangu_acl_tp_root_size());
  checked(pangu_acl_tp_root(root.data(), root.size()));
  int failures[2] = {0, 0};
  std::vector<std::thread> threads;
  for (int rank = 0; rank < 2; ++rank)
    threads.emplace_back([&, rank] {
      pangu_acl_session *s = nullptr;
      try {
        checked(pangu_acl_open(rank, &s));
        checked(pangu_acl_tp_init(s, 2, rank, root.data(), root.size()));
        for (bool sharded : {false, true}) {
          constexpr int m = 4, n = 64, k = 64;
          uint64_t x, w, y, op, graph;
          checked(pangu_acl_allocate(s, m * k * 2, &x));
          checked(pangu_acl_allocate(s, (sharded ? n / 2 : n) * k * 2, &w));
          checked(pangu_acl_allocate(s, m * n * 2, &y));
          std::vector<uint16_t> a(m * k), b(n * k), eager(m * n),
              captured(m * n), expected(m * n);
          for (int row = 0; row < n; ++row)
            for (int col = 0; col < k; ++col)
              b[row * k + col] =
                  bf(float(row % 13 - 6) / 8 + float(col % 5 - 2) / 16);
          auto weight = sharded ? b.data() + rank * (n / 2) * k : b.data();
          checked(
              pangu_acl_write(s, w, 0, weight, (sharded ? n / 2 : n) * k * 2));
          checked(pangu_acl_linear_prepare(s, x, w, y, m, n, k, &op));
          for (int iteration = 0; iteration < 3; ++iteration) {
            for (int row = 0; row < m; ++row)
              for (int col = 0; col < k; ++col)
                a[row * k + col] = bf(float(row + 1 + iteration) / 8 +
                                      float(col % 7 - 3) / 16);
            for (int row = 0; row < m; ++row)
              for (int col = 0; col < n; ++col) {
                float sum = 0;
                for (int inner = 0; inner < k; ++inner)
                  sum += fp(a[row * k + inner]) * fp(b[col * k + inner]);
                expected[row * n + col] = bf(sum);
              }
            checked(pangu_acl_write(s, x, 0, a.data(), a.size() * 2));
            checked(pangu_acl_operation_execute(s, op));
            checked(pangu_acl_read(s, y, 0, eager.data(), eager.size() * 2));
            if (eager != expected)
              throw std::runtime_error("TP result differs from CPU reference");
            if (iteration == 0)
              checked(pangu_acl_operation_capture(s, op, &graph));
            checked(pangu_acl_replay(s, graph));
            checked(
                pangu_acl_read(s, y, 0, captured.data(), captured.size() * 2));
            if (captured != eager)
              throw std::runtime_error("TP graph differs from eager");
          }
          std::printf("rank=%d sharded_weights=%d CPU/eager/graph exact; three "
                      "changing inputs passed\n",
                      rank, sharded);
        }
      } catch (const std::exception &e) {
        failures[rank] = 1;
        std::fprintf(stderr, "rank=%d %s\n", rank, e.what());
      }
      if (s)
        checked(pangu_acl_close(s));
    });
  for (auto &thread : threads)
    thread.join();
  checked(pangu_acl_close(bootstrap));
  return failures[0] || failures[1];
}
