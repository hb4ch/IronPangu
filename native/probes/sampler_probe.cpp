#include "pangu_acl.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
void check(int status) {
  if (status)
    throw std::runtime_error(pangu_acl_last_error());
}
uint16_t bf(float f) {
  uint32_t n;
  std::memcpy(&n, &f, 4);
  return uint16_t((n + 0x7fff + ((n >> 16) & 1)) >> 16);
}
float fp(uint16_t n) {
  uint32_t bits = uint32_t(n) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}
struct Test {
  pangu_acl_session *s = nullptr;
  int64_t v;
  uint64_t logits, params, counts, seen, mask, rng, random, out, probs, op,
      greedy, graph;
  std::vector<uint16_t> x;
  std::vector<float> c, h, m;
  float p[8] = {1, 0, 1, -INFINITY, 1, 1, 0, 0};
  int64_t control[3] = {42, 0, 0};
  uint64_t alloc(size_t n) {
    uint64_t id;
    check(pangu_acl_allocate(s, n, &id));
    return id;
  }
  template <class T> void write(uint64_t id, const T *ptr, size_t n) {
    check(pangu_acl_write(s, id, 0, ptr, n * sizeof(T)));
  }
  Test(int device, int64_t vocab) : v(vocab), x(v), c(v), h(v), m(v) {
    check(pangu_acl_open(device, &s));
    logits = alloc(v * 2);
    params = alloc(32);
    counts = alloc(v * 4);
    seen = alloc(v * 4);
    mask = alloc(v * 4);
    rng = alloc(24);
    random = alloc(2048 * 4);
    check(pangu_acl_random_fill(s, random, 2048, 42));
    out = alloc(8);
    probs = alloc(v * 4);
    control[2] = v - 1;
    upload();
    check(pangu_acl_sampler_prepare(s, logits, params, counts, seen, mask, rng,
                                    random, out, probs, v, 2048, 0, &op));
    check(pangu_acl_sampler_prepare(s, logits, params, counts, seen, mask, rng,
                                    random, out, probs, v, 2048, 1, &greedy));
    check(pangu_acl_operation_execute(s, op));
    check(pangu_acl_operation_capture(s, op, &graph));
  }
  ~Test() {
    if (s)
      pangu_acl_close(s);
  }
  void upload() {
    write(logits, x.data(), x.size());
    write(params, p, 8);
    write(counts, c.data(), c.size());
    write(seen, h.data(), h.size());
    write(mask, m.data(), m.size());
    write(rng, control, 3);
  }
  int64_t token(bool replay = true) {
    check(replay ? pangu_acl_replay(s, graph)
                 : pangu_acl_operation_execute(s, op));
    int64_t id;
    check(pangu_acl_read(s, out, 0, &id, 8));
    if (id < 0 || id >= v)
      throw std::runtime_error("invalid sample");
    return id;
  }
  double validate() {
    upload();
    auto a = token(false), b = token();
    if (a != b || b != token())
      throw std::runtime_error("fixed seed eager/graph mismatch");
    std::vector<float> expected(v);
    for (int64_t i = 0; i < v; ++i) {
      float z = fp(x[i]);
      if (h[i] > 0)
        z = z >= 0 ? z * p[5] : z * p[4];
      expected[i] = (z - c[i] * p[6] - (c[i] > 0 ? p[7] : 0) + m[i]) * p[0];
    }
    auto best =
        std::max_element(expected.begin(), expected.end()) - expected.begin();
    check(pangu_acl_operation_execute(s, greedy));
    int64_t greedy_id = -1;
    check(pangu_acl_read(s, out, 0, &greedy_id, 8));
    if (greedy_id != best)
      throw std::runtime_error("NPU greedy/reference mismatch");
    std::stable_sort(expected.begin(), expected.end(), std::greater<float>());
    float threshold = expected[control[2]], min = expected[0] + p[3],
          max = expected[0];
    double sum = 0;
    for (auto &z : expected) {
      z = z >= threshold && z >= min ? std::exp(z - max) : 0;
      sum += z;
    }
    double accum = 0;
    for (auto &z : expected) {
      float prob = z / sum;
      z = accum < p[2] ? prob : 0;
      accum += prob;
    }
    std::vector<float> got(v);
    check(pangu_acl_read(s, probs, 0, got.data(), v * 4));
    double error = 0;
    for (int64_t i = 0; i < v; ++i) {
      if (!std::isfinite(got[i]))
        throw std::runtime_error("nonfinite distribution");
      error = std::max(error, std::abs(double(got[i] - expected[i])));
    }
    if (error > 1e-5)
      throw std::runtime_error("distribution mismatch: " +
                               std::to_string(error));
    return error;
  }
};
int main(int argc, char **argv) {
  try {
    int device = argc > 1 ? std::stoi(argv[1]) : 1;
    double maxerror = 0;
    int cases = 0;
    for (int vocab : {32, 248320}) {
      std::cerr << "vocab " << vocab << " preparing\n";
      Test t(device, vocab);
      for (int i = 0; i < vocab; ++i)
        t.x[i] = bf(i < 24 ? float(i % 9 - 4) * 0.5f : -80.f);
      for (int mode = 0; mode < 6; ++mode) {
        t.p[0] = mode == 1 ? 2.f : 1.f;
        t.p[2] = mode == 2 ? .71f : 1.f;
        t.p[3] = mode == 3 ? std::log(.3f) : -INFINITY;
        t.control[2] = mode == 4 ? 0 : mode == 5 ? 6 : vocab - 1;
        t.p[4] = mode == 5 ? 1.3f : 1;
        t.p[5] = 1 / t.p[4];
        t.p[6] = mode == 5 ? .4f : 0;
        t.p[7] = mode == 5 ? 2 : 0;
        t.h[8] = 1;
        t.h[0] = 1;
        t.c[17] = 3;
        t.m[16] = mode == 5 ? -INFINITY : 0;
        maxerror = std::max(maxerror, t.validate());
        ++cases;
      }
    }
    Test t(device, 32);
    for (int i = 0; i < 32; ++i)
      t.x[i] = bf(i < 4 ? 0 : -80);
    t.control[2] = 3;
    t.upload();
    std::vector<int> bins(4);
    int changes = 0;
    int64_t previous = -1;
    for (int i = 0; i < 2048; ++i) {
      t.control[1] = int64_t(i);
      t.write(t.rng, t.control, 3);
      auto id = t.token();
      if (id >= 4)
        throw std::runtime_error("sample outside support");
      ++bins[id];
      changes += id != previous;
      previous = id;
    }
    for (int n : bins)
      if (n < 390 || n > 635)
        throw std::runtime_error("uniform sampling frequency failure");
    std::vector<float> original(2048), regenerated(2048);
    check(pangu_acl_read(t.s, t.random, 0, original.data(), 8192));
    check(pangu_acl_random_fill(t.s, t.random, 2048, 42));
    check(pangu_acl_read(t.s, t.random, 0, regenerated.data(), 8192));
    if (original != regenerated)
      throw std::runtime_error("request seed reset mismatch");
    check(pangu_acl_random_fill(t.s, t.random, 2048, 43));
    check(pangu_acl_read(t.s, t.random, 0, regenerated.data(), 8192));
    if (original == regenerated)
      throw std::runtime_error("RNG seed ignored");
    std::cout << "{\"device\":" << device << ",\"distribution_cases\":" << cases
              << ",\"max_probability_error\":" << maxerror
              << ",\"graph_seed_replay\":\"passed\",\"draws\":2048,\"bins\":["
              << bins[0] << "," << bins[1] << "," << bins[2] << "," << bins[3]
              << "],\"changes\":" << changes << "}\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
