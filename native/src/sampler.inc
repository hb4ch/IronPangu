// Prepared NPU sampler. Parameter/state buffers remain mutable under graph
// replay.
#include <aclnnop/aclnn_argmax.h>
#include <aclnnop/aclnn_cumsum.h>
#include <aclnnop/aclnn_gather.h>
#include <aclnnop/aclnn_ge_tensor.h>
#include <aclnnop/aclnn_gt_tensor.h>
#include <aclnnop/aclnn_lt_tensor.h>
#include <aclnnop/aclnn_s_where.h>
#include <aclnnop/aclnn_sort.h>
#include <aclnnop/aclnn_uniform.h>
extern "C" int32_t pangu_acl_sampler_prepare(
    pangu_acl_session *s, uint64_t logits, uint64_t params, uint64_t counts,
    uint64_t seen, uint64_t mask, uint64_t rng, uint64_t random,
    uint64_t output, uint64_t probabilities, int64_t vocab,
    int64_t random_count, int32_t greedy, uint64_t *operation) {
  return boundary([&] {
    if (!s || !operation || vocab < 2 || vocab > 1048576 || random_count < 1 ||
        (greedy != 0 && greedy != 1))
      throw std::runtime_error("invalid sampler preparation");
    Composite b(s);
    auto tensor = [&](uint64_t id, uint64_t bytes, aclDataType dtype) {
      return b.tensor(s->region(id, 0, bytes), {1, vocab}, dtype);
    };
    auto scratch = [&](aclDataType dtype = ACL_FLOAT) {
      return b.tensor(b.temp(vocab * (dtype == ACL_INT64  ? 8
                                      : dtype == ACL_BOOL ? 1
                                                          : 4)),
                      {1, vocab}, dtype);
    };
    auto constant = [&](float value) {
      auto *ptr = b.temp(4);
      check(aclrtMemcpy(ptr, 4, &value, 4, ACL_MEMCPY_HOST_TO_DEVICE),
            "sampler constant");
      return b.tensor(ptr, {1, 1}, ACL_FLOAT);
    };
    auto param = [&](int64_t i) {
      return b.tensor(s->region(params, 0, 32), {1, 1}, ACL_FLOAT, {1, 1}, i,
                      {8});
    };
    auto where = [&](aclTensor *cond, aclTensor *yes, aclTensor *no) {
      auto *out = scratch();
      b.step(
          aclnnSWhere,
          [&](auto *n, auto *e) {
            return aclnnSWhereGetWorkspaceSize(cond, yes, no, out, n, e);
          },
          "sampler where");
      return out;
    };
    auto ge = [&](aclTensor *a, aclTensor *c) {
      auto *out = scratch(ACL_BOOL);
      b.step(
          aclnnGeTensor,
          [&](auto *n, auto *e) {
            return aclnnGeTensorGetWorkspaceSize(a, c, out, n, e);
          },
          "sampler ge");
      return out;
    };
    auto gt = [&](aclTensor *a, aclTensor *c) {
      auto *out = scratch(ACL_BOOL);
      b.step(
          aclnnGtTensor,
          [&](auto *n, auto *e) {
            return aclnnGtTensorGetWorkspaceSize(a, c, out, n, e);
          },
          "sampler gt");
      return out;
    };
    auto mul = [&](aclTensor *a, aclTensor *c) {
      auto *out = scratch();
      b.mul(a, c, out);
      return out;
    };
    auto add = [&](aclTensor *a, aclTensor *c, bool sub = false) {
      auto *out = scratch();
      b.add(a, c, out, sub);
      return out;
    };
    auto *zero = constant(0),
         *neg_inf = constant(-std::numeric_limits<float>::infinity());
    auto *x =
        b.cast(tensor(logits, vocab * 2, ACL_BF16), {1, vocab}, ACL_FLOAT);
    auto *history = tensor(seen, vocab * 4, ACL_FLOAT),
         *generated = tensor(counts, vocab * 4, ACL_FLOAT);
    // Repetition covers prompt + output; presence/frequency cover generated
    // tokens only.
    auto *repeated = where(ge(x, zero), mul(x, param(5)), mul(x, param(4)));
    auto *adjusted = where(gt(history, zero), repeated, x);
    auto *present = b.cast(gt(generated, zero), {1, vocab}, ACL_FLOAT);
    auto *penalty = add(mul(generated, param(6)), mul(present, param(7)));
    auto *processed =
        add(add(adjusted, penalty, true), tensor(mask, vocab * 4, ACL_FLOAT));
    auto *out = b.tensor(s->region(output, 0, 8), {1, 1}, ACL_INT64);
    if (greedy) {
      b.step(
          aclnnArgMax,
          [&](auto *n, auto *e) {
            return aclnnArgMaxGetWorkspaceSize(processed, 1, true, out, n, e);
          },
          "sampler greedy argmax");
    } else {
      auto *scaled = mul(processed, param(0));
      void *sorted_ptr = b.temp(vocab * 4);
      auto *sorted = b.tensor(sorted_ptr, {1, vocab}, ACL_FLOAT);
      auto *indices = scratch(ACL_INT64);
      b.step(
          aclnnSort,
          [&](auto *n, auto *e) {
            return aclnnSortGetWorkspaceSize(scaled, true, 1, true, sorted,
                                             indices, n, e);
          },
          "sampler stable sort");
      auto *k_index =
          b.tensor(s->region(rng, 0, 24), {1, 1}, ACL_INT64, {1, 1}, 2, {3});
      auto *threshold = b.tensor(b.temp(4), {1, 1}, ACL_FLOAT);
      b.step(
          aclnnGather,
          [&](auto *n, auto *e) {
            return aclnnGatherGetWorkspaceSize(sorted, 1, k_index, threshold, n,
                                               e);
          },
          "sampler top-k threshold");
      // Keep all ties at the kth logit, matching vLLM's threshold semantics.
      auto *topk = where(ge(sorted, threshold), sorted, neg_inf);
      auto *maximum =
          b.tensor(sorted_ptr, {1, 1}, ACL_FLOAT, {vocab, 1}, 0, {1, vocab});
      auto *min_threshold = b.tensor(b.temp(4), {1, 1}, ACL_FLOAT);
      b.add(maximum, param(3), min_threshold);
      auto *filtered = where(ge(sorted, min_threshold), topk, neg_inf);
      auto *probs = scratch();
      b.step(
          aclnnSoftmax,
          [&](auto *n, auto *e) {
            return aclnnSoftmaxGetWorkspaceSize(filtered, 1, probs, n, e);
          },
          "sampler softmax");
      auto *cumulative = scratch();
      b.step(
          aclnnCumsumV2,
          [&](auto *n, auto *e) {
            return aclnnCumsumV2GetWorkspaceSize(probs, 1, true, false,
                                                 cumulative, n, e);
          },
          "sampler exclusive cdf");
      auto *keep = scratch(ACL_BOOL);
      b.step(
          aclnnLtTensor,
          [&](auto *n, auto *e) {
            return aclnnLtTensorGetWorkspaceSize(cumulative, param(2), keep, n,
                                                 e);
          },
          "sampler nucleus mask");
      auto *distribution = tensor(probabilities, vocab * 4, ACL_FLOAT);
      b.step(
          aclnnSWhere,
          [&](auto *n, auto *e) {
            return aclnnSWhereGetWorkspaceSize(keep, probs, zero, distribution,
                                               n, e);
          },
          "sampler nucleus distribution");
      auto *selected = b.tensor(b.temp(8), {1, 1}, ACL_INT64);
      auto *uniform = b.tensor(b.temp(4), {1, 1}, ACL_FLOAT);
      auto *random_values = b.tensor(s->region(random, 0, random_count * 4),
                                     {1, random_count}, ACL_FLOAT);
      auto *random_index = b.tensor(s->region(rng, 8, 8), {1, 1}, ACL_INT64);
      b.step(
          aclnnGather,
          [&](auto *n, auto *e) {
            return aclnnGatherGetWorkspaceSize(random_values, 1, random_index,
                                               uniform, n, e);
          },
          "sampler random draw");
      void *cdf_ptr = b.temp(vocab * 4);
      auto *cdf = b.tensor(cdf_ptr, {1, vocab}, ACL_FLOAT);
      b.step(
          aclnnCumsum,
          [&](auto *n, auto *e) {
            return aclnnCumsumGetWorkspaceSize(distribution, 1, ACL_FLOAT, cdf,
                                               n, e);
          },
          "sampler categorical cdf");
      auto *total = b.tensor(cdf_ptr, {1, 1}, ACL_FLOAT, {vocab, 1}, vocab - 1,
                             {1, vocab});
      auto *draw = b.tensor(b.temp(4), {1, 1}, ACL_FLOAT);
      b.mul(uniform, total, draw);
      auto *crossed = b.cast(gt(cdf, draw), {1, vocab}, ACL_FLOAT);
      b.step(
          aclnnArgMax,
          [&](auto *n, auto *e) {
            return aclnnArgMaxGetWorkspaceSize(crossed, 1, true, selected, n,
                                               e);
          },
          "sampler inverse cdf");
      b.step(
          aclnnGather,
          [&](auto *n, auto *e) {
            return aclnnGatherGetWorkspaceSize(indices, 1, selected, out, n, e);
          },
          "sampler selected token");
    }
    *operation = b.finish();
  });
}

// CANN host-seeded RNG runs once per request, outside capture. The draw table
// is consumed by a mutable device index; no per-token random-op preparation.
extern "C" int32_t pangu_acl_random_fill(pangu_acl_session *s, uint64_t buffer,
                                         int64_t count, uint64_t seed) {
  return boundary([&] {
    if (!s || count < 1 || count > 1048576)
      throw std::runtime_error("invalid RNG count");
    Composite b(s);
    auto *t = b.tensor(s->region(buffer, 0, count * 4), {count}, ACL_FLOAT);
    uint64_t bytes = 0;
    aclOpExecutor *executor = nullptr;
    check(aclnnInplaceUniformGetWorkspaceSize(t, 0.0, 1.0, seed, 0, &bytes,
                                              &executor),
          "prepare request RNG");
    void *workspace = s->dedicated_workspaces
                          ? (bytes ? b.temp(bytes) : nullptr)
                          : s->acquire_workspace(bytes);
    check(aclnnInplaceUniform(workspace, bytes, executor, s->stream),
          "generate request RNG");
    check(aclrtSynchronizeStream(s->stream), "request RNG completion");
  });
}
