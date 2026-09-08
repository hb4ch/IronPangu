#ifndef INFERFABRIC_ACL_H
#define INFERFABRIC_ACL_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* One session per dedicated device thread. No exceptions cross this ABI.
 * All calls, including close, must occur on the creating thread.
 * Graph resources outlive every replay; close fences before destruction.
 */
typedef struct inferfabric_acl_session inferfabric_acl_session;
uint32_t inferfabric_acl_abi_version(void);
const char *inferfabric_acl_last_error(void);
int32_t inferfabric_acl_open(int32_t device, inferfabric_acl_session **out);
/* Observed HBM watermark includes runtime allocations visible at samples;
 * it is not an exact hardware trace of transient allocation peaks. */
typedef struct inferfabric_acl_memory_stats {
  uint64_t total_bytes, free_bytes, observed_peak_bytes, buffer_bytes;
  uint64_t temporary_bytes, workspace_bytes, graphs, budget_bytes;
} inferfabric_acl_memory_stats;
int32_t inferfabric_acl_memory_snapshot(inferfabric_acl_session *,
                                  inferfabric_acl_memory_stats *);
/* Additive ABI: physical pool ownership and logical aliased demand. */
typedef struct inferfabric_acl_allocation_stats {
  uint64_t scratch_pool_bytes, layer_pool_bytes, snapshot_pool_bytes;
  uint64_t owned_temporary_bytes, logical_scratch_bytes,
      logical_workspace_bytes;
  uint64_t workspace_blocks, owned_buffers, buffer_views;
  uint64_t dedicated_activations, dedicated_workspaces;
} inferfabric_acl_allocation_stats;
int32_t inferfabric_acl_allocation_snapshot(inferfabric_acl_session *,
                                      inferfabric_acl_allocation_stats *);
int32_t inferfabric_acl_set_memory_budget(inferfabric_acl_session *, uint64_t bytes,
                                    uint64_t reserve);
int32_t inferfabric_acl_allocate(inferfabric_acl_session *, uint64_t bytes,
                           uint64_t *handle);
/* Non-owning aligned region; parent storage remains session-owned. */
int32_t inferfabric_acl_buffer_view(inferfabric_acl_session *, uint64_t parent,
                              uint64_t offset, uint64_t bytes,
                              uint64_t *handle);
int32_t inferfabric_acl_write(inferfabric_acl_session *, uint64_t handle, uint64_t offset,
                        const void *, uint64_t bytes);
int32_t inferfabric_acl_read(inferfabric_acl_session *, uint64_t handle, uint64_t offset,
                       void *, uint64_t bytes);
/* Weight-independent graph probe: capture D2D copy at stable addresses. */
int32_t inferfabric_acl_capture_copy(inferfabric_acl_session *, uint64_t source,
                               uint64_t destination, uint64_t bytes,
                               uint64_t *graph);
int32_t inferfabric_acl_replay(inferfabric_acl_session *, uint64_t graph);
/* BF16 X[M,K] @ W[N,K]^T -> Y[M,N], FP32 accumulation.
 * Prepare owns descriptors, a repeatable ACLNN executor and stable workspace.
 * Execute and capture do not re-plan; session owns resources until close. */
int32_t inferfabric_acl_linear_prepare(inferfabric_acl_session *, uint64_t x,
                                 uint64_t weight, uint64_t y, int64_t m,
                                 int64_t n, int64_t k, uint64_t *operation);
int32_t inferfabric_acl_linear_execute(inferfabric_acl_session *, uint64_t operation);
int32_t inferfabric_acl_linear_capture(inferfabric_acl_session *, uint64_t operation,
                                 uint64_t *graph);
/* Weighted RMS: BF16 X[M,K], FP32 gamma[K], BF16 Y[M,K].
 * Qwen zero-centered gamma must be precomputed as 1.0f +
 * float(checkpoint_weight). FP32 input cast, RMS with epsilon 1e-6, and final
 * BF16 cast are one operation. */
int32_t inferfabric_acl_rms_prepare(inferfabric_acl_session *, uint64_t x, uint64_t gamma,
                              uint64_t y, int64_t m, int64_t k,
                              uint64_t *operation);
int32_t inferfabric_acl_operation_execute(inferfabric_acl_session *, uint64_t operation);
int32_t inferfabric_acl_operation_capture(inferfabric_acl_session *, uint64_t operation,
                                    uint64_t *graph);
/* One recurrent token for each flattened request/head. Q,K,V,beta BF16;
 * g and persistent state FP32. Canonical state is [BH,Dk,Dv], no clamp.
 * Capture warmup mutates state: restore initial state before qualification
 * replay. */
int32_t inferfabric_acl_delta_prepare(inferfabric_acl_session *, uint64_t q, uint64_t k,
                                uint64_t v, uint64_t g, uint64_t beta,
                                uint64_t state, uint64_t out, int64_t bh,
                                int64_t dk, int64_t dv, uint64_t *operation);
/* Width-4 depthwise convolution + SiLU. All tensors BF16: X[B,C], W[C,4],
 * history[B,C,3] oldest first, Y[B,C]. Keeps the last three raw inputs. */
int32_t inferfabric_acl_conv_prepare(inferfabric_acl_session *, uint64_t x, uint64_t weight,
                               uint64_t history, uint64_t out, int64_t batch,
                               int64_t channels, uint64_t *operation);
int32_t inferfabric_acl_sequence_prepare(inferfabric_acl_session *, const uint64_t *,
                                   uint64_t count, uint64_t *operation);
/* Pointwise kinds: 0 add, 1 multiply, 2 SiLU. BF16 inputs/output. */
int32_t inferfabric_acl_pointwise_prepare(inferfabric_acl_session *, uint64_t a, uint64_t b,
                                    uint64_t out, int64_t count, int32_t kind,
                                    uint64_t *operation);
int32_t inferfabric_acl_delta_transforms_prepare(inferfabric_acl_session *, uint64_t conv,
                                           uint64_t a, uint64_t b,
                                           uint64_t a_log, uint64_t dt_bias,
                                           uint64_t q, uint64_t k, uint64_t v,
                                           uint64_t g, uint64_t beta,
                                           int64_t batch, uint64_t *operation);
int32_t inferfabric_acl_gated_rms_prepare(inferfabric_acl_session *, uint64_t x, uint64_t z,
                                    uint64_t weight, uint64_t out, int64_t bh,
                                    int64_t width, uint64_t *operation);
int32_t inferfabric_acl_attention_qk_prepare(inferfabric_acl_session *, uint64_t packed_q,
                                       uint64_t k, uint64_t qgamma,
                                       uint64_t kgamma, uint64_t positions,
                                       uint64_t qout, uint64_t kout,
                                       uint64_t gate, int64_t batch,
                                       uint64_t *operation);
/* Physically paged cache with flattened physical-token indices. Append/table
 * I64, additive mask FP32. All indices must be in [0,slots), each request must
 * expose at least one valid key, append rows must be unique. Mask excludes
 * unused rows. */
int32_t inferfabric_acl_paged_attention_prepare(inferfabric_acl_session *, uint64_t q,
                                          uint64_t k, uint64_t v, uint64_t gate,
                                          uint64_t keys, uint64_t values,
                                          uint64_t append, uint64_t table,
                                          uint64_t mask, uint64_t out,
                                          int64_t batch, int64_t slots,
                                          int64_t context, uint64_t *operation);
/* Internal ABI for compiler-emitted single-request block bindings. */
int32_t inferfabric_acl_embedding_prepare(inferfabric_acl_session *, uint64_t weight,
                                    uint64_t ids, uint64_t out, int64_t vocab,
                                    int64_t width, uint64_t *operation);
int32_t inferfabric_acl_model_layer_prepare(
    inferfabric_acl_session *, int32_t kind, const uint64_t *weights, uint64_t count,
    uint64_t input, uint64_t output, uint64_t positions, uint64_t append,
    uint64_t table, uint64_t mask, int64_t slots, int64_t context,
    uint64_t *state_a, uint64_t *state_b, uint64_t *operation);
/* On failure the session remains owned by the caller; do not free it. */
/* Fixed batch graph: unique physical cache ranges per slot, boolean active[B].
 */
int32_t inferfabric_acl_embedding_batch_prepare(inferfabric_acl_session *, uint64_t,
                                          uint64_t, uint64_t, int64_t, int64_t,
                                          int64_t, uint64_t *);
int32_t inferfabric_acl_model_layer_batch_prepare(inferfabric_acl_session *, int32_t,
                                            const uint64_t *, uint64_t,
                                            uint64_t, uint64_t, uint64_t,
                                            uint64_t, uint64_t, uint64_t,
                                            int64_t, int64_t, int64_t, uint64_t,
                                            uint64_t *, uint64_t *, uint64_t *);
int32_t inferfabric_acl_copy_region(inferfabric_acl_session *, uint64_t, uint64_t, uint64_t,
                              uint64_t, uint64_t);
int32_t inferfabric_acl_device_count(uint32_t *);
uint64_t inferfabric_acl_tp_root_size(void);
int32_t inferfabric_acl_tp_root(void *, uint64_t);
int32_t inferfabric_acl_tp_init(inferfabric_acl_session *, uint32_t, uint32_t, const void *,
                          uint64_t);
int32_t inferfabric_acl_close(inferfabric_acl_session *);
/* Params FP32[8]: inverse temperature, reserved, top_p, log(min_p),
 * repetition penalty, inverse repetition penalty, frequency, presence.
 * Counts/seen/mask FP32[vocab], controls INT64[3]: reserved, draw index,
 * top_k-1. Random FP32[random_count] is generated on device once per request.
 * Output INT64[1]; probabilities FP32[vocab] in sorted-logit order.
 * Caller validates finite ranges and controls. Greedy ignores stochastic
 * filters. */
int32_t inferfabric_acl_sampler_prepare(inferfabric_acl_session *, uint64_t logits,
                                  uint64_t params, uint64_t counts,
                                  uint64_t seen, uint64_t mask,
                                  uint64_t controls, uint64_t random,
                                  uint64_t output, uint64_t probabilities,
                                  int64_t vocab, int64_t random_count,
                                  int32_t greedy, uint64_t *operation);
int32_t inferfabric_acl_random_fill(inferfabric_acl_session *, uint64_t buffer,
                              int64_t count, uint64_t seed);
#ifdef __cplusplus
}
#endif
#endif
