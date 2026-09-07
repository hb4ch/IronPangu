#ifndef PANGU_ACL_H
#define PANGU_ACL_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* One session per dedicated device thread. No exceptions cross this ABI.
 * All calls, including close, must occur on the creating thread.
 * Graph resources outlive every replay; close fences before destruction.
 */
typedef struct pangu_acl_session pangu_acl_session;
uint32_t pangu_acl_abi_version(void);
const char *pangu_acl_last_error(void);
int32_t pangu_acl_open(int32_t device, pangu_acl_session **out);
int32_t pangu_acl_allocate(pangu_acl_session *, uint64_t bytes,
                           uint64_t *handle);
int32_t pangu_acl_write(pangu_acl_session *, uint64_t handle, uint64_t offset,
                        const void *, uint64_t bytes);
int32_t pangu_acl_read(pangu_acl_session *, uint64_t handle, uint64_t offset,
                       void *, uint64_t bytes);
/* Weight-independent graph probe: capture D2D copy at stable addresses. */
int32_t pangu_acl_capture_copy(pangu_acl_session *, uint64_t source,
                               uint64_t destination, uint64_t bytes,
                               uint64_t *graph);
int32_t pangu_acl_replay(pangu_acl_session *, uint64_t graph);
/* BF16 X[M,K] @ W[N,K]^T -> Y[M,N], FP32 accumulation.
 * Prepare owns descriptors, a repeatable ACLNN executor and stable workspace.
 * Execute and capture do not re-plan; session owns resources until close. */
int32_t pangu_acl_linear_prepare(pangu_acl_session *, uint64_t x,
                                 uint64_t weight, uint64_t y, int64_t m,
                                 int64_t n, int64_t k, uint64_t *operation);
int32_t pangu_acl_linear_execute(pangu_acl_session *, uint64_t operation);
int32_t pangu_acl_linear_capture(pangu_acl_session *, uint64_t operation,
                                 uint64_t *graph);
/* Weighted RMS: BF16 X[M,K], FP32 gamma[K], BF16 Y[M,K].
 * Qwen zero-centered gamma must be precomputed as 1.0f +
 * float(checkpoint_weight). FP32 input cast, RMS with epsilon 1e-6, and final
 * BF16 cast are one operation. */
int32_t pangu_acl_rms_prepare(pangu_acl_session *, uint64_t x, uint64_t gamma,
                              uint64_t y, int64_t m, int64_t k,
                              uint64_t *operation);
int32_t pangu_acl_operation_execute(pangu_acl_session *, uint64_t operation);
int32_t pangu_acl_operation_capture(pangu_acl_session *, uint64_t operation,
                                    uint64_t *graph);
/* One recurrent token for each flattened request/head. Q,K,V,beta BF16;
 * g and persistent state FP32. Canonical state is [BH,Dk,Dv], no clamp.
 * Capture warmup mutates state: restore initial state before qualification
 * replay. */
int32_t pangu_acl_delta_prepare(pangu_acl_session *, uint64_t q, uint64_t k,
                                uint64_t v, uint64_t g, uint64_t beta,
                                uint64_t state, uint64_t out, int64_t bh,
                                int64_t dk, int64_t dv, uint64_t *operation);
/* Width-4 depthwise convolution + SiLU. All tensors BF16: X[B,C], W[C,4],
 * history[B,C,3] oldest first, Y[B,C]. Keeps the last three raw inputs. */
int32_t pangu_acl_conv_prepare(pangu_acl_session *, uint64_t x, uint64_t weight,
                               uint64_t history, uint64_t out, int64_t batch,
                               int64_t channels, uint64_t *operation);
int32_t pangu_acl_sequence_prepare(pangu_acl_session *, const uint64_t *,
                                   uint64_t count, uint64_t *operation);
/* Pointwise kinds: 0 add, 1 multiply, 2 SiLU. BF16 inputs/output. */
int32_t pangu_acl_pointwise_prepare(pangu_acl_session *, uint64_t a, uint64_t b,
                                    uint64_t out, int64_t count, int32_t kind,
                                    uint64_t *operation);
int32_t pangu_acl_delta_transforms_prepare(pangu_acl_session *, uint64_t conv,
                                           uint64_t a, uint64_t b,
                                           uint64_t a_log, uint64_t dt_bias,
                                           uint64_t q, uint64_t k, uint64_t v,
                                           uint64_t g, uint64_t beta,
                                           int64_t batch, uint64_t *operation);
int32_t pangu_acl_gated_rms_prepare(pangu_acl_session *, uint64_t x, uint64_t z,
                                    uint64_t weight, uint64_t out, int64_t bh,
                                    int64_t width, uint64_t *operation);
int32_t pangu_acl_attention_qk_prepare(pangu_acl_session *, uint64_t packed_q,
                                       uint64_t k, uint64_t qgamma,
                                       uint64_t kgamma, uint64_t positions,
                                       uint64_t qout, uint64_t kout,
                                       uint64_t gate, int64_t batch,
                                       uint64_t *operation);
/* Physically paged cache with flattened physical-token indices. Append/table
 * I64, additive mask FP32. All indices must be in [0,slots), each request must
 * expose at least one valid key, append rows must be unique. Mask excludes
 * unused rows. */
int32_t pangu_acl_paged_attention_prepare(pangu_acl_session *, uint64_t q,
                                          uint64_t k, uint64_t v, uint64_t gate,
                                          uint64_t keys, uint64_t values,
                                          uint64_t append, uint64_t table,
                                          uint64_t mask, uint64_t out,
                                          int64_t batch, int64_t slots,
                                          int64_t context, uint64_t *operation);
/* Internal ABI for compiler-emitted single-request block bindings. */
int32_t pangu_acl_embedding_prepare(pangu_acl_session *, uint64_t weight,
                                    uint64_t ids, uint64_t out, int64_t vocab,
                                    int64_t width, uint64_t *operation);
int32_t pangu_acl_model_layer_prepare(
    pangu_acl_session *, int32_t kind, const uint64_t *weights, uint64_t count,
    uint64_t input, uint64_t output, uint64_t positions, uint64_t append,
    uint64_t table, uint64_t mask, int64_t slots, int64_t context,
    uint64_t *state_a, uint64_t *state_b, uint64_t *operation);
/* On failure the session remains owned by the caller; do not free it. */
int32_t pangu_acl_close(pangu_acl_session *);
#ifdef __cplusplus
}
#endif
#endif
