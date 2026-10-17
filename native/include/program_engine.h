#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// One persistent rank group. Calls are serialized by one owning Rust worker.
// Handles own all device resources; no activation pointer crosses this ABI.
uint32_t inferfabric_program_abi_version(void);
// preparation=1 is an offline qualification mode; never expose HTTP readiness.
int inferfabric_program_open(const char *bundle, const char *checkpoint, uint32_t preparation, void **out);
int inferfabric_program_reset(void *engine);
// Resident capacity is compiled into the executable. Selection only stages
// control metadata; cache/recurrent storage and graph addresses remain fixed.
// Reset one slot without disturbing other requests. Calls remain serialized.
int inferfabric_program_resident_capacity(void *engine, uint32_t *count);
int inferfabric_program_select_resident(void *engine, uint32_t slot);
int inferfabric_program_reset_resident(void *engine, uint32_t slot);
// Explicit complete-state D2D copy. Source must be exactly at boundary;
// destination must be reset with private pages reserved for every component.
// All ranks preflight before launch; position is published after all streams
// complete. This copies state only, not logits or immutable-cache ownership.
int inferfabric_program_copy_resident(void *engine, uint32_t source, uint32_t destination,
                                     uint32_t boundary);
// Bind a cache component's device block table after reset and before any token.
// Metadata only; storage addresses and ACLGraph captures remain unchanged.
int inferfabric_program_bind_pages(void *engine, uint32_t rank, const char *component,
                                  const int32_t *pages, uint32_t count);
int inferfabric_program_info(void *engine, uint32_t *context, uint32_t *vocabulary, uint32_t *eos);
int inferfabric_program_token(void *engine, uint32_t token, uint32_t position,
                              uint32_t sample, uint32_t *output);
// Maximum compiled chunk width, or 1 when no packed variant was compiled.
int inferfabric_program_prefill_capacity(void *engine, uint32_t *count);
// 0: exact captured shapes; 1: bounded uncaptured query lengths up to capacity.
int inferfabric_program_prefill_mode(void *engine, uint32_t *mode);
// Bit n is set iff exact prefill size n is prepared. Decode uses its own entry point.
int inferfabric_program_prefill_variants(void *engine, uint32_t *mask);
// Exact ascending widths, bounded by caller capacity. Unlike the legacy mask,
// this ABI represents widths >=32 without shifts or truncation.
int inferfabric_program_prefill_sizes(void *engine, uint32_t *sizes, uint32_t capacity, uint32_t *count);
int inferfabric_program_shared_pages(void *engine, uint32_t *enabled);
int inferfabric_program_bind_state(void *engine, uint32_t rank, uint32_t logical_node,
                                  const int32_t *pages, uint32_t count);
int inferfabric_program_prefill(void *engine, const uint32_t *tokens, uint32_t count,
                               uint32_t position, uint32_t sample, uint32_t *output);
// Packed sequence extension. Tokens concatenate in sequence order; one output
// per sample=1 member, also in sequence order. No activation crosses this ABI.
typedef struct IFProgramSequenceV1 {
  uint32_t resident, position, token_count, sample;
} IFProgramSequenceV1;
int inferfabric_program_batch(void *engine,const IFProgramSequenceV1 *sequences,uint32_t sequence_count,
                             const uint32_t *tokens,uint32_t token_count,
                             uint32_t *outputs,uint32_t output_capacity,uint32_t *output_count);
int inferfabric_program_close(void *engine);
const char *inferfabric_program_error(void);
#ifdef __cplusplus
}
#endif
