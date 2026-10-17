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
// Bind a cache component's device block table after reset and before any token.
// Metadata only; storage addresses and ACLGraph captures remain unchanged.
int inferfabric_program_bind_pages(void *engine, uint32_t rank, const char *component,
                                  const int32_t *pages, uint32_t count);
int inferfabric_program_info(void *engine, uint32_t *context, uint32_t *vocabulary, uint32_t *eos);
int inferfabric_program_token(void *engine, uint32_t token, uint32_t position,
                              uint32_t sample, uint32_t *output);
int inferfabric_program_close(void *engine);
const char *inferfabric_program_error(void);
#ifdef __cplusplus
}
#endif
