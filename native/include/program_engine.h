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
int inferfabric_program_info(void *engine, uint32_t *context, uint32_t *vocabulary, uint32_t *eos);
int inferfabric_program_token(void *engine, uint32_t token, uint32_t position,
                              uint32_t sample, uint32_t *output);
int inferfabric_program_close(void *engine);
const char *inferfabric_program_error(void);
#ifdef __cplusplus
}
#endif
