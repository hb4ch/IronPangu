# Owned optional descriptors

These six API source/header files are derived from omni-ops commit
`d75e893709cc065c3519f21a8a5d9431e7b69d50` (Gitee `omniai/omni-ops`), under
`inference/ascendc/src/ops-transformer/{attention/ai_infra_fused_infer_attention_sink,posembedding/ai_infra_kv_rms_norm_rope_cache}/op_api`.
The upstream license is included. They contain host API preparation only;
the installed vendor package continues to provide L0 operators and NPU kernels.

Local changes allocate the absent cache descriptor through
`aclOpExecutor::AllocTensor`. Optional attention outputs must remain external
descriptors for CANN `ViewCopy`; local RAII owners retain these through workspace
preparation and destroy them on success or error. Empty shapes,
dtypes, attention output flags and kernel selection remain unchanged. The
max-workspace forwarding helper leaves placeholder creation to the callee.

Public API names have an `ifOwned` prefix so correctness does not depend on ELF
interposition or load order. The native executor calls these functions directly.
This source requires the installed four-output attention L0 ABI; an incompatible
vendor package must fail to link, rather than silently selecting another kernel.

Full native 92B output and frozen-arena qualification is recorded in
`docs/validation/2026-10-18-owned-opapi`. Do not deploy this revision with an
older library's frozen arena or reuse its correctness/performance evidence.
