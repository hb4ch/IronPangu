# Owned vendor API placeholders — 2026-10-18

The native serving backend now calls locally compiled host API adapters for
window attention and normalized cache writes. The installed vendor L0 operators
and device kernels remain in use. These adapters fix the two surviving optional
descriptor paths identified in the pool-ownership trace.

The omitted rotary cache's one-dimensional empty tensor is now allocated by
its `aclOpExecutor`. Attention's optional empty outputs use local RAII owners,
retained through workspace preparation and released on success or error.
CANN `ViewCopy` rejects a workspace-owned output descriptor even when empty,
so attention deliberately uses external descriptors. No new output flags,
device copies, synchronization or host activation transfers are introduced.

Sources and license are under `native/vendor/omni_opapi`; upstream revision is
`d75e893709cc065c3519f21a8a5d9431e7b69d50`. Public adapter symbols have distinct
`ifOwned` names, and the CANN-facing static target uses the installed ABI 0.
No preload library or ELF interposition is required for serving.

## Full native qualification

On `7.156.99.58`, container `inferfabric-92b`, the fresh build is
`native-build-owned-opapi`. A fresh R3 bundle was compiled from
`family64k.inferfabric`, with context 65536, shared pages and maximum uncaptured
prefill 4096. The `device-zero-apc8k-fixture` alternates three independent 8K
references with short requests, using permuted pages and mixed execution.

All three stages (discovery, fixed workspace, frozen arena) passed 39 exact
reference output tokens across six requests. Every rank exercised sequence
decode graph replay. The frozen checks also require no execution-time device
allocation or internal completion boundary, and no forbidden activation transfer.

| Per-rank resource | Result, all eight ranks |
| --- | ---: |
| Frozen arena bytes | 40,492,359,704 |
| Shared workspace bytes | 159,221,248 |
| Activation host transfers | 0 |

These sizes match the previously qualified device-zero backend. The new backend
has its own fingerprint and freshly bound arena; it does not reuse the old
library's frozen arena. The 17 completed batches and real packed chunk geometry
are in `batch-trace-summary.json`; the output trace, full audits, stage summaries
and SHA-256 hashes are included. Pure decode batch timings are not HTTP TTFT or
a production throughput guarantee.

The small window-attention control also passed all six batch configurations
(including 4096 rows), checking 25,288,704 output values. An initial candidate
using workspace-owned attention outputs failed at `ViewCopy` before inference;
the external RAII output descriptors are the qualified implementation.

## HTTP load result: resource gate still fails

`owned-opapi-load-http` used the unchanged vLLM Rust 0.25.1 frontend, two active
residents, a 4096-token budget and this backend's new frozen bundle. The runner
completed all 24 four-client waves plus cancellation/recovery (120 exact EOS
reference outputs). No allocation tracer or allocator tuning was enabled.
Normal SIGTERM drain reports 171 admissions, 52 mixed ticks, 237,112 consumed
input tokens, zero remaining shared-page bytes and closed native workers.
All eight frozen-arena/workspace/transfer invariants still pass.

RSS grew from 7,367,770,112 to 7,720,783,872 bytes: **353,013,760 bytes
(336.7 MiB)**, exceeding the unchanged 128 MiB gate. The runner exited 1.
Threads (631) and FDs (76) stayed fixed. The final nine waves added 905,216
bytes; this short apparent plateau does not prove a long-term bound or justify
changing the gate. The full curve and failed gate are retained. The earlier
unmodified 24-wave run grew 790,700,032 bytes; different scheduling histories
prevent attributing the entire difference to this patch.

For the 24 mixed-load 8K retrieval requests, per-case median HTTP TTFT ranged
from 5.35 to 6.07 seconds (maximum 7.21 seconds). These include queueing behind
concurrent requests. They are not serial 64K measurements or pure decode TPOT.
`http-result.json` contains per-case timing and invariant checks; the raw run
directory contains output evidence, histories, audits and resource records.

The additional cache-writer control passed 20 launches and 9,775,872 checked
values across BF16/packed cache modes. Both small probes now finalize their own
resources without resetting the device.

Next distinguish bounded warmup/history growth from remaining retained metadata
with a longer, varied serving workload before declaring host stability. The
full serving objective, including current-library 64K qualification, remains open.

Reproduction uses `scripts/qualify-sequence-serving.py --batch-trace` with the
new `program_engine_probe`, `owned-opapi-source`, the fixture's requests/page
bindings/reference, and output prefix `owned-opapi-raii8k`. Source the usual
CANN and both vendor environments; HCCL mode is HOST, base port 40400.
