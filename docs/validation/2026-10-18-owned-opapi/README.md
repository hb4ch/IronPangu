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

The long-running HTTP host-resource gate is a separate requirement. Its result
must be inspected before claiming the RSS growth issue resolved.

Reproduction uses `scripts/qualify-sequence-serving.py --batch-trace` with the
new `program_engine_probe`, `owned-opapi-source`, the fixture's requests/page
bindings/reference, and output prefix `owned-opapi-raii8k`. Source the usual
CANN and both vendor environments; HCCL mode is HOST, base port 40400.
