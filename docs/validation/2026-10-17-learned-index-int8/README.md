# Explicit INT8 learned-index decoder integration

Date: 2026-10-17. Status: **experimental integration; full C8 parity fails**.
The context limit remains 512. This does not qualify 64K model serving.

The 92B learned-index DSL now declares INT8 index keys. Physical lowering emits
per-128 query/key dynamic quantization, separate FP16 scale casts, FP16 head-weight
casts, two cache append calls and `aclnnAiInfraQuantLightningIndexer`. The viewer
and serving manifest expose the separate key/scale arrays and combined 130-byte
per-token page budget. See [the execution contract](../../learned-index-quantization.md).
No model-specific executor path or host quantization was added.

Validation on `7.156.99.58` in Docker `inferfabric-92b`, full 46-layer checkpoint
`/models/92B`, TP8/EP8, two residents, shared shuffled pages and four-token prefill:

- Local compiler token suite: 11 tests passed. INT8 lowering, FP16 operand edges,
  page bytes and viewer recipes are checked; BF16 compatibility and rejection of
  unsupported INT4 index keys are also checked.
- Native build passed with warnings treated as errors. Native plan validation
  checked all eight rank bindings and rejected 33 malformed contracts, including
  missing scale casts, missing head-weight casts and invalid quantization width.
- Full standard fixture outputs: **99/99 tokens passed** in discovery, planned
  workspace preparation and frozen-arena execution.
- Boundary runs generated all **336 tokens** across three repetitions of 14
  prompts (lengths 1 through 504). Serial and packed/interleaved outputs match
  each other exactly. This checks layout/state consistency, not reference parity.
- Independent OmniInfer C8 reference: **9/14 sequences match**, versus 8/14 for
  the previous BF16-index implementation. Lengths **17, 31, 33, 65 and 480** still
  diverge. The comparison commands correctly exit nonzero. Reference tokens have
  not been replaced. Detailed results are in `boundary-comparison.json`; the
  pinned reference is [the earlier C8 capture](../2026-10-17-learned-index/reference-c8.json).
- All eight rank audits show frozen arenas, planned shared workspace, zero
  activation host transfers, zero workspace growth and zero retained-step device
  allocations/internal completion fences.

Arena: **12,451,963,904 bytes/rank**, 1,028,608 bytes less than the previous BF16
index arena. Shared workspace: **130,916,352 bytes/rank**, unchanged. The standard
fixture's observed HBM used range was 33,648,369,664–33,930,792,960 bytes across
ranks; this is a device-wide observation, not a 64K capacity or peak-memory claim.

Isolated serial diagnostic: 294 decode samples, mean **19.188 ms/token**, median
**19.083 ms**, range 18.857–19.794 ms. The probe reports each step's duration
directly; these are not differences of cumulative timestamps. This measurement
includes native sampling/consensus, excludes HTTP, and retains the reference
mismatch caveat. It is not a production performance acceptance result.

Artifacts: `native-results.json`, `native-audit.json`, `boundaries-audit.json`,
`serial-audit.json`, `boundary-comparison.json`, `serial-performance.json`.

HTTP regression passed with the vLLM 0.25.1 Rust frontend: full fixture text/SSE
agreement and usage, six concurrent requests, queued isolation, cancellation
during peer prefill, replacement requests, context overflow rejection,
unsupported stochastic sampling rejection and graceful worker shutdown. With
token budget 8, the 504-token prompt's mixed-workload TTFT was **5.388 seconds**;
the short peer completed in **715.7 ms**. Short isolated SSE text chunks arrived
about 19.3–19.5 ms apart; a text chunk is not necessarily a token. Shared-pool
accounting recorded 17 admissions, 864 cross-partition bindings and zero occupied
bytes after shutdown. All eight HTTP rank audits pass the residency/frozen-arena
checks above. See `http-results.json` and `http-audit.json`. These timings do not
meet the remaining long-context performance goals and are not a throughput claim.

SHA-256:

- Native library: `567f13711ebcd3631579b07a5522ff028d30201bc38969c7d49040c909647470`
- Execution binary: `be0495be755faac6e2d3120733d10692d1bcbc155debb0029879598e4f333045`
- Rust server: `f964e6e2c77df8806656ebc70ec7baba5a1673b2a3b3094e2a60760313746d06`

Next correctness gate: compare intermediate values and forced-prefix logits on
the first failing 17-token prompt against the pinned C8 reference. Short contexts
include every key in top-2048, so agreement here still cannot prove correct
learned exclusion beyond 2048. Long-context page tables, SWA recycling, hybrid
prefix checkpoints and 64K mixed-workload qualification remain required.
