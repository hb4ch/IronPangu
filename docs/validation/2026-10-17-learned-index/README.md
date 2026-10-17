# Learned-index decoder integration: experimental

Date: 2026-10-17. OpenPangu V2 92B, TP8/EP8, two residents, four-token packed
prefill, shared page ownership. **This is not a passed C8 parity or 64K release
gate.** The existing qualified C8 DSL remains unchanged. The additional
`92b-tp8-ep8-c8-learned.inferfabric` is an experimental executable definition.

## Implemented path

The DSL expands eight full indexers and sixteen attention consumers. Each full
indexer explicitly projects queries, keys and head weights, normalizes keys,
applies leading-column BF16 rotary, appends a BF16 paged index key, and selects
2048 logical token indices with `aclnnLightningIndexerEnhance`. Its later shared
consumer uses the same device SSA result and its own C8 attention cache. Named
record carries express this dependency without executor layer-name logic.

Index keys use 64-token pages and participate in the existing per-rank request
reservation, binding, resident switching, clearing and reclamation. C8 attention
consumes the selected indices directly. The physical viewer exposes kernel APIs,
index-cache storage, logical index shape, sharing edges and append-control offsets.
No activation is transferred to host and no implicit fallback was added.

## Verified scope and failed gates

- [Native fixtures](native-results.json): all 99 expected tokens pass in a frozen
  arena with shuffled pages, packed prefill and interleaved residents.
- [Boundary comparison](boundary-comparison.json): all 336 tokens agree between
  isolated serial and packed/interleaved execution, over lengths 1–504 and three
  repetitions. Thus these execution modes agree with each other.
- The previous causal-candidate oracle does **not** match all those outputs.
  The original qualification commands correctly exited with failure. Expected
  outputs were not replaced to turn those runs green.
- A fresh independent [OmniInfer C8 reference](reference-c8.json), using the same
  checkpoint, TP8/EP8 and input IDs, matches **8 of 14 full sequences**. Six
  diverge. **Independent full-reference qualification remains failed.**
- [HTTP qualification](http-results.json) passes the three normal full-text
  fixtures, SSE/usage consistency, six concurrent requests, cancellation during
  peer prefill, slot replacement, context-overflow rejection and graceful shutdown.
  The mixed 504-token case also matches. These passes do not override the failed
  broader boundary comparison.
- [Native](native-audit.json), [packed boundary](boundaries-audit.json),
  [serial](serial-audit.json), and [HTTP](http-audit.json) audits pass on all eight
  ranks: no activation host transfers, retained-step device allocation/internal
  fences, or workspace growth. Arena: **12,452,992,512 bytes/rank**; shared
  workspace: **130,916,352 bytes/rank**. This is 2,185,728 extra arena bytes/rank
  relative to the previous shared-page executable.

The native path currently uses **BF16 index queries/keys**. Inspection of the
reference establishes a concrete contract gap: OmniInfer `int8_ds_mla` dynamically
quantizes index queries and keys to INT8 and converts query/key scales and head
weights to FP16 before its quantized indexer. Attention-cache C8 precision is a
separate decision. This difference must be implemented explicitly in the plan;
it is not evidence that it alone explains every observed divergence. All tested
contexts are below topk=2048, so none proves learned exclusion of old keys.

## Measurements

HTTP mixed workload: 504-token prompt TTFT **5.149 s**, short replacement total
**668 ms**. Previous shared-page result was 4.881 s and 628 ms. Isolated short
stream text-chunk intervals are approximately **19.0–19.3 ms**, versus roughly
18.2–18.5 ms previously; these are not labeled TPOT because chunks may coalesce
tokens. Native token timing records are retained in the fixture report; its
interleaved request intervals include peer work.

HTTP state ownership: 17 admissions, 1,296 component/rank bindings outside old
partitions, peak owned state 212,303,872 bytes across eight ranks, zero remaining
ownership at shutdown. This is ownership within preallocated HBM, not device-wide
memory usage.

## Reproduction and identity

Compile the experimental DSL with `--resident-sequences 2 --prefill-tokens 4
--shared-pages`; discover workspace, prepare with the maximum workspace, then
bind the frozen arena. Final bundle: `learned-index-v1-final-bound`. Its execution
binary is byte-identical to `learned-index-v1-bound`; final inspection metadata
adds the index-cache/control details. Raw remote runs are `learned-index-v1-qualified`,
`learned-index-v1-boundaries`, `learned-index-v1-serial-diagnostic` and
`learned-index-http-v1`.

The independent reference ran in Docker on devices 8–15:

```sh
REFERENCE_NPU_DEVICES=8,9,10,11,12,13,14,15 \
bash scripts/reference-openpangu-decoder.sh \
  --cases resident-boundary-cases.json --kv-cache-dtype int8_ds_mla \
  --output learned-index-omni-c8-boundary-reference.json
```

For token-ID cases the reference ignores EOS and generates exactly the requested
output count, matching the diagnostic native probe. Torch remains offline only.
Rust compiler builds ran locally in WSL; native builds/execution ran in Docker.
DSL/compiler tests pass, including the eight-producer/two-consumer graph contract.

| Artifact | SHA-256 |
|---|---|
| Native library | `360af14a1360fb5e4372ade6c6c87582f3fdaefea018649c57dba6e40729c5dc` |
| Execution | `545535bd19a34edc97471c11ad11aea4586d496403516a60e0565fa92ecaa045` |
| Rust server | `f964e6e2c77df8806656ebc70ec7baba5a1673b2a3b3094e2a60760313746d06` |

Next: explicit INT8 index-query/key and FP16 scale contracts, resolve reference
divergences, then remove fixed attention/page-table limits with SWA reclamation
and qualify positions beyond 2048 and through 64K. Hybrid prefix checkpoints,
larger efficient prefill chunks and production robustness remain open.
