# Planned bulk collective domain — 2026-10-18

Full 92B discovery, fixed-workspace and frozen-arena runs pass 39/39 reference
tokens over six interleaved 8K/short requests. All eight ranks preserve decode
graph replay, zero activation-host transfers, zero execution allocations or
internal completion boundaries, fixed workspace and drained transient metadata.

`IFTOKN09` explicitly selects AI_CPU AllReduce for uncaptured bulk/mixed forwards
and HOST for pure decode, AllGather and sampler consensus. Two independently
initialized communicators use explicit 200 MiB buffer configuration each. The
choice is visible in the generic viewer and independent of compact output rows.
Older binaries reject v9. No global expansion-mode mutation or new fence is used.

In separate unprofiled runs of the same fixture, six batches of at least 4000
tokens have median completed-forward time **485.931 ms**, versus **2174.087 ms**
for the preceding HOST-only candidate. Three single-sequence decode batches have
median **22.115 ms**, versus **21.503 ms** previously. These are small samples from
separate runs, not a controlled alternating A/B or an HTTP latency claim.

Frozen activation arena remains 38,009,938,432 bytes/rank; shared workspace remains
159,221,248 bytes/rank. Total measured HBM is 59.740–60.025 GB/rank, approximately
500–506 MB/rank higher than the preceding run. HCCL owns allocations outside the
arena; its configured buffer size is not its total allocator overhead.

The candidate remains compilation opt-in. Long-context HTTP mixed scheduling,
prefix recovery, sustained service and broader quality gates are separate work.

## Reproduction and evidence

Local WSL compiler checks: 14 unit tests, 24 token-format tests, four physical
program tests and ARM cross-build. The generated viewer passes collective summary
and detailed domain/buffer visibility checks. Native build runs only inside
`inferfabric-92b` on 7.156.99.58.

Workspace: `/data/p00603624/inferfabric/openpangu92`. Build/source:
`native-build-bulk-domains` / `native-bulk-domains-source`. Compile the existing
`output-rows.inferfabric` (canonical learned-C8 DSL, context 65536) with
`INFERFABRIC_EXPERIMENTAL_OUTPUT_ROWS=1` and
`INFERFABRIC_EXPERIMENTAL_BULK_COLLECTIVES=1`, locally cross-built
`../inferfabric-compiler-bulk-domains compile-token output-rows.inferfabric
/models/92B bulk-domains-source --allow-stale-index-size --resident-sequences 3
--prefill-tokens 4096 --shared-pages`.

Use normal CANN/vendor environment, global HOST expansion, HCCL base port 43800,
decode graph and experimental-long-context flags. Run
`qualify-sequence-serving.py --batch-trace` with the new program probe,
`bulk-domains-source`, existing `device-zero-apc8k-fixture` requests, bindings and
reference, output prefix `bulk-domains8k`. The resulting bundle is
`bulk-domains8k-frozen-plan`; no earlier arena is retagged.

`summary.json` records exact library/program hashes, latency samples and HBM
comparison. Frozen raw tokens, process result and all resource audits are
preserved; earlier stages retain their gate summaries. The previous comparison
is `../2026-10-18-output-rows/output-rows8k-frozen`.
