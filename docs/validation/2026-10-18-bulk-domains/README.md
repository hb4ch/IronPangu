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


## 64K vLLM Rust HTTP qualification

The completed HTTP run passes through-first-EOS references for early/middle/late
retrieval prompts (65,520 input tokens), SSE/collected agreement, queued isolation,
disconnect recovery, short-request progress, context overflow, prefix salt
isolation and eviction, and graceful shutdown. Serving configuration is
`max_num_seqs=2`, `max_num_batched_tokens=4096`; the compiled resident capacity is
three. It executes 405,921 input tokens with 38 admissions and six mixed ticks;
all input histories are accounted for and shared-page occupancy drains to zero.

Mixed long-request TTFT: **11.751 / 11.695 / 11.714 seconds**. Mixed short-request
TTFT: **739 / 759 / 763 ms**; all short requests finish before their long peer.
Short SSE text-chunk intervals are **508–547 ms**, not isolated token TPOT.
Continuous batching makes progress, but a decode request sharing a bulk forward
still experiences its chunk execution latency. This remains an optimization gap.
The preceding candidate's separate run had long TTFT 33.985–49.693 seconds and
short TTFT 1.313–3.080 seconds; this is a historical comparison, not a controlled
alternating trial.

Cold-prefix TTFT is 9.529 seconds for salt A, 11.063 seconds for salt B, and
11.052 seconds after eviction. Warm TTFT is 2.074 / 2.014 seconds with 61,440
cached tokens and a 4,080-token suffix. Two hits restore 122,880 tokens; six
snapshots are published and no cache entries remain after shutdown. Snapshot
copies remain device-to-device: 6,952,009,728 bytes in 1600 copies per rank over
the full run. No prefill graph replays occur. All rank resource and residency
checks pass. Long-context native TPOT is measured separately.

Reproduction: same frozen bundle/library and environment, HCCL port 44000,
`qualify-program-http.py`, unchanged `inferfabric-server-apc-device-zero` vLLM Rust
frontend, `batched16-preparation-reference.json`,
`retrieval-64k-reference.json`, `device-zero-apc-prefix64-case.json`, port 18085,
packed prefill, first-EOS and arithmetic-history checks, 300-second request
inactivity timeout. Raw evidence is `bulk-domains64k-http/`; `results.json` is the
terminal completed report. Intermediate `*-progress.json` files are not terminal
status records.


## Native 64K completion and TPOT

A separate unprofiled frozen-only run passes all 39 reference tokens over six
requests and 196,629 completed input tokens in 53 batches. The first 65,520-token
prompt executes exactly **15 x 4096 + 4080** input rows; no per-token prefill or
long-prefill graph is used. Four pure single-sequence decode calls after long
context take **28.120 / 22.424 / 22.414 / 22.908 ms**, median **22.666 ms**. This
small native completed-call sample is not a sustained production TPOT percentile.
All eight rank residency, frozen arena, workspace, graph replay and metadata
lifetime checks pass. `native64-summary.json` and
`bulk-domains64k-native-frozen/` preserve the samples and full evidence.

Reproduce with a fresh `prepare-sequence-fixture.py` output from the frozen plan,
`retrieval-64k-reference.json`, arithmetic peer in
`batched16-preparation-reference.json`; run `qualify-sequence-serving.py
--frozen-only --batch-trace` with the same backend and HCCL base port 44200.
No profiler is enabled in this run.
