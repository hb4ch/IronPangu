# Prepared prefill family: native and Rust HTTP qualification

2026-10-17. The compiler can declare exact prefill widths with
`--prefill-variants 2,4,8,16`. Version-6 execution binaries and serving manifests
carry the same finite family. Existing binaries keep their single-size contract.
Each width has its own prepared ACLGraph, output lease and custom-kernel lifetime;
weights, resident state and the maximum workspace are shared. Preparation covers
all widths before arena binding and seals graph creation before readiness.

The native ABI reports the exact supported mask. The Rust adapter rejects a
manifest/backend mismatch and the scheduler chooses the largest declared width
that fits the remaining prompt and token budget. Single-token tails use the
explicit token entry. Physical recipes and the generic viewer show each width's
shapes, API repetitions and assembly copies, retaining maximum storage dimensions
and control offsets. This is not latency-cost-based scheduling or multi-request
packed decode.

## Full 92B NPU results

TP8/EP8, C8 learned-index checkpoint path, two residents, 4,096 total context.
All **123 immutable reference tokens** pass in each of discovery, planned-workspace
preparation and frozen replay, including a 529-token prompt, shuffled page mappings,
interleaved residents and slot reuse. Every rank replays T=2/4/8/16 graphs
7/10/10/100 times respectively in the final native test (including one warm replay
per graph). These counts are checked against the exact prompt decomposition.

The maximum workspace stays **131,020,288 bytes/rank**. The frozen arena is
**12,620,067,840 bytes/rank**, an 8,817,664-byte increase over the single T=16
batched-rotary arena. All-rank audits show zero activation host transfers,
workspace growth, replay allocations or internal completion fences. Native
resource/correctness qualification overlapped the independent reference on
devices 8–15; its recorded timings are not isolated performance evidence.

## Rust HTTP results

The pinned vLLM 0.25.1 Rust frontend passes ordinary text/SSE fixtures, six concurrent
requests, cancellation while a 4,088-token peer prefills, short-peer progress,
context/unsupported-sampling rejection and graceful shutdown. Token budget is 32.
The overload burst returns eight accepted requests and eight typed 503 responses
with `Retry-After: 1`; recovery output is correct. All 18 admissions are reclaimed,
leaving zero shared-pool occupancy. Every rank replays all four prefill graphs
(11/18/19/256), with the same frozen/resident invariants.

After the independent reference stopped, this HTTP run measured long-peer TTFT
14.720 s and short-peer TTFT 271 ms. Streamed text-chunk intervals are recorded in
`http-results.json`; they are not necessarily individual token latency.

## Local and reader checks

Isolated native long tests (three repetitions, no reference or HTTP overlap) have
mean TTFT 7,195.364 / 10,614.408 / 14,095.962 ms for 2,049 / 3,072 / 4,088 input
tokens, with TPOT 19.978 / 20.044 / 20.075 ms. All match through reference EOS and
are stable across repetitions. The 3,072-token fixture still fails its forced
post-EOS continuation, as the previous single-size backend did; `long-comparison.json`
preserves this failure. The other two full eight-token outputs pass. Long-run
audits retain the frozen, allocation-free, device-resident contract on every rank.

- 19 compiler token tests, four viewer tests and six native Rust library tests pass.
- Nine scheduler tests pass, including mixed workload fairness, cancellation,
  shutdown, rank failure and the new family budget/tail test.
- Native reader tests accept binary versions 2–6 and reject ten malformed families.
- Native plan verification accepts all eight rank bindings and rejects 36 invalid
  kernel contracts. Eight Python qualification-tool tests pass.
- Rust binaries cross-compiled locally in WSL; native builds and NPU runs stayed
  inside the remote Docker container.

`identities.json` pins the executable, backend, frontend and reference. Compact
audits preserve SHA-256 identities of the full remote audits. This milestone does
not qualify 64K serving, resolve known numerical debt, implement hybrid prefix
snapshots/COW, or provide a production latency guarantee. Default admission remains
512; the larger envelope still requires the explicit experimental opt-in.
