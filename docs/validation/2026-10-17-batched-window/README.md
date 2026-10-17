# Batched BF16 window attention

Date: 2026-10-17. Window attention and its metadata producer now process all
query rows in one call. One 4,096-byte metadata buffer is shared by consumers
of the same declared window. The physical recipe/viewer exposes query-count
controls, final device KV length, per-row causal bounds, batched output and zero
output assembly copies. Absolute positions remain distinct from ring addresses.

## Independent operator checks

The CPU oracle uses distinct query rows, BF16 key/rotary values and sinks. The
probe prewrites the chunk, poisons storage outside the retained union and checks
each row's own window. It covers page and ring boundaries through position 65,535.
[Raw results](operator-probe.log).

| Queries | Window | Values checked | Exact graph replays | Maximum absolute error |
|---:|---:|---:|---:|---:|
| 1 | 512 | 626,688 | 136 | 0.000134930 |
| 4 | 512 | 626,688 | 34 | 0.000160951 |
| 16 | 512 | 2,506,752 | 34 | 0.000160951 |
| 4 | 2,048 | 626,688 | 34 | 0.000164491 |

Tolerance is `0.001 + 0.02 * abs(expected)`. Production-state scatter also passes
65,536 addresses, 524 graph replays, peer preservation and page reassignment.
These operator tests do not establish full-model 64K correctness.

Multiple probe instances exposed a test-harness descriptor-ownership bug: tensor
lists received descriptors already owned by `DeviceTensor`, causing allocation
failure/segmentation fault in the combined run. Lists now own separate descriptors;
the final combined run passes. Production cache-view lists already do this.

## Full 92B output and timing

Discovery, preparation and frozen execution each pass all 99 standard tokens,
including interleaved residents and permuted pages. Frozen replay counts remain
106 decode and 28 prefill. Window-attention preparations fall from 450 to 180;
metadata preparations from 15 to 6. Per four-token chunk, the 30 BF16 layers now
issue 30 calls instead of 120 and remove 120 output assembly copies.
[Short results](short-results.json), [audits](short-audit.json).

All 72 long-run native tokens remain identical to the preceding implementation.
Each prompt runs three times with eight forced outputs. All match the independent
budget-four reference through EOS. The known post-EOS difference at 4,088 remains,
so full-reference qualification still fails. [Comparison](comparison.json).

| Prompt tokens | Previous mean TTFT | Batched window TTFT | Reduction | Mean TPOT |
|---:|---:|---:|---:|---:|
| 2,049 | 17,490.985 ms | 15,294.150 ms | 12.56% | 20.120 ms |
| 3,072 | 26,059.916 ms | 22,760.470 ms | 12.66% | 20.127 ms |
| 4,088 | 34,691.198 ms | 30,265.066 ms | 12.76% | 20.153 ms |

Warm model, serial requests, four-token chunks, three repetitions. TTFT excludes
loading, HTTP and queueing. Operator tests finished before this native run; HTTP
started afterward. These are separate measurements, not confidence intervals.
[Before/after](performance.json).

Maximum shared workspace rises by 20,992 to **130,937,344 bytes/rank**, planned
before binding. Metadata shrinks from 16,384 to 4,096 bytes for this plan's window.
The net arena increase is 8,704 bytes, to **12,536,883,712 bytes/rank**. All eight
[long-run audits](long-audit.json) report frozen arenas, zero workspace growth,
activation host transfers, retained-step allocations and internal completion
boundaries. Live process memory remains 32,120 MB/rank, a sample rather than peak.

## Rust HTTP regression

The vLLM Rust 0.25.1 frontend passes full fixture text, SSE agreement, six
concurrent requests, queue isolation, disconnect recovery and cancellation during
a peer's 504-token prefill. The short request finishes first; its text-chunk
intervals are 50.4–51.0 ms, not guaranteed individual-token intervals. Two active
residents, token budget eight and 25 mixed ticks are exercised. All 17 admissions
retire their shared pages, leaving zero occupancy. Context overflow and unsupported
sampling are rejected; SIGTERM drains and closes workers. All eight HTTP
residency/arena audits pass. [Results](http-results.json), [audits](http-audit.json).

This HTTP regression uses an experimental 4K bundle but executes prompts only
through 504 tokens. It does not qualify concurrent 4K behavior for this revision,
multi-request batched decode or prefix reuse. Artifacts are `batched-window-http`.

## Reproduction details

Rust builds ran locally in WSL; native builds/tests ran in Docker `inferfabric-92b`
on 7.156.99.58, directory `/data/p00603624/inferfabric/openpangu92`. Run
`native-build/sliding_window_probe all` for the operator suite. Full-model artifacts
use `batched-window-{source,discovery,workspace,preparation,bound,qualified,long-qualified}`.
Use the same 4K learned-index DSL, two residents, shared pages, four prefill tokens
and budget-four reference as the preceding batched-attention experiment. Rebuild
workspace/arena bindings for the new backend fingerprint. The long run uses
`--permuted-pages --packed-prefill --experimental-long-context --output-diagnostics`;
retain its nonzero comparison result.

Seventeen compiler tests pass, including one metadata buffer per window, batched
invocation counts and shapes at query counts 2/4/16. Full-model evidence here uses
four queries; sixteen-query support is operator evidence in this revision.

| Artifact | SHA256 |
|---|---|
| Native library | cf8151d17fd727528f2823ca1705c6a38d175da6bf7bb7bd5be49f9873ed11c2 |
| execution.bin | 3b0c4789a6743ffdbc5b22e9d4b7dd67651e97b0840f0559945ba178169d1084 |
| Rust frontend | a81c91652b82e4917b8db7e767138c51f75fc3f2baa10ce799d476ae37ec582d |

Next: qualify larger full-model chunks with both attention paths batched, then
support multiple compiled chunk sizes under a measured decode-latency budget.
Per-query recurrent/projection work, numerical parity, multi-request batched
decode, hybrid prefix restore and robust 64K serving remain incomplete. Default
context admission remains 512.
