# Batched learned-index selection

Date: 2026-10-17. INT8 sparse-index selection now executes one TND call per
prefill chunk rather than one call per token. For four-token chunks and eight
selectors this removes 24 indexer calls and 32 output-row assembly copies per
chunk. Output rows remain device aliases for the declared attention consumers.
BF16 index selection retains its existing per-query schedule.

The compiler's physical recipe and viewer expose one API invocation, zero assembly
copies, `[T,1,2048]` output and the initialization-only query-count array.
Selection consumes the final device KV length and query count `T`; causal
alignment excludes later chunk rows. All keys are appended on the same stream
before the selector. No activation readback, implicit cast or fence was added.

## Full-model results

All 99 standard fixture tokens pass discovery, preparation and frozen execution,
including interleaved residents, page permutation and reuse. Frozen execution
reports 106 decode and 28 packed-prefill graph replays; indexer preparations fall
from 120 to 48. [Short results](short-results.json), [rank audits](short-audit.json).
Fifteen local compiler tests pass, including INT8/BF16 physical schedule and
output-shape checks at chunk sizes 2, 4 and 16. Rust was cross-compiled locally;
native compilation and all NPU execution ran in Docker.

All 72 long-run native output tokens are identical to the
[previous implementation](../2026-10-17-long-index-selection/README.md), across
three repetitions each of 2,049-, 3,072- and 4,088-token retrieval prompts.
All match the independent reference through EOS. The previously recorded
post-EOS difference at 4,088 remains, and the full-reference qualification still
returns failure. This optimization does not resolve that numerical difference or
qualify general long-context correctness. [Comparison](comparison.json).

| Prompt tokens | Previous mean TTFT | Batched mean TTFT | Batched mean TPOT |
|---:|---:|---:|---:|
| 2,049 | 19,349.869 ms | 19,064.159 ms | 20.093 ms |
| 3,072 | 29,033.372 ms | 28,549.600 ms | 20.089 ms |
| 4,088 | 38,660.339 ms | 38,050.014 ms | 20.108 ms |

These are separate warm-model runs with serial requests, four-token prefill,
eight forced outputs and three repetitions. Loading, HTTP and queue time are
excluded. No reference or HTTP test ran concurrently with the timed native run.
The measured TTFT reduction is 1.48–1.67%; this is modest and not a confidence
interval or evidence of production throughput. [Before/after](performance.json).

The frozen arena remains 12,536,874,496 bytes/rank; maximum shared workspace is
130,916,352 bytes/rank. All eight [long-run audits](long-audit.json) report zero
activation host transfers, workspace growth, retained-step device allocations and
internal completion boundaries. The live process-memory sample remains 32,120 MB
per rank, not a peak. Old arenas are rejected by backend fingerprint; discovery
and preparation were rerun before binding this implementation.

## Rust HTTP regression

The current vLLM Rust 0.25.1 frontend passes full fixture text, SSE agreement,
six concurrent requests, queued isolation, disconnect recovery and cancellation
during a peer's 504-token prefill. It exercised two active residents, token
budget eight, 23 mixed scheduling ticks and 175 prefill graph replays per rank.
All 17 admissions retired their pages: remaining occupied bytes were zero.
Context overflow and unsupported sampling were rejected; SIGTERM drained and
closed workers. All eight HTTP arena/residency audits also pass.
[Results](http-results.json), [audits](http-audit.json).

This remains chunked prefill with serial decode; it does not establish batched
decode, 4K concurrent HTTP correctness, non-greedy sampling or prefix reuse.
The HTTP test used the 4K bundle with the explicit experimental opt-in, but
its longest executed request was 504 input tokens. An initial attempt selected
an old frontend binary and rejected recycled state before loading; rebuilding
the current frontend locally fixed the test deployment without a source change.

## Reproduction and remaining work

Use the prior experiment's model and reference fixtures. Compile with two
residents, shared pages and four prefill tokens. Run the normal discovery,
maximum shared workspace, preparation and frozen binding stages. Native artifacts
are `batched-index-{source,discovery,workspace,preparation,bound,qualified,long-qualified}`
under `/data/p00603624/inferfabric/openpangu92`, Docker `inferfabric-92b`, host
7.156.99.58. Long qualification uses `--permuted-pages --packed-prefill
--experimental-long-context --output-diagnostics` and the unchanged
`index-select-reference-c8-chunk4.json` oracle. Use
`scripts/compare-program-reference.py` to retain mismatches before returning
failure; timings use individual native step durations.

| Artifact | SHA256 |
|---|---|
| Native library | 0d9c48e55251441ea1c32b3f8b6332b6e3c594279feba83b7580faf593c75c2a |
| Rust frontend | 077d9c060b0838a53d68724385092bc15a6d09540bd39179a491e18396c3ddfd |
| execution.bin | 3b0c4789a6743ffdbc5b22e9d4b7dd67651e97b0840f0559945ba178169d1084 |

The executable instruction sequence is unchanged; physical recipes and the
backend implementation change. Full decoder 64K qualification remains open.
Next compare larger prefill chunks against a reference with the same chunk
budget, batch remaining per-query attention work, and qualify diverse
8K/16K/32K/64K prompts, concurrent requests, pressure and hybrid prefix reuse.
The default context admission limit remains 512.
