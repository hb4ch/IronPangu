# Shared frozen pool and 8K batch traces — 2026-10-18

PASS: discovery, fixed-workspace and `IFARENA2` frozen execution each match all
39 reference output tokens across six requests, through first EOS. Driver and
native processes exited 0. This is native qualification; HTTP mixed-tail
qualification is a separate gate.

The previous instruction-to-slot eligibility policy failed HTTP requests even
when the fixed arena had sufficient free storage. Equal-capacity closure fixed
one case but left gaps for unsampled tail sizes. Limiting eligible backing
capacities to a sampled maximum also failed: a 1,083,392-byte request may need
to lease a larger free slot while all smaller slots are live.

The new contract limits requested bytes per instruction, and permits any
sufficient free slot in the fixed shared pool. Workspace remains permanently
reserved and excluded from temporary selection. Live leases, owning-stream
ordering, capture addresses, binary fingerprints and failure on exhaustion are
preserved. This does not introduce arena growth, copies or synchronization.
See [the format and contract](../../native-physical-planning.md).

[Results](results.json) and [completed batch/output trace](tokens.jsonl) include:

- Three 8,176-token retrieval prompts, each actually executed as 4,096 + 4,080
  prefill tokens, interleaved with three arithmetic requests.
- 26 native batch calls, 24,597 completed input tokens, validated contiguous
  request positions, sampling boundaries and total batch budget.
- All eight ranks: activation host transfers zero. Frozen run: no recorded
  step device allocations or internal fences; workspace growth zero.
- Arena 39,572,690,960 bytes/rank and workspace 159,221,248 bytes/rank, unchanged
  from the prior 64K-capacity plan. Only 8K input correctness is qualified here.
- Single-sequence pure-decode ABI median 21.427 ms (7 calls); two-sequence
  median 28.836 ms (11 calls). These include sampling/consensus, exclude mixed
  prefill calls and do not include scheduler waiting. No division by batch size.
  The small sample and outliers are retained; this is not a sustained-load SLO.
- Long-request first-output intervals 14.050/12.816/12.196 seconds, including
  offline scheduling waits and excluding model load; not HTTP TTFT.

The compact per-rank binding file is 103,461 bytes versus 37,086,462 bytes in
the rejected expanded-envelope attempt. `arena.json` is approximately 9.2 MB,
versus approximately 999 MB. The viewer distinguishes request limits from
observed addresses and retains workspace demand/access information.

Checks: 10 portable Python tests; Docker C++ parser/allocator tests including
18 negative contracts, 4,096 tail sizes, oversized backing and live-lease
isolation; actual generated viewer kernel/collective/workspace/pool tests.
Native library and probe built inside Docker; no remote Rust compilation.

Run `qualify-sequence-serving.py --batch-trace` with the same `sequence64k-source`
and `sequence8k-fixture` inputs documented in the
[8K baseline](../2026-10-18-sequence-8k/README.md), output prefix `pooled8k-clean`.
The new backend requires fresh fingerprints and preparation. For later input
lengths, `--frozen-only` reuses a bound plan without repeating discovery; it
still checks exact output, resource audits and completed batch geometry.

An earlier run (`pooled8k-serving`) was aborted after confirmed out-of-memory
and SDMA errors: an external `test_npu_load` container was allocating 40 GB on
every device. It was stopped under the user's existing authorization. The
failed run is excluded from these results. A read-only query confirmed roughly
65 GB free/device before the clean run. No device reset was performed.
