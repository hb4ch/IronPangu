# Fixed maximum shared workspace — 2026-10-17

Workspace growth is fixed by measuring the maximum first, then starting a fresh
preparation process with one permanent shared workspace lease. Both C8 and BF16
plan **130,916,352 bytes per rank**, in slot 0. Every instruction uses that fixed
address on its owning stream. An oversized request fails before enqueue.

[Inspect the C8 plan and fixed arena](execution.html) ·
[Planning protocol](../../shared-workspace-planning.md) ·
[Pre-fix baseline](../2026-10-17-packed-c8/README.md)

## Measurements

Full 46-layer 92B, TP8/EP8, batch one, context capacity 512, three prompts.

| Format | Graph TPOT before → after, ms/token | Arena before → after, bytes/rank | Median HBM before → after, GiB |
|---|---:|---:|---:|
| C8 | 17.9239 → 17.9611 | 12,523,057,728 → 12,408,071,232 | 31.4607 → 31.3535 |
| BF16 | 17.7293 → 17.6792 | 12,433,409,600 → 12,412,394,048 | 31.3709 → 31.3513 |

C8 saves **114,986,496 arena bytes (109.66 MiB) per rank** and measured median HBM
falls by **109.79 MiB per rank**. BF16 saves 20.04 MiB of arena space and 20.05 MiB
of measured HBM. This removes intermediate workspace-growth buffers, including
C8's obsolete 93,970,944-byte allocation and the earlier approximately 20 MiB
allocation. The final maximum workspace remains available to every operator.

C8's fixed arena is 4.12 MiB smaller than BF16's. Total device HBM remains slightly
higher for C8 (about 2.25 MiB in this sample), because it includes graph/runtime
allocations beyond the arena. Persistent KV cache sizes remain unchanged:
22 MiB/rank for mixed DSA C8/SWA BF16 and 25.875 MiB/rank for all BF16.

Graph TPOT is effectively unchanged in these short samples; this milestone makes
no latency-speedup claim. C8 median TPOT is 17.8656 ms and sample p95 is 18.5611 ms.
C8 eager execution also passed, averaging 175.6271 ms/token in this run, versus
149.3042 ms/token in the earlier run. These are single three-prompt samples and
are not production latency SLOs or a controlled attribution of eager timing changes.

Each graph measurement contains 30 inter-output intervals. Timing excludes the
first output and includes decoding, selection, consensus and output delivery.
Before measurements use the previous committed binary; both after measurements
use the same new binary. No probe overlapped the final bound benchmark runs.
HBM medians use 24 device-wide snapshots per mode, taken after generation while
the decoder is alive. Raw snapshots, free-memory baselines and hashes are in
`measurements.json`; arena savings are exact rather than sampled.

## Correctness and residency

- All three C8 graph and eager runs reproduce their unchanged **complete** output
  records: token IDs, selected logits and top-five probabilities. BF16 graph runs
  reproduce the unchanged BF16 fixture. Final runs contain **99 output records**.
- C8 still produces the same 33 greedy tokens as BF16 for arithmetic, geography
  and Chinese prompts. C8 versus BF16 numerical differences are unchanged from
  the preceding milestone; no quantization or attention kernel was changed here.
- **72 rank audits / 1,704 token steps** verify zero workspace growth, zero token
  allocations, zero internal completion fences and zero activation host transfers.
  Graph replay is required explicitly. Bound execution uses one arena allocation;
  checkpoint loading is separate.
- Discovery and fresh fixed preparation each passed the same full-model fixtures
  for both formats before binding. The binder refuses discovery/growing pools.
- Python regressions cover maximum merging, program/backend consistency, capture
  evidence, growth/overflow rejection and workspace instruction bindings.
  Native portable tests reject 16 malformed arena/workspace contracts. The NPU
  prepared-execution probe verifies fixed addresses, overflow rejection and
  existing lease, transfer and arena contracts.
- Compiler/DSL tests passed locally. Viewer tests pass for the new bound artifact
  and the historical C8 artifact without workspace metadata.

The implementation was committed as `79a74cd`. The C8 and BF16 execution programs
are unchanged from the prior qualification. `artifacts.sha256` records the new
native binary, programs, arenas and executable workspace plans. The two
`*-workspace-plan.json` files retain per-instruction maximum demands and discovery
audit hashes. The viewer exposes capacity, permanent slot, instruction demand and
zero growth count directly from bound metadata.

The bound is the maximum observed under these workloads and qualified shapes.
An untested kernel/shape requesting more fails; it never silently grows or falls
back to host execution. Long-context, multi-sequence and multi-stream scheduling
are outside this qualification.

## Reproduce

Rust compilation stays local via the cross-build; native build and NPU execution
stay inside Docker. Source CANN and the Omni training/inference operator environments,
and use fresh directories and HCCL port ranges. For the recorded C8 run:

```bash
python3 scripts/benchmark-native-tpot.py --binary native-build/openpangu_decode \
  --plan cache-c8-plan --checkpoint /models/92B --cases cache-c8-preparation/results.json \
  --output workspace-c8-discovery --prepare-arena --decode-graph --resident-audit \
  --hccl-base-port 40000
python3 scripts/plan-shared-workspace.py cache-c8-plan workspace-c8-plan \
  workspace-c8-discovery/arithmetic-0/prepared \
  workspace-c8-discovery/geography-0/prepared \
  workspace-c8-discovery/chinese-0/prepared
python3 scripts/benchmark-native-tpot.py --binary native-build/openpangu_decode \
  --plan workspace-c8-plan --checkpoint /models/92B --cases cache-c8-preparation/results.json \
  --output workspace-c8-preparation --prepare-arena --decode-graph --resident-audit \
  --hccl-base-port 40256
python3 scripts/bind-prepared-arena.py workspace-c8-plan \
  workspace-c8-preparation/arithmetic-0/prepared workspace-c8-bound-plan
python3 scripts/benchmark-native-tpot.py --binary native-build/openpangu_decode \
  --plan workspace-c8-bound-plan --checkpoint /models/92B --cases cache-c8-preparation/results.json \
  --output workspace-c8-bound-tpot --decode-graph --resident-audit --hccl-base-port 40512
```

Repeat the last command without `--decode-graph` for `workspace-c8-bound-eager` on
port range 40768. Repeat discovery/planning/preparation/binding/graph measurement
for BF16 with `composite-final-plan`, immutable `paired-linear-eager/results.json`,
`workspace-bf16-*` outputs and port ranges 41024, 41280 and 41536.

```bash
python3 scripts/summarize-packed-c8-qualification.py . shared-workspace-measurements.json \
  --c8-stem workspace-c8 --bf16-stem workspace-bf16 --require-planned-workspace
node scripts/test-token-viewer.cjs docs/validation/2026-10-17-shared-workspace/execution.html
```

The viewer is a deterministic template refresh of the bound bundle, retaining its
compiled program and measured arena metadata without hand-authored decisions.
