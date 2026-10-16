# Native resident TPOT — 2026-10-16

Measured on `7.156.99.58`, inside Docker `inferfabric-92b`, using the qualified
92B TP8/EP8 binary without changing or rebuilding it. NPU process inspection
showed no other running NPU processes before this measurement.

**Pooled mean TPOT: 2,046.86 ms/token (0.489 tokens/s).** Median: 2,010.83 ms;
nearest-rank p95: 2,424.28 ms; range: 1,712.04–2,550.12 ms.

| Case | Prompt tokens | Output tokens | Decode intervals | Mean TPOT | Process → first output |
|---|---:|---:|---:|---:|---:|
| Arithmetic | 12 | 8 | 7 | 2,342.39 ms | 49.82 s |
| Geography | 15 | 8 | 7 | 1,981.62 ms | 45.83 s |
| Chinese | 14 | 17 | 16 | 1,946.11 ms | 44.29 s |

All 33 output tokens match the previously qualified native sequences. These are
three single-request runs, each starting a fresh native process, in the table's
order. The pooled statistic weights all 30 measured intervals equally. Different
run times do not establish a prompt-dependent performance difference.

## Method and limits

The native executable flushes one JSON record after each generated output. A
local Python harness in the same Docker container records receipt times using
`perf_counter_ns`; it performs no model execution. TPOT is
`(last output receipt - first output receipt) / (output tokens - 1)`.

This excludes checkpoint loading, lazy initialization and prompt prefill, while
including the native decode step, greedy selection, top-five diagnostic output,
HCCL token consensus, serialization and pipe delivery. It is wall-clock native
output latency, not kernel-only time or an HTTP serving benchmark. Process-to-first
output includes loading and prefill and must not be interpreted as warm serving
TTFT. EOS is counted as an output token.

Configuration: batch/sequence count one, greedy decode, short prompts, serial
prefill, no full-model ACLgraph. The implementation still uses eight fixed expert
route slots per rank, duplicate device expert banks, per-call temporary
allocation and frequent synchronization. These are known optimization targets;
this timing run does not profile or apportion their individual costs. No previous
host-staged throughput comparison was measured. P95 is a small-sample descriptive
statistic, not a production latency guarantee.

[Summary and binary/program hashes](summary.json) ·
[All token receipt timestamps, intervals and outputs](runs.json) ·
[Residency correctness evidence](../2026-10-16-device-resident/README.md)

## Reproduction

Inside the configured remote Docker workspace, after sourcing CANN and both
qualified omni custom-operator environment scripts:

```sh
export HCCL_OP_EXPANSION_MODE=HOST HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120
python3 scripts/benchmark-native-tpot.py \
  --binary native-build/openpangu_decode \
  --plan resident-dsl-plan --checkpoint /models/92B \
  --cases decoder-resident-qualified/results.json \
  --output NEW_OUTPUT_DIRECTORY
```

The output directory must be new. `--repeats N` repeats the cases, with a fresh
process each time. The harness verifies exact output tokens, rejects coalesced
records that prevent interval measurement, and applies a per-run timeout.
