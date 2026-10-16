# Prepared execution qualification — 2026-10-16

**Mean TPOT: 1,281.69 ms/token (0.780 tokens/s)** across six complete native
92B TP8/EP8 requests and 60 output intervals. Compared with the preceding
[resident baseline](../2026-10-16-resident-tpot/README.md), 2,046.86 ms/token,
this is **37.38% lower latency, or 1.60× throughput**. This is a same-machine
historical comparison, not a randomized paired A/B experiment.

All **66/66 output records**, including selected logits and top-five diagnostics,
match the qualified resident reference exactly. All **48 case/rank audits** pass:
zero activation host transfers, zero rejected copies, zero device-pool allocations
at generated-token positions, zero completion boundaries inside decoder steps,
and exactly one owning-stream output completion per emitted token. Output D2H
is bounded to 64 bytes/rank/token plus 80 bytes on rank zero. The separate final
profile adds another complete 17-token exact-output run with all eight audits.

## End-to-end timing

| Case | Repeat | Output tokens | Mean TPOT (ms) | Process → first output (s) |
|---|---:|---:|---:|---:|
| Arithmetic | 0 | 8 | 1,237.00 | 30.69 |
| Geography | 0 | 8 | 1,189.84 | 33.47 |
| Chinese | 0 | 17 | 1,484.23 | 33.04 |
| Arithmetic | 1 | 8 | 1,523.16 | 28.77 |
| Geography | 1 | 8 | 1,420.73 | 34.87 |
| Chinese | 1 | 17 | 972.42 | 30.89 |

Pooled median: **1,306.29 ms**; empirical nearest-rank p95: **1,558.27 ms**;
range: **828.95–1,598.60 ms**. Pooling weights output intervals equally. Run-to-run
variation is material; neither the best run nor this small sample establishes a
production SLO. Process-to-first-output includes checkpoint loading, preparation
and serial prefill, so it is not warm serving TTFT.

Configuration matches the prior measurement: Docker `inferfabric-92b` on
`7.156.99.58`, eight NPUs, `/models/92B`, batch one, short prompts, greedy output,
512-token context capacity, HCCL HOST expansion, no full-model ACLgraph and no
HTTP/frontend overhead. The NPU process table was empty before work started.

## Mechanism and remaining cost

The [implementation contract](../../prepared-device-execution.md) describes
single-stream leases, pinned staging and retained invocation metadata.
The sampled decoder now performs **zero aclrtMalloc, aclrtFree or
aclrtSynchronizeStream calls**. The previous profile recorded 28,222 allocations,
28,222 frees and 15,852 stream synchronizations per rank/token.

The full audit confirms zero pool allocation at every measured generated-token
position, rather than relying only on the two profiled steps. In the Chinese
rank-zero run, the context retains 3,488 device blocks totaling 11,522,298,208
bytes, with 917,143 lease reuses and 7,693 tensor/view descriptors. The pool includes
persistent expert banks and state; it is **not total HBM use or activation-only
memory**. Model checkpoint storage and vendor-internal allocations are separate.

The final profile's eight-rank mean host submission time is 1,473.28 ms/step.
It records 277 HcclAllGather calls taking 1,136.54 ms and 44 HcclAllReduce calls
taking 75.14 ms per rank/token. Asynchronous copies take 57.05 ms of host submission
time. Removing explicit fences can move waiting into blocking vendor/HCCL calls;
these API spans do not prove network bandwidth saturation. Collective submission
is still the next major target. These profiled host submission timings exclude
the output completion wait and are not the end-to-end TPOT above.

## Reproduction and evidence

Inside the configured remote Docker workspace, source CANN and both omni custom
operator environment scripts, then:

```sh
export HCCL_OP_EXPANSION_MODE=HOST HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120
cmake -S native -B native-build -DCMAKE_BUILD_TYPE=Release
cmake --build native-build --target openpangu_decode prepared_execution_probe resident_tensor_probe -j4
native-build/prepared_execution_probe
native-build/resident_tensor_probe
python3 scripts/benchmark-native-tpot.py \
  --binary native-build/openpangu_decode --plan resident-dsl-plan \
  --checkpoint /models/92B --cases decoder-resident-qualified/results.json \
  --output NEW_TIMING_DIRECTORY --resident-audit --repeats 2
python3 scripts/profile-native-decode.py \
  --binary native-build/openpangu_decode --plan resident-dsl-plan \
  --checkpoint /models/92B --cases decoder-resident-qualified/results.json \
  --case chinese --output NEW_PROFILE_DIRECTORY
```

- `prepared-tpot/summary.json`, `runs.json`: binary/program hashes, every output
  record, every receipt timestamp and interval. Per-run subdirectories contain
  all rank transfer and prepared-execution audits.
- `prepared-final-profile/`: final binary/fixture provenance, rank host timers,
  output records and audits.
- `prepared-probe.jsonl`: 128 queued iterations, 127 cached-invocation replays,
  temporary/staging lifetime checks and negative ownership/transfer contracts.
- `resident-probe-after-prepared.txt`: existing ownership, BF16 rounding and
  transfer rejection test result.
- `analysis.json`: post-run checks and aggregate profile/comparison calculations.
- `physical.json`: fresh compiler output describing reusable-pool preparation,
  repeatability policy and request-output completion. The updated
  [hierarchical viewer](../2026-10-16-reusable-dsl/execution.html) embeds this plan.
  Its internal launch DAG/arena bounds remain explicitly unresolved.

Final native binary SHA-256:
`33f6d5818ae51598d1d9816c5f2850d5071a2b1336729ede29867cee952d5639`.
Execution program SHA-256 remains:
`46fa67de15080a6ccf370ea852f2d7a6744b532603bdf73ded7f59c34a082513`.
The newly compiled `prepared-dsl-plan` has the same execution binary as the tested
`resident-dsl-plan`; only physical annotations/backend source identity change.
Rust was tested and cross-built locally in WSL: all 42 compiler tests, formatting
and Clippy with warnings denied passed. C++ builds and both NPU probes passed in
Docker. Python syntax and saved-evidence checks passed, including exact output-byte
bounds added to the benchmark harness after the timing run.

This qualifies the current fixed-shape, single-sequence decoder. Prepared pooling
is runtime preparation, not execution of a frozen physical arena. Broad numerical
coverage, long-context behavior, concurrent requests, full-model capture and
production serving remain separate qualifications.
