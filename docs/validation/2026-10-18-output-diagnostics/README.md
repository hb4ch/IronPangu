# Output-only batch diagnostics — 2026-10-18

**The long-output correctness gate remains open.** This run inspects only eight
outputs after 65,408 prompt tokens, in explicit arena preparation mode. It does
not reproduce the frozen run's first divergence at output index 7: its first
difference is at index **4**, native token 844 versus reference token 198.
At this equal-history position, native favors 844 by about 1 logprob, while the
independent reference favors 198 by about 1. Later score comparisons are invalid
because their autoregressive histories have already diverged.

In particular, the native tie between tokens 17 and 220 at index 7 is **not** an
explanation of the frozen run's failure. It occurs after a different history.
An initial manual interpretation missed the earlier difference; the automated
comparison retained here corrects it. Do not claim an eight-token parity pass.

## Implementation and scope

`INFERFABRIC_OUTPUT_DIAGNOSTICS` now supports the batch execution path. Normal
batch argmax runs first, then the existing device TopK/log-softmax output path
checks that its argmax agrees. Only requested output scores/IDs return to host;
all eight transfer audits report zero activation host transfers and rejections.
The diagnostic adds output kernels and owning-stream completions. It is opt-in,
requires preparation mode for uncaptured-prefill programs, and is not a serving
performance configuration. No global fence or implicit host fallback is added.

The source plan, allocation mode, new binary and instrumentation differ from
the previously frozen bundle. This experiment cannot isolate which difference
causes the changed result. Neither the default serving bundle nor its frontend
was promoted. A future attribution experiment must hold these factors fixed;
repeated full-model runs with multiple changed factors are not useful evidence.

`analyze-token-divergence.py` verifies identical prompt IDs, contiguous output
steps, corresponding absolute diagnostic positions and complete requested output
count. It compares scores only at the first difference, records partial output
scope and hashes, and rejects placeholder/intervention references. The ordinary
program/reference comparator also rejects intervention references. Three focused
analysis tests and six comparator tests pass locally in WSL.

## Reproduction and retained evidence

The retained `run.sh` ran in Docker `inferfabric-92b` on 7.156.99.58, with TP=8,
EP, C8, three resident capacity, 4096-token uncaptured prefill and decode graph
configuration enabled. This request uses one resident and eight output tokens.
The Docker C++ build and native process completed successfully; this is not a
full-output or frozen-resource qualification. `evidence.json` pins the binary,
source, plan and independent oracle and includes every rank's transfer audit.
`tokens.jsonl` and `comparison.json` retain the actual outputs and comparison.
The full request and reference remain in the remote workspace recorded there.

An earlier attempt correctly rejected an old binary's workspace fingerprint.
Another frozen legacy diagnostic attempt exhausted its declared arena. Neither
failure was bypassed; the successful diagnostic uses the unbound source plan
and explicit preparation. Resource/timing observations from this mode must not
be reported as frozen serving memory or TPOT.

Recompute the comparison with the original full request/reference:

```sh
python3 scripts/analyze-token-divergence.py \
  --reference extended-answer64-reference.json \
  --tokens tokens.jsonl --requests requests.txt \
  --case extended-answer-65536-middle --output new-comparison.json
```
