# Frozen 32K native qualification — 2026-10-18

PASS: 39/39 output tokens through first EOS, six interleaved requests. Native
and qualifier exit 0. Three 32,752-token retrieval prompts (early/middle/late
needle) alternate with arithmetic requests on two reused residents, TP8/EP8.

[Results/resource audits](results.json), [completed batch and output trace](tokens.jsonl),
[fixture provenance](fixture-provenance.json).

- Each long prompt executes **seven 4096-token chunks plus a 4080-token tail**.
- The existing `pooled8k-clean-frozen-plan` is reused without rebinding. Every
  rank retains arena 39,572,690,960 bytes and workspace 159,221,248 bytes.
  Activation host transfers, workspace growth, recorded step allocations and
  internal fences are zero. Decode graphs remain active; prefill is uncaptured.
- Long first-output intervals: 50.591/47.985/47.632 seconds, including offline
  scheduling waits and excluding model load. These are not HTTP TTFT.
- Pure-decode ABI medians: 21.358 ms for one sequence, 29.655 ms for two.
  These include sampling/consensus, exclude mixed-prefill calls and preserve all
  outliers in the trace. They are not sustained-load latency guarantees.

The fixture generator now records input reference and plan hashes, rejects
placeholder answers, scopes output at the compiled EOS, and derives seeded page
permutations from the manifest. Its generated requests, expected tokens, page
bindings and case JSON were byte-identical to this run's existing inputs.
Two portable generator tests pass; it does not produce reference answers.

Reproduce in the established NPU Docker work directory:

```sh
python3 scripts/prepare-sequence-fixture.py \
  --plan pooled8k-clean-frozen-plan --references retrieval-32k-reference.json \
  --peers batched16-preparation-reference.json --peer-case arithmetic \
  --output sequence32k-reproduced
python3 scripts/qualify-sequence-serving.py --frozen-only --batch-trace \
  --binary native-build-head-batch/program_engine_probe \
  --plan pooled8k-clean-frozen-plan --checkpoint /models/92B \
  --requests sequence32k-reproduced/requests.txt \
  --bindings sequence32k-reproduced/page-bindings.txt \
  --reference-output sequence32k-reproduced/reference.jsonl \
  --output-prefix pooled32k
```

Use fresh output paths. This qualifies native 32K inputs, not 32K HTTP or 64K
correctness. HTTP multi-chunk progress was qualified at 16K; the next full HTTP
gate targets 64K after its native gate. APC and sustained-load gates remain open.
