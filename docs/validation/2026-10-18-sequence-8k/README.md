# Native two-chunk prefill qualification — 2026-10-18

PASS: discovery, fixed-workspace and frozen-arena runs each produced all 39
reference tokens across six interleaved requests. Each native process and the
three-stage driver exited 0. References are compared through first EOS.

The three retrieval prompts contain 8,176 tokens, with the needle early, middle
and late. Each alternates with a 12-token arithmetic prompt; both native
residents are reused. The budget is 4,096 total tokens per forward, with
uncaptured bulk prefill and sequence decode graphs. This uses the current
2,862-instruction bulk lowering, TP8/EP8, shared permuted pages and a compiled
65,536-token capacity. **Only 8K inputs are qualified here**, not 64K output
correctness or the HTTP scheduler at 8K.

[Stage evidence](results.json) includes process outcomes, first-token timings,
all rank resource summaries, graph counters and fixture hash.
[Frozen output records](tokens.jsonl) preserve exact token IDs.

- All eight ranks: zero activation host transfers.
- Frozen arena: 39,572,690,960 bytes per rank; workspace: 159,221,248 bytes.
- Each rank replayed `sequence_decode_1` eight times and `sequence_decode_2`
  twelve times, including preparation; no prefill graph exists.
- Frozen long-request first-token intervals: 14.106, 13.761 and 13.236 seconds.
  They include waiting within the offline resident-order driver, exclude model
  loading, and are not HTTP TTFT. One short request waited 13.834 seconds behind
  the first long request; this driver is not a fairness benchmark. Mixed output
  intervals must not be represented as pure decode TPOT.

Reproduction on 7.156.99.58 inside `inferfabric-92b`, work directory
`/data/p00603624/inferfabric/openpangu92`, with CANN/vendor/HCCL environment:

```sh
/data/p00603624/inferfabric/inferfabric-compiler-bulk-prefill compile-token \
  family64k.inferfabric /models/92B sequence64k-source \
  --allow-stale-index-size --resident-sequences 2 --prefill-tokens 4096 --shared-pages
python3 scripts/qualify-sequence-serving.py \
  --binary native-build-head-batch/program_engine_probe \
  --plan sequence64k-source --checkpoint /models/92B \
  --requests sequence8k-fixture/requests.txt \
  --bindings sequence8k-fixture/page-bindings.txt \
  --reference-output sequence8k-fixture/reference.jsonl \
  --output-prefix sequence8k-serving
```

Fixtures alternate each `retrieval-8k-reference.json` case with the arithmetic
case from `batched16-preparation-reference.json`. Output counts stop at EOS
148902. Page permutations use seed 1729 and manifest capacities, as in the
existing native qualifier. Output directories must be fresh.
