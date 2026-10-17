# Frozen 16K native serving qualification — 2026-10-18

PASS: six interleaved requests, 39/39 output tokens correct through first EOS.
The native process and qualifier exited 0. Three 16,368-token retrieval prompts
(early/middle/late needles) alternate with arithmetic prompts on two reused
residents. TP8/EP8, 4K uncaptured prefill, captured sequence decode.

The existing `pooled8k-clean-frozen-plan` is reused without discovery, rebinding
or increasing memory. [Results and resource audits](results.json),
[completed batch and token records](tokens.jsonl).

- Every long prompt executes exactly **4096 + 4096 + 4096 + 4080** tokens.
- All eight ranks retain the 39,572,690,960-byte frozen arena and
  159,221,248-byte workspace. Workspace growth and activation host transfers
  are zero; recorded step allocations/internal fences are zero.
- Long first-output intervals: 25.543/23.948/24.796 seconds. These include
  offline scheduler waits and exclude model loading, not HTTP TTFT.
- Pure-decode ABI medians: single sequence 22.369 ms; two sequences 28.982 ms.
  Calls include native sampling/consensus and exclude mixed prefill calls.
  Small samples and all outliers remain in the trace; this is not a load SLO.

This qualifies native 16K output only. HTTP progress/cancellation at 16K and
32K/64K correctness remain separate gates. The compiled 64K capacity is not a
64K qualification claim, and the experimental context guard remains enabled.

Inside the established NPU Docker environment:

```sh
python3 scripts/qualify-sequence-serving.py --frozen-only --batch-trace \
  --binary native-build-head-batch/program_engine_probe \
  --plan pooled8k-clean-frozen-plan --checkpoint /models/92B \
  --requests sequence16k-fixture/requests.txt \
  --bindings sequence16k-fixture/page-bindings.txt \
  --reference-output sequence16k-fixture/reference.jsonl \
  --output-prefix pooled16k
```

The fixture alternates each `retrieval-16k-reference.json` case with the
arithmetic reference, scopes outputs at EOS 148902 and reuses the 64K-capacity
page permutation fixture. Use a new output prefix when repeating.
