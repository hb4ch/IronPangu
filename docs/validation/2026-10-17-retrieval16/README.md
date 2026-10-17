# Full 92B positional retrieval at 16K total context

2026-10-17, TP8/EP8, C8 learned index, two residents, sixteen-token prefill,
shuffled pages and frozen memory. All nine requests retrieve the intended code
and match the independent reference through EOS. **All three full forced-output
comparisons fail after EOS.** This remains experimental evidence, not production
16K or 64K qualification.

## Cases and results

The deterministic early/middle/late cases each contain 16,368 input tokens and
reserve sixteen output tokens. The independent OmniInfer reference uses context
16,384, prefill budget sixteen, C8 (`int8_ds_mla`), TP8/EP, eager execution,
one sequence and no prefix caching. The reference finishes before native timing
starts. [reference.json](reference.json) retains input IDs, expected answers,
needle spans and reference logits; [comparison.json](comparison.json) retains
all 144 native outputs and independently checks decoded answers.

| Needle | Expected answer | Correct native repetitions | TTFT mean, ms | TPOT mean, ms |
|---|---|---:|---:|---:|
| Early | 3379 | 3/3 | 63,574.669 | 20.288 |
| Middle | 9612 | 3/3 | 63,458.646 | 20.283 |
| Late | 3630 | 3/3 | 63,435.536 | 20.272 |

All repetitions are stable. The first token difference is at zero-based output
step six in each case, after EOS at step four. Both the qualifier and comparator
return nonzero. No successful long `results.json` is produced. These timings
exclude loading, tokenization and HTTP; TPOT uses individual step durations.
They use the same pre-rotary-batching runtime as the prior 8K baseline, not the
newer isolated batched-rotary build.

## Preparation and memory

At the larger compiled context, discovery and preparation each reproduce a
529-token post-EOS difference against the unchanged **4K-context** short
reference. The standard 99 tokens still match, but the full 123-token regression
fails. [discovery-comparison.json](discovery-comparison.json) preserves that
distinction; expected tokens were never replaced with native outputs.

Resource preparation is evaluated separately from numerical qualification. After
checking all eight transfer/workspace audits, the maximum workspace was planned
and the arena bound for this explicitly unqualified experiment. This is not a
successful short-regression promotion. The final long run passes all eight rank
resource checks: frozen arena, planned workspace, no growth, no activation host
transfers/rejections and no retained step allocations/internal fences. Each rank
records 9,208 prefill graph replays and 136 decode replays, including warmup.

- Shared workspace: **131,020,288 bytes/rank**, unchanged from 8K.
- Frozen arena: **12,894,879,744 bytes/rank**.
- Process HBM sample: **32,482 MB/rank**, not a peak measurement.

## Reproduction and provenance

Generate inputs with `scripts/make-long-context-cases.py --contexts 16384` and
the checkpoint tokenizer. Input SHA256:
`d7b088c49425503db0e1c5c92d825d835f29ace6dfe0ab18eeb91db8372db5c3`.
Tokenizer SHA256:
`7816b7b6138024e37c508ff82321e11b5ee559289184fa18f7536a3e460d3941`.
Compile [model.inferfabric](model.inferfabric) with `--resident-sequences 2
--prefill-tokens 16 --shared-pages`. Discovery/preparation use the first four
cases from `../2026-10-17-prefill16/reference-budget16.json` and must retain
their failed token comparisons when deriving resource plans.

The long qualifier uses `--permuted-pages --packed-prefill
--experimental-long-context --output-diagnostics`, followed by the comparator
with `--eos 148902 --tokenizer /models/92B/tokenizer.json`.

- Native library SHA256: `cf8151d17fd727528f2823ca1705c6a38d175da6bf7bb7bd5be49f9873ed11c2`.
- Execution binary SHA256: `a2b13c9edd3964ae5be125d9410d12b1fbb716c0101f97ba5d98cf0aa6c5a89a`.
- Remote artifacts: `retrieval16-{source,discovery,workspace,preparation,bound,long}`,
  `retrieval-16k-{input,reference}.json`, `retrieval16-comparison.json` under
  `/data/p00603624/inferfabric/openpangu92` in Docker `inferfabric-92b` on
  `7.156.99.58`. Local raw copies: `.deploy/retrieval16-evidence`.

No concurrent 16K HTTP or pressure soak has been completed. Numerical parity,
32K/64K coverage, faster prefill, latency-aware chunk selection and the remaining
production-serving gates remain open.
