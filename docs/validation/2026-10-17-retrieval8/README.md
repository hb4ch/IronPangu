# Full 92B positional retrieval at 8K total context

2026-10-17, TP8/EP8, C8 learned-index model, sixteen-token prefill, two resident
slots, shuffled pages and frozen execution. All nine requests retrieve the
correct answer and match the independent reference through EOS. **Full forced
continuation parity fails** in the middle and late cases; this is experimental
evidence, not a serving-limit promotion or a 64K claim.

## Cases and oracle

Each request has 8,176 input tokens and sixteen generated tokens. The deterministic
generator places a different access-code record early, in the middle, or late in
routine-maintenance filler. Exact needle spans and expected answers are retained
in [reference.json](reference.json). Reproduce the input with:

```sh
python3 scripts/make-long-context-cases.py --tokenizer /models/92B/tokenizer.json \
  --output retrieval-8k-input.json --contexts 8192
```

Input SHA256: `bd6d74cd87933bffb7c6e92ef7be779b3fedde35393b385e301dfbcebf3183c7`.
Tokenizer SHA256: `7816b7b6138024e37c508ff82321e11b5ee559289184fa18f7536a3e460d3941`.
The offline OmniInfer reference uses TP8/EP, context 8,192, token budget sixteen,
`int8_ds_mla`, eager execution, no prefix caching, one sequence and forced output
past EOS. Native measurements start after the reference process exits.

The [comparison](comparison.json) checks every native repetition against both
the independent token sequence and the expected answer decoded before EOS:

| Needle | Expected answer | Correct native retrievals | Full 16-token parity | TTFT mean, ms | TPOT mean, ms |
|---|---|---:|---|---:|---:|
| Early | 8121 | 3/3 | Pass | 31,731.151 | 20.277 |
| Middle | 3423 | 3/3 | Fail after EOS | 31,596.798 | 20.220 |
| Late | 7889 | 3/3 | Fail after EOS | 31,593.044 | 20.263 |

All native repetitions are stable. The middle/late differences begin at
zero-based output step seven, after the first EOS at step four. The qualifier
and comparator both return nonzero; no successful long `results.json` is
fabricated. All 144 forced tokens remain in the comparison. A correct retrieved
answer does not erase the numerical discrepancy or earlier short-boundary debt.

## Resource and regression checks

Discovery, planned-workspace preparation and frozen short execution each pass
123 expected tokens including the 529-token case, interleaving residents with
shuffled pages. Discovery/preparation overlapped the offline reference on the
other eight devices; their timing is not used as isolated performance evidence.

All eight long-run rank audits pass: frozen arena, planned workspace, no growth,
no activation host transfers/rejected transfers, and no retained step allocation
or internal fence. Each rank executes 4,600 prefill graph replays and 136 decode
replays, including warmup. Workspace is **131,020,288 bytes/rank**, unchanged from
4K. Arena is **12,705,792,000 bytes/rank**. Process HBM is sampled at **32,302 MB
per rank**; this is not a peak-memory measurement. Native TTFT excludes loading,
tokenization and HTTP; TPOT uses each output step's own duration.

## Reproduction

Compile [model.inferfabric](model.inferfabric) with `--resident-sequences 2
--prefill-tokens 16 --shared-pages`, discover/plan the maximum shared workspace,
prepare again, then bind the arena. The unchanged first four cases of
`../2026-10-17-prefill16/reference-budget16.json` are used for preparation.
Run `qualify-program-engine.py` with the committed reference, `--permuted-pages
--packed-prefill --experimental-long-context --output-diagnostics`. Then run
`compare-program-reference.py` with `--eos 148902 --tokenizer
/models/92B/tokenizer.json`; a nonzero comparison exit is expected here.

- Native library SHA256: `cf8151d17fd727528f2823ca1705c6a38d175da6bf7bb7bd5be49f9873ed11c2`.
- Execution binary SHA256: `749191689e0baec8f565694756d0a56d68f773bcb60cf918f36001eec0717fdd`.
- Remote: `retrieval8-{source,discovery,workspace,preparation,bound,qualified,long}`,
  `retrieval-8k-{input,reference}.json` and `retrieval-8k-reference.log` under
  `/data/p00603624/inferfabric/openpangu92` in Docker `inferfabric-92b` on
  `7.156.99.58`. Local raw copies: `.deploy/retrieval8-evidence`.

No concurrent 8K HTTP/pressure soak has been performed. Continue experimental
16K/32K/64K coverage, numerical diagnosis, multiple prepared chunk sizes and
multi-sequence kernels; production readiness remains open.
