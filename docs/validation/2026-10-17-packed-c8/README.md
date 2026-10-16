# Packed C8 qualification — 2026-10-17

OmniInfer-compatible C8 attention-cache execution is implemented and qualified
on the full 46-layer 92B decoder, TP8/EP8, context capacity 512, batch one.
It uses per-128 latent scales, BF16 rotary storage, C8 on the 16 DSA layers only,
and attention consuming packed rows directly. The other 30 layers retain BF16.

[Inspect the physical plan and frozen arena](execution.html).
The model definition is
[92b-tp8-ep8-c8.inferfabric](../../../examples/openpangu/92b-tp8-ep8-c8.inferfabric);
[the format contract](../../packed-mla-cache.md) documents its exact scope.

## Measured results

| Format | Graph TPOT, ms/token | KV cache per rank | Frozen arena per rank, bytes | Median HBM used, GiB |
|---|---:|---:|---:|---:|
| BF16 | 17.7293 | 25.875 MiB | 12,433,409,600 | 31.3709 |
| DSA C8 / SWA BF16 | 17.9239 | 22.000 MiB | 12,523,057,728 | 31.4607 |

C8 is 1.10% slower in this short-context sample. The cache itself saves
3.875 MiB per rank (14.98%), but **total memory does not decrease** here.
The packed-attention vendor API requests 93,970,944 bytes (89.62 MiB) of workspace.
The first C8 attention call creates a workspace slot of that size. Later
preparation grows the shared workspace to 130,916,352 bytes for another operator;
subsequent C8 calls reuse that larger workspace. The pool retains the intermediate
93,970,944-byte allocation, and the frozen arena records both slots. This is one
extra retained slot, not a separate allocation for each C8 layer. The resulting arena grows by 85.50 MiB; measured device-wide HBM grows
by 92 MiB. This is not a full-cache BF16 reconstruction: the selected packed
attention kernel requests the scratch space internally. The numerical probe
records the actual workspace request in `kernel-checks.json`.

The next memory optimization is planning the maximum shared workspace before
arena binding, eliminating obsolete growth slots while preserving capture and
stream ordering. Compatible-size reuse and packed-kernel workspace/tiling
reduction can follow. These are not implemented or counted as savings here.

Both graph runs use the same native binary, eight ranks, the same three prompts,
and 30 inter-output intervals each. TPOT excludes the first output and includes
selection, consensus and output delivery. C8 graph median is 17.8168 ms and sample
p95 is 18.9717 ms. C8 eager execution also passed, averaging 149.3042 ms/token.
These small, short-context samples are not production latency SLOs.

HBM is a device-wide snapshot after generation while the decoder is alive, with
24 snapshots per mode (eight ranks × three cases). It includes checkpoint storage,
the arena, graphs, runtime overhead and any other device use. `measurements.json`
preserves individual snapshots, free-memory baselines, hashes, exact byte counts,
timing statistics and audit totals.

## Output and execution verification

- Arithmetic, geography and Chinese full-model answer checks passed. All **33 C8
  greedy tokens match the immutable BF16 fixture**, including EOS. Full C8 records
  are in `c8-results.json`. Quantization changes logits/top-five probabilities;
  maximum selected-token logit differences are 6, 12 and 10.5 for these cases.
- Bound C8 graph and eager runs reproduce all their C8 token/logit/top-five records
  exactly. The fresh BF16 run reproduces the unchanged BF16 fixture exactly.
  The three final modes contain 99 output records and 72 per-rank audits.
- All 1,704 audited token steps across those final modes report zero allocations
  and zero internal completion fences. Every rank reports zero activation host
  transfers and zero rejected transfer attempts. Graph replay is checked explicitly.
  Each bound execution obtains one arena allocation; checkpoint loading is separate.
- The numerical probe checks 512 cache rows, all four per-128 scales per row,
  and exact BF16 rotary bits. It includes an all-zero row. Scale maximum absolute
  error is 2.58313e-9; latent quantization differs by at most one integer at a
  rounding boundary against the independent double-precision CPU calculation.
- Packed attention is checked against a CPU reference at positions 0, 31, 32 and
  511, in eager execution and two graph replay passes: 36,864 values, maximum
  absolute error 0.00120384. Graph output matches eager exactly.
- Native validation passes all eight initialization bindings and rejects 28 malformed
  contracts. Local compiler/DSL regressions check DSA-only selection, explicit
  precision attributes, one metadata producer and no cache-reconstruction operation.
  The viewer test checks actual rendered kernel names, fields, offsets and strides.

The BF16 fixture SHA256 is
`7a9eafc79473c5843741a3c657d692374d71f8b658f65e00f3ae5fa28a8a6cab`.
`artifacts.sha256` identifies the native binary, execution program, arena and
checkpoint config/tokenizer. The source implementation milestone is `bbd09b1`.
This qualification establishes the packed cache/operator contract and short-context
full-model behavior. It does not establish broad model quality, long-context learned
indexer parity or bitwise parity with an independent full OmniInfer C8 engine run.

## Reproduce

Cross-compile Rust locally as documented in [build.md](../../build.md). Execute all
following commands inside the authorized NPU Docker container with the CANN and
Omni inference/training operator environments sourced. Build the Ascend C cache
library before linking the decoder. Use fresh output directories; do not mutate
or overwrite an already qualified plan bundle.

```bash
cmake --build cache-build -j8
cmake --build native-build --target openpangu_decode packed_cache_probe token_plan_test -j8
native-build/packed_cache_probe
./inferfabric-compiler compile-token examples/openpangu/92b-tp8-ep8-c8.inferfabric \
  /models/92B cache-c8-plan --allow-stale-index-size
native-build/token_plan_test cache-c8-plan
INFERFABRIC_PREPARE_ARENA=1 python3 scripts/qualify-openpangu-decoder.py \
  --binary native-build/openpangu_decode --checkpoint /models/92B \
  --plan cache-c8-plan --output cache-c8-preparation
python3 scripts/benchmark-native-tpot.py --binary native-build/openpangu_decode \
  --plan cache-c8-plan --checkpoint /models/92B --cases cache-c8-preparation/results.json \
  --output cache-c8-graph-preparation --prepare-arena --decode-graph --resident-audit \
  --hccl-base-port 52000
python3 scripts/bind-prepared-arena.py cache-c8-plan \
  cache-c8-graph-preparation/arithmetic-0/prepared cache-c8-bound-plan
python3 scripts/benchmark-native-tpot.py --binary native-build/openpangu_decode \
  --plan cache-c8-bound-plan --checkpoint /models/92B --cases cache-c8-preparation/results.json \
  --output cache-c8-bound-tpot --decode-graph --resident-audit --hccl-base-port 53000
```

Repeat the last command without `--decode-graph`, with a fresh output directory
`cache-c8-bound-eager` and port range. Prepare and bind BF16 with the same binary,
using the BF16 DSL and unchanged BF16 qualification fixture. The recorded baseline
uses `composite-final-plan` → `cache-c8-bf16-bound-plan` and
`cache-c8-bf16-bound-tpot`. The original immutable fixture remains in
`paired-linear-eager/results.json` on the qualification machine.

```bash
python3 scripts/summarize-packed-c8-qualification.py . packed-c8-measurements.json
node scripts/test-token-viewer.cjs docs/validation/2026-10-17-packed-c8/execution.html
```

The checked-in viewer preserves the compiled metadata and bound arena; its template
was updated deterministically with `scripts/refresh-token-viewer.py`. It is generic
and contains no hand-authored layer/kernel inventory.
