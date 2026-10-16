# Executable INT8 / INT4 KV cache — 2026-10-16

Both formats execute the full 46-layer OpenPangu 92B decoder with TP8/EP8,
device-resident activations, and a frozen arena. Quantization and reconstruction
are explicit physical instructions. BF16 remains the default.

| Cache format | Graph TPOT (ms/token) | Persistent KV + scales per rank | Frozen arena per rank (bytes) | Median device HBM used after generation |
|---|---:|---:|---:|---:|
| BF16 | 17.7334 | 25.875 MiB | 12,433,409,600 | 31.3701 GiB |
| INT8 | 21.9857 | 13.027 MiB | 12,420,527,680 | 31.3584 GiB |
| INT4 | 21.9492 | 6.559 MiB | 12,413,744,704 | 31.3525 GiB |

**This implementation saves cache memory and costs decode latency.** It uses
packed persistent caches and an explicit shared BF16 attention scratch buffer,
not fused attention on packed operands. One 589,824-byte scratch slot per rank
is reused by all 46 layers; no BF16 cache mirror persists per layer. INT8/INT4
reduce KV payload-plus-scale storage by 49.65%/74.65%. Total HBM savings are small
at the currently qualified 512-token capacity because weights/prepared parameters
dominate. Learned sinks remain BF16 constants and are excluded from the KV column.

Measurements use the same native binary, TP8/EP8, batch one, context capacity 512,
and three short prompts. TPOT excludes the first output and includes selection,
consensus, and output delivery. Graph samples contain 30 BF16 intervals, 30 INT8
intervals, and 43 INT4 intervals. INT4 produces a longer arithmetic answer, so
its pooled average is not an identical-token-workload comparison. These samples
are not production latency SLOs. Ordinary execution also passed, at 149.72 ms/token
for INT8 and 137.26 ms/token for INT4.

HBM numbers are the median of 24 device-wide snapshots per format (8 ranks ×
3 cases), taken after generation while the decoder is alive. They include
checkpoint storage, arena, graphs, runtime overhead, and any other device use.
`cache-measurements.json` preserves all samples, baseline/free-memory readings,
exact arena sizes, hashes, timing statistics, and audit totals.

## Numerical and execution checks

- Device storage kernels match an independent CPU reference for INT8/INT4
  quantization, signed packing, scales and BF16 reconstruction. Row boundaries
  0/1/31/511 and graph replay with positions 0/1/511/0 passed, including zero rows
  and clearing future output rows. See `cache-kernel-checks.jsonl`.
- All three full-model answer checks passed in both formats. INT8 retains all
  **33 BF16 greedy tokens**, although logits/top-five probabilities change.
  INT4 retains the geography and Chinese responses; arithmetic still answers
  `1 + 1 = 2`, then adds an emoji and a follow-up sentence. See
  `int8-results.json`, `int4-results.json`, and `bf16-comparison.json`.
- **191 complete output records** passed final bound execution checks: 33 BF16
  graph records match the unchanged historical fixture; 66 INT8 and 92 INT4
  graph/eager records match each format's saved qualification results exactly.
  Format-specific fixtures test repeatability, not independent model quality.
- **120 rank audits** confirm one arena reservation per rank, zero allocations
  and internal completion fences in every recorded token step, and zero host
  activation transfers. No global fence or host activation fallback was added.
- **70 local compiler/DSL tests** passed across the baseline suite and added
  page-padding test. Rust was built/cross-compiled locally. Native validation
  passed all eight rank bindings and rejected **28** malformed contracts for
  each integer-cache plan.
- The deterministic viewer data agrees with physical operations, storage types,
  scales and scratch dependencies; all ranks use scratch slot 193 across layers.
  Both embedded viewer scripts pass Node syntax validation. Browser visual QA
  was unavailable because the browser-control runtime failed to initialize.

These short prompts and numerical storage checks do not establish broad model
quality, long-context quality, or production readiness of lossy cache precision.
INT4's logit changes are material even when its greedy answer is unchanged.

## Inspect the plans

- [INT8 physical plan](int8/execution.html) and [exact DSL source](int8/model.inferfabric)
- [INT4 physical plan](int4/execution.html) and [exact DSL source](int4/model.inferfabric)

Each integer-cache graph contains 2,753 physical instructions, including 46 each
of `cache_normalize`, `cache_quantize_append`, and `cache_dequantize`. Attention
consumes the explicit scratch value and cache state. Use the **Quantization /
cache precision** filter; instruction details show APIs, precision, dependencies,
structured fields, and measured arena offsets. The compiler emits the viewer;
`bind-prepared-arena.py` adds measured bindings deterministically.

Storage uses signed symmetric ranges [-127,127] and [-7,7], nearest rounding with
ties away from zero, one FP32 scale per combined latent/rotary token/head row,
and scale 1 for a zero row. INT4 stores the earlier element in the low nibble.
Reconstruction rounds to BF16. This scalar Ascend C implementation establishes
correct compressed execution; vectorized storage kernels and fused attention
on packed caches remain performance work.

## Build and reproduce

Native builds and model execution run inside Docker `inferfabric-92b` on
`7.156.99.58`, under `/data/p00603624/inferfabric/openpangu92`, with checkpoint
`/models/92B`. Rust compilation remains local; see [build instructions](../../build.md).

Inside Docker, with CANN and both Omni operator environments loaded:

```sh
cmake -S native/kernels -B cache-build -DSOC_VERSION=Ascend910_9391 -DCMAKE_BUILD_TYPE=Release
cmake --build cache-build -j8
cmake -S native -B native-build
cmake --build native-build --target openpangu_decode cache_quantization_probe token_plan_test -j8
native-build/cache_quantization_probe
export HCCL_OP_EXPANSION_MODE=HOST HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120
```

Compile each saved DSL using the locally cross-compiled compiler, then prepare:

```sh
./inferfabric-compiler compile-token model.inferfabric /models/92B cache-plan --allow-stale-index-size
native-build/token_plan_test cache-plan
INFERFABRIC_PREPARE_ARENA=1 INFERFABRIC_DECODE_GRAPH=1 \
  python3 scripts/qualify-openpangu-decoder.py --checkpoint /models/92B \
  --plan cache-plan --binary native-build/openpangu_decode --output preparation
python3 scripts/bind-prepared-arena.py cache-plan preparation/arithmetic-prepared bound-plan
python3 scripts/benchmark-native-tpot.py --binary native-build/openpangu_decode \
  --plan bound-plan --checkpoint /models/92B --cases preparation/results.json \
  --output graph-tpot --resident-audit --decode-graph --hccl-base-port 45000
```

Use fresh output directories and non-overlapping HCCL port ranges. Omit
`--decode-graph` for the eager check. The same-format preparation fixture checks
replay consistency; retain the unchanged BF16 fixture for the BF16 regression
and compare quality separately. Normal execution requires a bound arena; a
binary rebuild requires preparation again.

The final deployed bundles are `cache-bf16-bound-plan`, `cache-int8-bound-plan`,
and `cache-int4-bound-plan`. `cache-artifacts.sha256` pins the decoder, statically
linked device-kernel library, checkpoint configuration and tokenizer.
To repeat cross-rank evidence aggregation in that workspace:

```sh
python3 scripts/summarize-cache-qualification.py . cache-measurements.json
```

To repeat local plan-data and JavaScript checks from the repository root:

```sh
python3 docs/validation/2026-10-16-integer-cache/verify_plans.py
node --check .deploy/int8-viewer.js
node --check .deploy/int4-viewer.js
```

Milestones: `2c3796a` adds hardware-tested storage kernels; `0a95e6f` integrates
physical lowering, runtime execution and precision inspection. The final evidence
commit adds frozen-arena timings, memory audits, BF16 regression and eager checks.
