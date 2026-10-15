# TP8/EP8 placement and native pipeline qualification

## Checkpoint comparison status

The requested path `/mnt/sfs_turbo/bucket-910c-6055/s00562949/models/openPangu_2.0_Flash_INT8/global_step_26-iterative-int8-0626` was absent on `7.156.99.58` on 2026-10-15. It was checked inside the existing `/mnt`-mounted container and through an exact read-only Docker host bind mount, which Docker rejected because the source did not exist. **No checkpoint equality claim is possible.** The available checkpoint used here remains `/mnt/sdb_data/models/92B`, mounted read-only at `/models/92B`.

When the candidate becomes accessible, compare config/tokenizer/generation metadata and tensor names/shapes/dtypes first, then hash all tensor payloads to establish weight identity. Matching model dimensions, names, or aggregate byte counts alone does not prove identical checkpoints. Neither checkpoint should be modified to make an audit pass.

## Placement contract

[DSL](../../../examples/openpangu/92b-tp8-ep8.inferfabric): `contract=openpangu_v2_tp_ep_v1`, TP8 and EP8 share ranks/devices 0–7; pipeline parallelism is 1. The parser rejects duplicate/unknown fields, mismatched TP/EP, repeated device IDs and invalid serving budgets. The existing Qwen DSL and serving runtime are unchanged.

The compiler emits exact shard filenames, byte offsets, local tensor shapes, dtypes, transforms and placement roles. Routed experts remain whole on one EP owner; their matrices are not additionally TP-sharded. Quantized weights and their per-output scales use matching row slices. Non-expert output-row TP shards reconstruct full canonical activations after every projection. Embeddings, mHC, router, indexer, low-rank input projections and state remain replicated. This is a correctness-first communication-heavy layout, not a performance claim.

| Rank/device | Owned experts, inclusive | Stored weight bytes | Prepared weight bytes |
|---|---|---:|---:|
| 0 | 0–31 | 12,885,336,434 | 12,945,669,860 |
| 1 | 32–63 | 12,885,336,434 | 12,945,669,860 |
| 2 | 64–95 | 12,885,336,434 | 12,945,669,860 |
| 3 | 96–127 | 12,885,336,434 | 12,945,669,860 |
| 4 | 128–159 | 12,885,336,434 | 12,945,669,860 |
| 5 | 160–191 | 12,885,336,434 | 12,945,669,860 |
| 6 | 192–223 | 12,885,336,434 | 12,945,669,860 |
| 7 | 224–255 | 12,885,336,434 | 12,945,669,860 |

Each rank has 9,956 weight slices across the 46 main layers and model boundary operations. Prepared bytes account for declared FP32 casts; they exclude load staging, kernel packing, KV/index/MoME state, activations, workspace, communication and graph memory. MTP is excluded. [Plan summary](plan-summary.json) preserves checkpoint identity, budgets, rank totals and the ordered collective schedule. The full generated plan is stored remotely at `/data/p00603624/inferfabric/openpangu92/tp8-ep8.json`.

The planned MoE schedule uses replicated input tokens and identical routes on each rank, owner-local experts, FP32 weighted output accumulation and all-reduce. Shared experts run through the TP path and are added once after the routed reduction. All-to-all dispatch, token partitioning and TP×EP Cartesian meshes are not implemented here.

## Executed native pipeline

The test runs the following connected graph on eight real NPUs:

1. Real layer-0 dense down projection `[2560,9216]`, output rows divided across TP8.
2. HCCL all-gather followed by device copies from rank-major to canonical `[batch,2560]` layout.
3. Real layer-2 expert gate/up projections, SiLU, elementwise multiply and down projection on each rank. Tested expert IDs are 0, 32, 64, 96, 128, 160, 192 and 224, one representative from each ownership interval.
4. BF16-to-FP32 conversion, explicit per-token routing coefficients and FP32 HCCL all-reduce.

These layers are selected to test compatible tensor shapes and distributed primitives; this connection is **not the model's decoder graph**. The probe does not implement learned router selection, the shared expert, all 256 experts, attention/mHC/MoME or full-model execution. The plan is marked `executable=false`; its schedule is not yet consumed by a production runtime. The probe explicitly targets devices 0–7, rather than accepting arbitrary DSL device permutations.

At batches 1 and 4, three routing cases cover equal contributions from all eight owners, one selected owner per token, and all-zero routing coefficients with **nonzero activations**. Ranks with zero routing coefficients still run the fixed expert computation and participate in both collectives. This verifies masking/combination, not skipped-compute dispatch efficiency.

Both eager execution and three rounds of graph replay passed. Replays change input and routing buffers without recapture. TP reconstruction is checked against each peer's actual shard output. Every quantized projection is checked against an independent CPU integer dot product using device-produced quantized activations/scales. SwiGLU and routing multiplication are checked separately; all-reduce is checked against a host sum of all owner contributions. Captured canonical activations and reduced outputs must match eager results exactly.

The DynamicQuant scale reference uses two FP32 epsilons relative to `max(1, expected_scale)`. An initial fixed absolute tolerance rejected 145.133850 versus 145.133865 in the chained batch-four test (one FP32 rounding step); the kernel was not changed. The inherited BF16 projection bound is `0.0001 + 0.015625 * abs(expected)`; SwiGLU uses `0.0001 + 0.016 * abs(expected)` and collective sums use `0.0001 + 0.00002 * abs(expected)`. These are correctness qualification tolerances, not measured full-model accuracy.

No torch/Python execution is involved. Rust was cross-compiled locally; native compilation and NPU runs occurred inside `inferfabric-92b`, using the [previous milestone's image and driver setup](../2026-10-15-openpangu92/README.md). HCCL HOST expansion is required. The installed HCCL requires `HCCL_CONNECT_TIMEOUT >= 120`; root creation also requires a current ACL device context. The test initializes a bootstrap context before creating the HCCL root.

## Reproduce

Build locally as described in [the build guide](../../build.md), and stage the repository plus `.deploy/inferfabric-npu` and `.deploy/export_openpangu_linear` in the container workspace. With devices 0–7 assigned and free, run **inside Docker**:

```bash
bash scripts/qualify-openpangu-tp8.sh /models/92B qualification-tp8
```

This exports the selected real projection payloads into the output directory, builds the native target and runs batches 1 and 4 with a 240-second timeout per test. It explicitly allows only the known stale index-total discrepancy. Payload fixtures stay outside Git. Checkpoint paths and device access must be verified before executing this hardware script.

The plan command alone does not occupy NPUs:

```bash
./inferfabric-npu plan-openpangu examples/openpangu/92b-tp8-ep8.inferfabric /models/92B tp8-ep8.json --allow-stale-index-size
```

## Validation evidence

- [Batch 1](reports/tp8-ep8-batch1.json): 61,440 reduced values checked; maximum intermediate linear absolute error 0.00195312.
- [Batch 4](reports/tp8-ep8-batch4.json): 245,760 reduced values checked; maximum intermediate linear absolute error 1 on larger synthetic activations, within the documented relative BF16 bound.
- [Single-device regression after refactoring](reports/refactored-linear-batch4.json) passed.
- [Native artifact hashes](reports/tp8-artifacts.sha256) and 25 projection metadata/hash records in `fixtures/` identify the exact inputs without including model payloads.
- 68 local workspace tests, formatting, Clippy and local ARM64 cross-compilation passed. Parser and placement tests cover rank uniqueness, TP/EP mismatch, capacity bounds, contiguous shard coverage, scale slicing and single-owner expert placement.

## Remaining work for 92B serving

Integrate the plan's rank slices and communicator schedule into the model loader/executor; implement learned routing and all owner-local experts, shared experts, attention, mHC and MoME; compare full decoder layers and complete-model logits; qualify startup memory and failure handling; then connect the existing vLLM 0.25.1 Rust frontend. This milestone establishes placement and a real eight-rank component pipeline, not a 92B serving endpoint.
