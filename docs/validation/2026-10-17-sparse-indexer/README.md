# Paged sparse-indexer kernel qualification

Date: 2026-10-17. Docker `inferfabric-92b` on 7.156.99.58, logical device 0.
This is a kernel boundary test, **not 64K model-output qualification**. Serving
admission remains capped at 512 until the full decoder is qualified.

The installed `aclnnLightningIndexerEnhance` passed an independent score oracle
with BF16 queries `[2,24,128]`, BF16 paged keys `[1024,64,1,128]`, BF16 head
weights `[2,24]`, and INT32 output `[2,1,2048]`. Two opposing active heads with
unequal weights exercise weighted per-head ReLU, rather than a single dot-product
top-k. Exactly representable two-column inputs give independently computed FP32
scores. Other columns/heads are zero; this does not qualify arbitrary learned
projection error or index-cache quantization.

The test permutes every physical page, checks unique logical indices, causal
bounds, top-k cutoff membership (allowing legitimate score ties) and `-1` padding.
Lengths are 2, 511, 512, 513, 2048, 2049, 4096, 16384 and 65536. Each is checked
eagerly and twice through one captured graph with device length metadata updated
between replays. All **70,653 selected indices** pass; **18 graph replays** pass.
Workspace is **73,458,688 bytes**, allocated before capture and checked for growth.
Only the test uploads synthetic input and reads results for its CPU oracle.
It does not introduce any production host fallback.

| Artifact | SHA-256 |
|---|---|
| Native probe | `620cfd8f43beba634b8aeac6a5a7756e71f16ca261873bb34e7fd7908f711ba2` |
| Training vendor API library | `66a812fb5b7bbc04959d43c1917ce6cbadfbebc32d44e8eff0688a70c93f7e95` |
| Current checkpoint config | `c1b93581ae04c7cc82f4a9557d3179aba7dbbc82f1b848b6dec329f1e7c0487c` |

Build/run inside the existing Docker checkout:

```sh
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/ascend-toolkit/vendors/omni_training_custom_transformer/bin/set_env.bash
cmake -S native -B native-build
cmake --build native-build --target sparse_indexer_probe -j4
native-build/sparse_indexer_probe
```

Result:

```json
{"status":"passed","capacity":65536,"topk":2048,"checked_indices":70653,"workspace_bytes":73458688,"graph_replays":18}
```

Next integration must expose projection, leading-column rotary, paged index
append and selection as physical instructions; keep the selected-index value
alive through its shared consumer; and reserve index state in the same all-rank
transaction as attention and recurrent state. No context limit is raised here.
