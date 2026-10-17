# INT8 sparse-indexer kernel qualification

Date: 2026-10-17. Result: **kernel boundary passed on NPU**. This is not
end-to-end 92B C8 parity or 64K serving qualification. The experimental decoder
still uses the BF16 indexer until explicit quantization is integrated.

`native/probes/quant_sparse_indexer_probe.cpp` invokes the inference vendor's
`aclnnAiInfraQuantLightningIndexer` with INT8 queries/keys, FP16 query/key scales
and FP16 head weights, matching the reference interface. It uses 24 heads of
width 128, 64-token shuffled pages, top-2048 selection and causal TND/PA_BSND
layouts. The independent CPU oracle computes weighted ReLU scores from exact
integer inputs and power-of-two scales. Opposing heads have unequal weights;
per-key scales vary, so ignoring scales or confusing physical and logical rows
changes the expected selected set. Ties at the cutoff are accepted; duplicates,
future/out-of-range indices and incorrect padding are rejected.

Lengths tested: 2, 511, 512, 513, 2047, 2048, 2049, 4096, 8192, 16384, 32768,
65536. Each length runs eagerly and twice through a captured graph at fixed
addresses with changing device length metadata. All **107,508 selected indices**
passed, including **24 ACLGraph replays**. Reported workspace: **70,279,680 bytes**.
Output readback and stream synchronization are confined to this qualification
probe. This does not introduce activation transfers into the serving executor.

Built and executed inside `inferfabric-92b` on `7.156.99.58`, Ascend 910 device 0;
no competing NPU processes were present. Build uses `-Wall -Wextra -Werror`.

```sh
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/ascend-toolkit/vendors/omni_custom_transformer/bin/set_env.bash
cmake -S native -B native-build
cmake --build native-build --target quant_sparse_indexer_probe -j4
./native-build/quant_sparse_indexer_probe
```

```json
{"status":"passed","capacity":65536,"topk":2048,"checked_indices":107508,"workspace_bytes":70279680,"graph_replays":24}
```

SHA-256:

- Probe binary: `cbeed7940dec8839b71d1551396cf6770781a0a99cc357651082597114f89eb6`
- Inference `libcust_opapi.so`: `aa73d7fd12f51d172c171ff4e7a36c05dc4c1c3183f69f6e0cdadecd06642d3b`

Still required: explicit per-head/per-key dynamic quantization and FP32-to-FP16
scale conversion in the physical plan, packed index-cache ownership, frozen
arena requalification, full decoder parity and long-context serving tests. This
fixture supplies already quantized inputs and therefore does not validate the
dynamic quantizer, model projections, cache append or full decoder output.
