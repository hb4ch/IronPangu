# Packed MLA C8 execution

The `int8_ds_mla` cache format matches OmniInfer's Pangu C8 attention-cache
contract. It is distinct from InferFabric's older `int8` per-token/head format.
The latter also quantizes rotary values and inserts an explicit full-cache BF16
reconstruction; it is not OmniInfer C8.

## Authoring and layer selection

The executable example is [92b-tp8-ep8-c8.inferfabric](../examples/openpangu/92b-tp8-ep8-c8.inferfabric).
Its compile-time module argument selects C8 for checkpoint DSA layers
0, 3, 6, ..., 45. The other 30 decoder layers retain BF16 sliding-window caches.
There is no layer-name or index heuristic in the executor.

```text
cache_norm_rope_append(latent, rotary, sinks,
    weight = w.kv_a_layernorm, eps = 1e-5, theta = 6400000,
    cache_dtype = int8_ds_mla,
    cache_quantization = pertile128,
    cache_scale_dtype = f32,
    cache_rope_dtype = bf16)
```

The compiler rejects another scale precision, quantization scheme, rotary dtype,
or use of this format with the unfused `cache_append` operation. These are
numerical semantics, not optional optimizer choices.

## Storage and physical instructions

Each token occupies 656 bytes in a paged `[blocks, 32, 1, 656]` INT8 allocation:

| Byte offset | Contents | Shape per token |
|---|---|---|
| 0 | Signed INT8 latent values | 512 |
| 512 | BF16 rotated key values | 64 |
| 640 | FP32 symmetric scales, one per 128 latent values | 4 |

The fused vendor writer computes RMS normalization, rotary encoding and tiled
quantization. Quantization uses the vendor's nearest-even integer conversion;
scales derive from each tile's absolute maximum divided by 127. A zero tile has
zero scale, with the vendor's epsilon guarding division. Rotary values are not
quantized to INT8. Learned sinks remain BF16.

The compiler emits three explicit operations:

1. `cache_c8_norm_rope_append`: `aclnnAiInfraKvRmsNormRopeCache` with
   `quant_mode=pertile128`, combined packed cache, and BF16 rotary outputs.
2. `packed_attention_metadata`: one shared device kernel derives causal INT32
   candidates and KV length from the device position. Future indices are -1.
3. `packed_latent_attention`: `aclnnAiInfraKvQuantSparseFlashAttention` consumes
   the packed cache directly, with quant modes 2, scale repository mode 1,
   tile size 128, rotary dimension 64 and learned sinks.

There is no full-cache dequantization instruction, scratch reconstruction, or
persistent BF16 mirror on this path. Vendor attention performs the necessary
operand conversion inside the selected kernel. Both execution and preparation
use the existing buffer leases and frozen-arena contract. There are no new
activation transfers, host fallback, or stream/global fences.

The plan viewer lists the actual selected APIs and exposes the packed allocation,
latent/rotary/scale view shapes, byte offsets, strides and metadata dependencies.
The shipped operator declaration retains Huawei's license header; the runtime
implementation is supplied by the installed Omni custom operator package.

## Qualified scope

The 92B example remains TP8/EP8, one sequence, context capacity 512. All causal
candidates fit below the checkpoint's `index_topk=2048`; this path does not claim
an implementation of long-context learned indexer selection or its separate
quantized index cache. SWA, MTP and long-context policy must not be inferred from
the packed format. C4 compatibility with OmniInfer is not claimed.

Validation commands and measured full-model results are recorded under
[packed C8 validation](validation/2026-10-17-packed-c8/README.md).
