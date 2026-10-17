# Learned-index precision and physical execution

The `index_cache_append` operator accepts `cache_dtype = "bf16"` (the default)
or `cache_dtype = "int8"`. The experimental 92B learned-index DSL chooses INT8
explicitly. This controls index selection precision, independently of the
latent/rotary attention cache's `cache_dtype = int8_ds_mla`.

```text
let cache = index_cache_append(key, width = 128, page_tokens = 64,
                               cache_dtype = "int8");
let selected = sparse_index(query, head_weights, cache, heads = 24,
                            head_dim = 128, topk = 2048,
                            scoring = "weighted_relu");
```

Logical query and key operands are BF16 after their declared rotary operations.
The compiler expands the INT8 contract into the following physical DAG:

1. Key `index_quantize`: `aclnnDynamicQuant`, one independent 128-element row,
   producing INT8 values and one FP32 scale per token.
2. Key `quant_scales_f16`: `aclnnCast` of scales to FP16; the INT8 values retain
   their existing device allocation.
3. `index_cache_append`: two `aclnnInplaceIndexCopy` calls append values and scales
   using the same physical row. No quantization or casts happen inside append.
4. Query `index_quantize`: `aclnnDynamicQuant` treats each of the 24 heads as an
   independent 128-element row, producing 24 FP32 scales per token.
5. Query `quant_scales_f16` and head-weight `cast(dtype=f16)` are separate planned
   operations.
6. `sparse_index`: `aclnnAiInfraQuantLightningIndexer` consumes those INT8/FP16
   operands directly and returns 2048 INT32 logical token indices. Shared
   attention consumers retain this device value; they do not recompute it.

For a packed chunk of `T` tokens, the quantized selector makes one TND call with
queries `[T,24,128]`, scales/head weights `[T,24]` and output `[T,1,2048]`.
All chunk keys are appended on the owning stream first. Causal alignment uses
query length `T` and the final chunk KV length, so row `r` cannot select a later
chunk row. A persistent device array containing `1..max_query_tokens` supplies
the selected query-count scalar; KV length remains a device metadata dependency.
Neither length requires activation readback. Consumers alias the output rows;
there are no index assembly copies. The physical recipe and viewer expose one
call per chunk, zero assembly copies and the count-array shape. The BF16 path
continues to use ordered per-query calls and explicit row assembly.

Changing this backend requires fresh workspace discovery and arena preparation;
the frozen bindings validate the backend fingerprint. Reusing an older arena
with different descriptor or allocation behavior is not allowed.

Index keys use `[physical_pages,64,1,128]` INT8 storage and scales use
`[physical_pages,64,1]` FP16 storage. Both arrays share one page-ownership component
and block table. Their combined payload is 130 bytes per token, versus 256 bytes
for BF16 index keys. Reset clears only owned pages in both arrays before release.
The serving manifest includes both arrays in its page byte budget.

The physical viewer shows the actual quantizer, casts, append calls, indexer,
intermediate shapes/dtypes, cache scale array and control bindings. These are
prepared invocations with resident buffers and a frozen arena; no activation
host conversion is permitted. FP16 scale conversion must not be removed as a
redundant cast: it is part of the reference's numerical contract.

The BF16 alternative retains `aclnnLightningIndexerEnhance` and emits none of the
INT8 quantization steps. INT4 index keys are rejected; attention-cache INT4 support
does not imply that the vendor indexer supports INT4 operands.

The [kernel boundary qualification](validation/2026-10-17-quant-sparse-indexer/README.md)
passes at 64K keys. Full decoder correctness and serving context qualification
remain separate gates; this interface does not raise the current context limit.
