# Integer-cache execution qualification

## Milestone 1: device storage kernels

Validated on 2026-10-16 inside `inferfabric-92b` on 7.156.99.58.
`cache_quantization_probe` checks INT8 and packed INT4 against an independent CPU
reference, row addresses 0/1/31/511, zero rows, and graph replay with changed
device position. Both formats passed. This is kernel evidence, not full-model
quality or TPOT evidence.

Storage contract: symmetric per-token-head FP32 scale, signed range [-127,127]
or [-7,7], nearest rounding with ties away from zero. INT4 packs the earlier
element in the low nibble. A zero row uses scale 1. Dequantization rounds to
BF16 and clears unused future rows in its output scratch. All operations run
on device, including replay position selection.

At 512 tokens and width 576, persistent payload plus scales is 296,960 bytes
(INT8) or 149,504 bytes (INT4), versus 589,824 bytes BF16. Attention scratch
is separate and must be included when reporting total arena memory.

Build inside the device container, before configuring native targets:

```sh
source /usr/local/Ascend/cann/set_env.sh
cmake -S native/kernels -B cache-build -DSOC_VERSION=Ascend910_9391 -DCMAKE_BUILD_TYPE=Release
cmake --build cache-build -j8
cmake -S native -B native-build
cmake --build native-build --target cache_quantization_probe -j8
native-build/cache_quantization_probe
```

The scalar storage implementation establishes a correctness baseline. It does
not yet claim competitive bandwidth or fused quantized attention.

## Milestone 2: full decoder integration

Both formats completed all 46 layers with TP8/EP8 and ACL graph replay.
`int8-results.json` and `int4-results.json` retain the complete output records.
The arithmetic, France-capital, and Chinese Beijing-description checks passed.
These are short-prompt semantic checks plus the independently checked numerical
storage kernels, not a general quality benchmark or bit-identical BF16 claim.
INT4 changes wording and continuation length. Subsequent exact-record comparisons
use each format's own saved results solely to check replay consistency.

Each integer-cache plan has 2,753 physical instructions, including 46 each of
cache normalization, quantize/write, and dequantization. Native plan validation
checks all eight rank bindings and rejects 28 malformed contracts, including
cache scale type, storage format, scratch extent, and missing dependencies.
The model's default BF16 definition is unchanged. Frozen-arena measurements and
same-binary BF16 regression are recorded below when complete.
