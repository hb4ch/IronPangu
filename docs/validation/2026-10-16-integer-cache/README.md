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
