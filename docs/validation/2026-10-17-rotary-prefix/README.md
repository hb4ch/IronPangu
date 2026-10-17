# Native leading-column rotary qualification

Date: 2026-10-17. Ran on logical device 0 in Docker `inferfabric-92b` on
7.156.99.58. This qualifies the rotary building block, not 64K model output.

The executor and probe call the same permanent `resident_rotary.h` helpers for
strided BF16 column materialization and BF16 half rotation. They use the active
prepared context's stream, descriptors, allocation leases and workspace. The
helpers have no upload, readback, implicit dtype conversion or synchronization.
The existing physical instructions still expose each materialization separately.

The probe uses head dimension 128 with leading rotary width 64, heads 1 and 24,
and token counts 1 and 4. A deterministic BF16 fixture exercises both signs and
all columns; each packed row has its own absolute position. The independent CPU
oracle computes half rotation from the BF16 inputs and coefficients, then rounds
once to BF16. Comparison is **bit-exact**, and every unrotated tail is also checked
bit-exact. Each shape runs eagerly, then uses one graph for positions 1, 511,
2048, 65535 and 0 (packed rows use position plus row offset).

All **48,000 rotary/tail pairs** pass, including **20 ACLGraph replays** with
changed inputs and coefficient buffers. Maximum absolute error is zero. Fixture
uploads and output reads exist only in the probe; the production helper remains
device resident. This test does not certify the full decoder's frozen-arena
binding, learned query/key projections, index cache, selected-index sharing,
SWA recycling or long-context generated output.

Both `rotary_prefix_probe` and `program_engine_probe` build successfully in Docker.
Probe SHA-256: `fad22617f150b687fe8f84598c536cee40565a5e7580ebc312b3af32d92552aa`.

```sh
source /usr/local/Ascend/cann/set_env.sh
cmake -S native -B native-build
cmake --build native-build --target rotary_prefix_probe program_engine_probe -j4
native-build/rotary_prefix_probe
```

```json
{"status":"passed","checked_rotary_and_tail_pairs":48000,"graph_replays":20,"max_abs_error":0}
```
