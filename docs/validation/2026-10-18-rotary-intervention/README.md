# RoPE coefficient intervention — 2026-10-18

The 64K exact-output failure remains unresolved. Replacing the offline
reference's RoPE tables with native-style coefficients changes its answer, but
**does not make it equal to native**. This establishes sensitivity to the tables,
not a sufficient explanation or a validated production fix. Native binaries,
frozen plans, device-residency rules and graph policy were not changed.

## Observations

The checkpoint uses base 6,400,000, rotary dimension 64 and Neox layout. Native
`set_input_rows` computes FP32 `position / pow(base, exponent)`, then host libm
sin/cos rounded to BF16. The reference constructs inverse frequencies, multiplies
positions and evaluates sin/cos on NPU. The actual model selects
`NPUDeepseekScalingRotaryEmbedding` (factor=1, mscale=1), including its YaRN
frequency blending, rather than `NPURotaryEmbedding`.

A preliminary small coefficient experiment compares ordinary unscaled NPU
inverse-frequency arithmetic with native host arithmetic. At position 65,414,
four of 64 independent sine/cosine values differ after BF16 rounding, maximum
0.00390625. Multiplying by a host-computed reciprocal reduces that count to two
but does not eliminate it. This preliminary test omits the actual class's YaRN
blend and is not an exact trace of the full model. Its script and observations
are retained; it does not justify changing native arithmetic blindly.

The first full reference trial targeted the unused ordinary rotary class. It
therefore did not apply the intervention. Its 128 output tokens exactly repeat
the baseline, but its automatic `reference_intervention` label must not be read
as proof that the hook executed. `unapplied-trial-output.json` preserves this
failed diagnostic attempt explicitly.

The corrected trial wraps the actual DeepSeek class, preserves its original
initialization, then replaces both separate sine/cosine tables and the combined
base cache with native-style BF16 coefficients. Eight `applied-*.jsonl` records
confirm the hook ran on NPU 0..7. The same cached rotary object is reused by
model layers. The table covers all 524,288 declared positions; the request uses
65,408 prompt tokens and a 128-token generation budget.

| Run | Output prefix |
| --- | --- |
| Independent reference | `6077\n1,2,3,4,...` |
| Native, both isolated and mixed | `6077\n1, 2, 3, 4,...` |
| Reference with native-style coefficients | `6077, 1, 2, 3, 4,...` |

The applied intervention first diverges from the baseline at output index 4,
before the native/baseline difference at index 7. Therefore other computational
differences still need attribution. Do not substitute the intervention output
for the independent oracle, normalize whitespace to pass the gate, or report
this as a corrected native implementation.

## Reproduction and scope

All model execution and C++ compilation ran inside `inferfabric-92b` on
7.156.99.58, workspace `/data/p00603624/inferfabric/openpangu92`. Both reference
processes exited zero; no jobs from this diagnostic remain live. Reference
settings stay TP8/EP enabled, C8, eager 4096-token chunked prefill, greedy,
no prefix cache and maximum context 65536. This is not a performance comparison.
The applied trial uses HCCL port 47000; the unapplied trial used 46800.

To reproduce the tables in that container:

```sh
mkdir diagnostic-native-rope
c++ -O3 make-native-rope.cpp -o diagnostic-native-rope/generate
./diagnostic-native-rope/generate diagnostic-native-rope
printf '%s\n' '{"positions":524288,"dimension":64,"base":6400000}' > diagnostic-native-rope/manifest.json
```

Use `run-intervention.sh` with the current `scripts/reference_rotary_override.py`
and reference runner. The generator is a diagnostic replica of the current
native formula, not new model-serving infrastructure. Both 64 MiB tables remain
remote; generated table SHA-256:

- cos: `03a0d9beeb4b0273795c65fc36362565fe5bf0b7bc09da4461d1855c0b481db0`
- sin: `ede10162748d5e54144e6ca81df9c1682b88812997c822d84bb6b1e50ed0775f`

The full applied reference log remains remote as
`extended-answer64-rope-intervention2.log`, SHA-256
`913ad3d34a816a382b50cbd3c1c6660249cd296925451fd7d94963dbc921736e`.
Retained outputs include every generated ID, top-five log probabilities and a
hash of the omitted deterministic prompt IDs. Inputs are the preceding extended
64K fixture. Intervention results are now rejected by both HTTP qualification
and native sequence-fixture construction. Their 18 local helper tests pass.

Next inspect declared reduction precision and reference operation boundaries.
The actual frozen native plan has 92 AllReduce nodes, all FP32; changing them
requires matching each operation's reference semantics and a newly compiled,
prepared and verified physical plan, not an executor-side hidden cast.
