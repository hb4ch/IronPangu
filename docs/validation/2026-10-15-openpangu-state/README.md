# OpenPangu 92B mHC and MoME native qualification

Date: 2026-10-15. Roadmap OP-05 / OP-07 component evidence. This does not qualify an entire decoder or a serving endpoint.

## Direct native operator access

The existing Docker image on 7.156.99.58 includes `omni_training_custom_transformer` and `omni_custom_transformer` under `/usr/local/Ascend/ascend-toolkit/vendors`. We compile against their installed ACLNN headers and call the C entry points directly. No Torch dispatch, Python extension, model Python code, or Python launcher executes in these probes. Dependency lists and binary/package hashes are recorded under each `reports/` directory.

The native CMake targets are conditional on the corresponding installed headers. Each standalone probe links one explicit custom package. Both packages use the same library basename `libcust_opapi.so`; the eventual combined executor must resolve both packages explicitly (for example, path-specific dynamic handles and symbol lookup), rather than assuming that linking two identical SONAMEs loads both. The whole production dependency closure and failure recovery remain unqualified.

## mHC

Real layer-0 attention mHC phi, alpha, beta and gamma weights are exported through the locally cross-built Rust exporter. BF16 checkpoint values are promoted to FP32; gamma is folded into phi in FP32, matching the A3 reference implementation. The direct calls are:

- `aclnnAiInfraManifoldConstrainedHyperConnectionPre`
- `aclnnManifoldConstrainedHyperConnectionSinkhornEnhance`
- `aclnnAiInfraManifoldConstrainedHyperConnectionPost`

The four residual streams have width 2560. Both normalization and hyper-connection epsilon are **1e-6** on this reference path; this is distinct from the ordinary model RMSNorm epsilon. Sinkhorn runs 20 iterations: row softmax + epsilon, initial column normalization, then 19 row/column normalization pairs. Post mixing multiplies the residual streams by the transpose of the normalized residual matrix.

An independent double-precision CPU implementation checks inverse RMS, raw phi projection, pre/post gates, residual logits, pre output, all Sinkhorn entries, and post output. The API's `outMmRes` diagnostic is the **raw** dot product before inverse-RMS scaling; the initial test incorrectly compared it with the normalized dot product and was corrected against the operator documentation.

Batches 1 and 4 each pass three synthetic-input cases (including zero residual), eager execution, and three complete graph replay rounds with changed residual and branch inputs. Graph/eager final BF16 outputs must match bit-for-bit. All intermediates are numerically rechecked on replay.

| Batch | Values checked | Maximum FP32 absolute error | Maximum BF16 absolute error |
|---|---:|---:|---:|
| 1 | 154,380 | 6.10352e-5 | 6.10352e-5 |
| 4 | 617,520 | 6.10352e-5 | 0.00195312 |

FP32 tolerance: 2e-5 + 2e-5 * abs(reference). BF16 tolerance: 1e-4 + 0.016 * abs(reference). Final pre-only merge, sandwich/block norm composition and all-layer integration are still required.

## MoME

The direct `aclnnAiInfraFusedCausalConv1d` call uses all three real layer-0 convolution weight sets: Q-latent width 1024, compressed-KV width 512, and attention-output width 6144. Checkpoint `[channels,1,3]` weights are transposed to the operator's `[3,channels]` layout. State holds the last two **input** tokens, not convolution outputs.

The qualified settings are width 3, no activation, residual connection enabled, Pangu convolution mode 1, in-place output, non-APC state, and no speculative tokens. Pangu mode suppresses the convolution contribution for the first two tokens of a new sequence, returning the residual input. It is not ordinary zero-padded convolution at those positions.

Each test processes six successive packed steps with two sequence lanes and four physical history slots. It varies chunk lengths and slot order, resets a reused slot while its old state is nonzero, pads one lane with slot -1, and includes a trailing empty lane. A CPU causal-convolution/state machine checks every output and exact history storage, including untouched slots. The same six-step trajectory is replayed three times through one captured graph; each trajectory starts with deliberately nonzero old storage. Metadata and input addresses stay fixed while their contents change. Eager/graph outputs match exactly.

| Channels | Values checked | Maximum output absolute error |
|---|---:|---:|
| 512 | 49,152 | 0 |
| 1024 | 98,304 | 0 |
| 6144 | 589,824 | 0 |

The checker permits 1e-4 + 0.016 * abs(reference); observed outputs and histories are exact for these cases. A padded in-place lane retains its input and does not update state; the future executor must separately mask its downstream computation. These checks do not qualify APC, speculative rollback, scheduler cancellation, or arbitrary production shape buckets.

## Reproduce

Cross-build the exporter locally with `bash scripts/cross-openpangu-probe.sh`; stage sources, scripts, and `.deploy/export_openpangu_tensor` into the qualified container and make the exporter executable. Inside Docker:

```bash
bash scripts/qualify-openpangu-mhc.sh /models/92B mhc-qualification
bash scripts/qualify-openpangu-mome.sh /models/92B mome-qualification
```

The scripts source the appropriate custom package environment. All NPU execution and C++ compilation happen in Docker. Evidence includes fixture hashes, not model payloads.
