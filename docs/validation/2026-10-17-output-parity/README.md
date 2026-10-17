# Output-parity investigation

Date: 2026-10-17. Status: numerical differences remain unresolved; the serving
context stays at 512. This investigation does not replace the C8 reference or
relax the output checks.

`INFERFABRIC_OUTPUT_DIAGNOSTICS=1` enables the persistent engine's existing sampler
diagnostic output for requested samples only. It prints the top-five token IDs,
their log probabilities and the winning logit already copied as request output.
It adds no device operation, activation readback or completion boundary. Warmup
samples remain silent. The engine qualifier's `--output-diagnostics` option
checks one-to-one correspondence with returned tokens and writes
`output-diagnostics.json` before testing expected output tokens, preserving
evidence when a comparison fails. Keep this opt-in setting off for normal serving.

The five first-divergence prefixes were reconstructed from the pinned reference:
the original prompt plus all reference-generated tokens before the first mismatch.
Each prefix was replayed serially three times through the full native TP8/EP8
decoder, with shuffled pages and a newly bound frozen arena. All repetitions
produce identical top-five outputs. The native choice matches the previously
observed autoregressive divergence in each case.

| Original prompt length | Generation step (zero-based) | Reference token | Native token | Native score advantage | Reference score advantage |
|---:|---:|---:|---:|---:|---:|
| 17 | 3 | 1687 | 34755 | 0.5 | 0 (tie) |
| 31 | 1 | 25 | 198 | 2.0 | 1.0 |
| 33 | 1 | 198 | 297 | 0.5 | 0.5 |
| 65 | 4 | 297 | 1114 | 1.0 | 0 (tie) |
| 480 | 7 | 148902 | 148900 | 3.25 | 3.0 |

Advantages are differences between log probabilities of these two candidates,
within the same output distribution; normalization cancels. Native JSON scores
are printed to six significant digits. The two reference ties do not explain
the native differences: native scores are unequal there, and the other cases
have opposing nonzero margins. A tie-breaking change is therefore not a justified
fix. These diagnostic requests intentionally fail exact output comparison.

Eight-rank audits show zero activation host transfers, workspace growth, replay
allocations or internal completion fences. Arena size remains 12,451,963,904
bytes/rank and shared workspace remains 130,916,352 bytes/rank. Before binding,
the diagnostic binary also passed the standard 99-token fixture in discovery
and workspace preparation. The sampler diagnostic switch does not change the
executable program or allocation schedule.

Artifacts: `forced-prefix-cases.json`, `forced-prefix-comparison.json`,
`native-audit.json`. Reference scores come from
[`reference-c8.json`](../2026-10-17-learned-index/reference-c8.json), not native
outputs. The exact 92B checkpoint and INT8 indexer executable are unchanged.

Native library SHA-256:
`65188f8c091b827bc4e65ee13cfccc244765ad43f48bd87c6f5bf81be82f2175`.
Execution binary SHA-256:
`be0495be755faac6e2d3120733d10692d1bcbc155debb0029879598e4f333045`.

Next: isolate numerical differences at matching layer/operator boundaries while
keeping native activations on device. Do not increase the context limit or claim
full C8 equivalence from the existing short natural-language fixtures alone.

## Reference prefill-shape isolation

The offline reference harness now accepts `--max-num-batched-tokens` and enables
chunked prefill explicitly. A completed TP8/EP8 C8 run on devices 8–15 used budget
1, context 512, one request at a time, eager execution and no prefix caching.
All other checkpoint/model settings matched the previous budget-512 reference.
The native decoder diagnostic ran on disjoint devices 0–7. Timing from these
concurrent investigations is not used as a performance benchmark.

Changing reference prefill budget changed **6/14 reference sequences**: prompt
lengths 17, 32, 33, 64, 65 and 480. Native matches **10/14** budget-1 reference
sequences, with mismatches at 17, 31, 32 and 64. This supports prefill-shape
sensitivity as one source of variation; it does not prove the remaining native
operators correct or that every numerical difference is harmless. Neither
reference capture replaces the other. Both are retained in `chunk-comparison.json`
and `reference-chunk1.json` alongside the earlier reference linked above.

Reproduction inside the remote Docker workspace:

```sh
REFERENCE_NPU_DEVICES=8,9,10,11,12,13,14,15 HCCL_IF_BASE_PORT=56400 \
  bash scripts/reference-openpangu-decoder.sh \
  --cases resident-boundary-cases.json \
  --output learned-index-reference-c8-chunk1.json \
  --kv-cache-dtype int8_ds_mla --max-num-batched-tokens 1
```

Further comparisons must align prefill shape and feed the same prefix before
comparing per-operator numerical results. Autoregressive token divergence alone
cannot locate the responsible operator, and changing tie-breaking cannot repair
the opposing score margins measured above.
