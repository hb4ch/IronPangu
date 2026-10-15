# Native OpenPangu paged sparse attention qualification

Roadmap OP-04 / OP-07 component milestone, 2026-10-15. This is not full-model inference evidence.

Direct C API: `aclnnAiInfraSparseFlashAttentionPioneer`, from the installed Omni inference operator package. Builds and execution run inside `inferfabric-92b` on 7.156.99.58; no Torch or Python inference is involved.

The test uses BF16 absorbed MLA queries (512 latent + 64 rotary dimensions), paged compressed keys, 128 sink keys, 512-dimensional outputs, and scale 1/sqrt(192). Inputs are deterministic synthetic values, not checkpoint projections. A double-precision CPU implementation independently computes attention over selected causal keys and sinks, using latent keys as values. Both six heads (TP=8) and 48 heads are tested.

Four cases cover noncontiguous/reordered cache pages, changed sequence lengths and query values, nonconsecutive sparse candidates, and short-sequence -1 padding. Two packed sequences and four queries share a fixed captured graph. Each case runs eagerly and in two replay rounds. Outputs must match eager execution bit-for-bit and the CPU reference within 0.001 + 0.02*abs(reference).

An initial deliberately underfilled candidate list failed the numerical check: supplying arbitrary fewer-than-topk candidates with -1 padding is not a supported way to implement variable windows. The qualified contract requires min(causal_length, topk) distinct valid candidates, then -1 padding. A host metadata validator now rejects underfilled, duplicated, noncausal or malformed candidates before upload. A separate window size/kernel policy remains necessary for sliding-window attention; this test does not establish that path.

Reproduce from the staged repository:

```bash
bash scripts/qualify-openpangu-attention.sh attention-qualification
```

Results and binary/dependency fingerprints are stored alongside this document. Real checkpoint sink normalization, q/k/v projection composition, rotary embedding, learned indexer selection/sharing, cache writes, SWA-512, and full decoder logits remain to be integrated and qualified.
