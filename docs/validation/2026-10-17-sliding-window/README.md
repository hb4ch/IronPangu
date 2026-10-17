# Sliding-window ring contract on NPU

Date: 2026-10-17. Scope: BF16 sink-attention operator and page-layout contract,
not full 92B long-context execution. The production context cap remains 512.

The vendor `aclnnAiInfraFusedInferAttentionSink` and its metadata producer accept
an absolute-position block table with repeated physical page IDs. With
`pre_tokens=511`, they read the correct 512-token window even after physical
pages have been recycled. RoPE keys and queries are already rotated inputs;
the executor must continue computing their rotations from absolute positions.

For page size P, window W and prewritten chunk Q, the retained union contains
W+Q-1 tokens. Across arbitrary page alignment, reserve
`ceil((W+Q-2)/P)+1` pages, capped by the full-context page count. Thus W=512,
Q=4 and P=16 require **34 pages / 544 rows**, not 32. The extra capacity protects
earlier queries from writes for later queries in the same chunk.

`SlidingWindowLayout` defines the checked layout and absolute append mapping.
It is not yet connected to production state allocation or the manifest.

## Evidence

Docker `inferfabric-92b`, host `7.156.99.58`, logical NPU 0, existing CANN and
omni custom transformer package. No other NPU process was reported before testing.

- CPU layout test exhaustively checked 262,095 chunk starts at context 65,536,
  for widths 1, 4, 16 and 32. Every live row maps uniquely and table addressing
  agrees with append addressing. Short contexts, invalid inputs and overflow
  are also checked.
- NPU cases start at 0, 15, 16, 509, 511, 512, 525, 527, 528, 1023, 2047,
  4095, 16383, 32767 and 65532, testing all four query rows of each chunk.
- The fixture loads the complete retained union before running any query.
  Unused physical rows contain poison values. The CPU reference independently
  computes the causal 512-token window plus 128 sink rows, including both latent
  and rotary dot products. It does not use the ring table to select keys.
- 552,960 output values checked across eager and two graph replay passes;
  maximum absolute error **0.00013493**, within the declared
  `0.001 + 0.02 * abs(reference)` tolerance.
- **120 ACLGraph replays**, bit-exact with their eager outputs. Device addresses
  stay fixed; query lengths and fixture contents change. No per-step workspace
  growth is allowed by the operation wrapper.

Synthetic fixture upload and output readback are test-only. This does not claim
that the serving executor has a ring, nor qualify scatter writes, model output,
concurrent ownership, prefix restoration, latency or memory under 64K serving.
The production executor was not modified and its full-model regression was not
rerun for this operator-only milestone.

## Reproduction

Inside the existing Docker working directory, with CANN and the omni inference
vendor environment sourced:

```sh
cmake -S native -B native-build
cmake --build native-build --target sliding_window_layout_test sliding_window_probe -j4
native-build/sliding_window_layout_test
native-build/sliding_window_probe
```

Remote logs: `sliding-window-layout-test.log`, `sliding-window-probe.log` under
`/data/p00603624/inferfabric/openpangu92`.

| Binary | SHA256 |
|---|---|
| sliding_window_probe | 3fc10fb9a4929d73277655202a47ba1f7d89dd91154eab9474cf8ac974b33933 |
| sliding_window_layout_test | 9e7db0bfb78fd8254786ab03da9553224f0e77abe42ca22fe4d43298eaf0d784 |
