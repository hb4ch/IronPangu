# CANN pool-client ownership — 2026-10-18

The host RSS gate still fails. This milestone narrows the ownership defect;
it does not change the production executor or qualify serving stability.

`host-pool-ownership-http` completed eight four-client load waves, with 40
wave/recovery outputs checked against independent references through EOS.
Normal SIGTERM drain reports 75 admissions, 24 mixed ticks, 98,472 consumed
input tokens, zero remaining shared-page bytes and closed native workers.
RSS grew by 158,130,176 bytes (150.8 MiB), over the unchanged 128 MiB gate.
Threads (632) and FDs (78) remained stable. The runner therefore exited 1.

The library remains `native-build-device-zero/libinferfabric_program.so`,
the bundle `device-zero-apc8k-frozen-plan`, and frontend
`inferfabric-server-apc-device-zero`. There are two active residents and a
4096-token uncaptured prefill budget. Prefix caching is disabled for this test.
Remote directory: `/data/p00603624/inferfabric/openpangu92/host-pool-ownership-http`.
HCCL port: 40000. Per-rank memory and transfer audits are included.

## Ownership evidence

The offline probe now intercepts CANN 9.1 `op::internal::Allocate/DeAllocate`
in addition to libc. A returned pool client disappears from the semantic
records even if its backing malloc block remains cached. Sampling is 1/1024
per thread, independently for each ownership level. Between-wave snapshots
are acknowledged before submitting the next wave, and report zero drops.

| Sampled live bytes, first to last acknowledged wave | Growth |
| --- | ---: |
| libc backing allocations | 219,531 |
| CANN pool clients | 32,728 |

These levels overlap: **never add them**, scale them to estimate a leak, or
interpret sampled bytes as a complete heap census. Raw compressed snapshots,
the exact wave timestamps and the grouped report are included.

The largest surviving pool-client stacks resolve through `aclCreateTensor`
at `libnnopbase.so+0xf95a4` / `+0xf9574`. Their vendor callers are:

* `libcust_opapi.so+0x953a0`, reached from attention workspace preparation at
  `+0x96584`. Disassembly shows an internal helper creating a zero-sized FLOAT
  tensor when the softmax-LSE output flag is false. `addr2line` labels the
  address with the nearest exported `CreateMaxSumPlaceholder` symbol; the
  actual helper starts at `+0x952f0`, so that label alone is not a source name.
* `libcust_opapi.so+0x13214c`, in
  `aclnnAiInfraKvRmsNormRopeCacheGetWorkspaceSize`. Disassembly shows the
  `kCacheRef == nullptr` branch creating a zero-sized tensor with the input
  dtype; the non-null branch creates an executor-owned view instead.

The first family reaches native `Decoder::attend`; the second reaches
`resident::normalized_cache_append`. These records establish survival across
the measured waves, not the full destruction behavior of every vendor path.
The installed vendor library differs from the local omni-ops source. In the
local attention source, the temporary tensor pointer is passed by value and
the caller subsequently destroys its unchanged null pointer, but this source
observation must not be treated as an exact reconstruction of the installed DSO.

## Small operator control

`host_metadata_probe` performs 17 batches of 128 BF16 strided-to-contiguous
Cast invocations (2176 total) on a small device buffer, without model weights.
The four included JSONL traces show first-to-last allocated-arena growth:

| Lifetime mode | Growth (bytes) |
| --- | ---: |
| Ordinary auto-destroyed executors | 4,112 |
| Repeatable executors destroyed after stream completion | 4,112 |
| Fresh descriptors retained until probe exit | 67,600 |
| Fresh descriptors released after stream completion | 4,112 |

This control does not reproduce the model's vendor placeholder path. It does
rule against changing ordinary executor destruction or globally disabling
descriptor reuse based solely on the earlier libc stacks.

The container diagnostic self-test checks pool-client allocation/free without
selecting a device, alongside malloc/free, failed realloc, snapshot
acknowledgement, helper isolation and signal-driven shutdown. Three allocation
summary tests and twelve HTTP helper tests pass. The C++ probe builds in the
container with warnings treated as errors and through its CMake target.

Next fix the affected vendor placeholder ownership with a matching ABI/source
or a semantically equivalent caller-owned empty descriptor where supported.
Keep output flags, kernels, device residency and decode graph behavior intact.
Then rerun the full EOS/resource gate with a freshly qualified library/plan;
do not raise its tolerance or present this diagnostic run as a performance win.
