# Remote Docker development

Validated on 2026-09-07. SSH: `root@7.156.99.58`. Remote project: `/data/p00603624/ironpangu`.

## Container and compatibility

Container `ironpangu-dev` uses:

```
registry-cbu.huawei.com/omniai_omniinfer_dev/ai-infra-infer-1.0.3-a3-arm-feature/vllm_0.25.1-202609021706-daily:0.0.1
```

Image ID: `19cc2ebef117`. The originally suggested September 3 develop image was not present.
The image's server entrypoint is overridden with `/bin/bash -lc 'exec sleep infinity'`.
Only `/data/p00603624` is mounted writable. Driver files and `npu-smi` are read-only mounts.
The container has management devices but no compute-device assignment; no existing workloads were stopped.

| Component | Observed value |
|---|---|
| Remote architecture | aarch64 |
| Container OS / libc | openEuler 22.03 LTS / glibc 2.34 |
| CANN | `/usr/local/Ascend/cann-9.1.0` |
| Driver | 25.5.1 |
| Container C++ | GCC 10.3.1 |
| Local Rust | 1.98.1 in WSL Ubuntu |
| Local cross C/C++ | aarch64-linux-gnu GCC 15.2 |
| vLLM | v0.25.1, `752a3a504485790a2e8491cacbb35c137339ad34` |

All 16 devices were occupied by existing VLLMWorker_TP processes, with roughly 54 GiB HBM used per device. NPU execution is deferred. This does not establish hardware compatibility of any operator or graph.

## Build locally

Rust builds and Rust dependency compilation always run locally in WSL. The remote container compiles only the project C++ ACL component. The full frontend requires local `protoc`, the ARM64 Rust target, and cross GCC/G++.

From local PowerShell:

```powershell
./scripts/prepare-sysroot.ps1
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/omni/IronPangu && scripts/cross-frontend.sh'
./scripts/deploy.ps1
```

The sysroot comes from the selected Docker image. The wrapper forces its C/C++ headers, startup objects and libraries. A plain local ARM64 build of the full frontend required GLIBC_2.38/2.39 and failed inside the container; the sysroot build requires at most GLIBC_2.34 and runs successfully. `.cross/` and `.deploy/` are local generated artifacts, excluded from Git. The wrapper currently targets the observed GCC 10.3.1 C++ header layout and produces a debug build stripped for deployment.

The deploy script copies the local Rust binary and source, compiles native code through `docker exec`, and loads the native ABI without allocating NPU resources. Stop this project's running frontend before replacing its executable; deployment does not stop processes automatically.

## Run and verify inside Docker

Start the weight-independent synthetic fixture:

```sh
ssh root@7.156.99.58 "docker exec -d ironpangu-dev bash -lc 'cd /data/p00603624/ironpangu && exec ./pangu-server --mock frontend/fixtures/mock.pangu frontend/fixtures/mock-tokenizer 18080 > frontend.log 2>&1'"
ssh root@7.156.99.58 "docker exec ironpangu-dev /data/p00603624/ironpangu/pangu-server --smoke-test http://127.0.0.1:18080"
```

The server binds container loopback only. Its public model name is `ironpangu-mock`. It uses the actual pinned vLLM Rust HTTP/chat/tokenizer/SSE implementation and a persistent Iron Pangu mock scheduler. The fixture vocabulary is intentionally synthetic, not Qwen tokenization or inference. Request `temperature=0`; unsupported sampling features are rejected.

Successful smoke output checks health, chat, stream/collected agreement, four concurrent requests, and HTTP 400 for unsupported sampling. `native-build/acl_graph_probe --abi` returned ABI version 1. The device graph probe takes an explicit device ID and must wait for a suitable compute-device assignment.

## Validation and remaining work

Passed: core Rust formatting/lint/tests; frontend lint and stream/lifecycle tests; local ARM64 cross-build; container ABI load; container HTTP smoke tests. No model weights were read and no NPU graph executed.

The real `AscendBackend` and registered transfer adapter remain unimplemented. The C++ bridge currently provides checked allocations/copies and a graph-copy probe, not full model operators. Next work includes Rust checkpoint loading, verified Qwen mathematical/weight descriptors, ACLNN/custom operator adapters, typed device-state transport, and real graph mutation/PD numerical validation. The live scheduler currently has an explicit mock constructor; its synchronous mock fences must never be treated as NPU transfer fences.

## NPU execution enabled

The user released the cards later on 7 September. `ironpangu-npu` now has explicit compute-device access to devices 0 and 1. [Native bring-up](npu-bringup.md) records successful checkpoint projection/RMSNorm numerical checks and graph-replay qualification on both devices. Native ABI version 2 adds prepared math operations. The earlier statements about occupied devices and no NPU execution describe the initial bring-up stage, not current status.
