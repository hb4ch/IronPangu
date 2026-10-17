# Build and cross-compilation guide

This is the current build procedure for InferFabric. Run Rust builds and tests locally in Linux/WSL; cross-compile Rust for the ARM64 deployment container. Build the C++ ACL/CANN bridge inside that container. Serving uses the vendored **vLLM 0.25.1 Rust frontend**, not a Python inference engine.

Local setup, CPU tests, and Rust cross-builds need neither model weights nor an NPU. Native compilation needs CANN headers/libraries but does not need a compute device. Only the final hardware qualification and native serving steps occupy NPUs. The commands below document deployment; they are not prerequisites for CPU-only development.

## 1. Build layout and outputs

Run shell commands from the repository root unless stated otherwise. Examples use Windows checkout `D:\omni\IronPangu`, WSL path `/mnt/d/omni/IronPangu`, and remote workspace `/data/p00603624/inferfabric`. Substitute your actual paths and container name; older records use `ironpangu-npu` and `/data/p00603624/ironpangu`.

| Component | Build command in local WSL | Artifact |
|---|---|---|
| Core crates and CPU CLI | `cargo build --workspace --locked` | Cargo target directory, `debug/inferfabric` |
| Local CPU frontend | `cargo build --manifest-path frontend/Cargo.toml --locked` | Frontend target directory, `debug/inferfabric-server` |
| ARM64 CLI, debug | `bash scripts/cross-compiler.sh` | `.deploy/inferfabric-compiler` |
| ARM64 CLI, release | `bash scripts/cross-npu-probe.sh` | `.deploy/inferfabric-npu` |
| ARM64 frontend, debug | `bash scripts/cross-frontend.sh` | `.deploy/inferfabric-server` |
| ARM64 frontend, release | `bash scripts/cross-frontend.sh --release` | `.deploy/inferfabric-server-release` |
| ARM64 OpenPangu W8A8 fixture exporter | `bash scripts/cross-openpangu-probe.sh` | `.deploy/export_openpangu_linear` |
| Native C++ library and probes, inside target Docker | `cmake -S native -B native-build && cmake --build native-build -j4` | `native-build/libinferfabric_acl.so` and probe executables |

Both ARM64 CLI scripts build the same `inferfabric-cli` package; their profile and copied filename differ. Despite its name, `cross-npu-probe.sh` only compiles and strips locally. It does not connect to a remote machine or execute a probe. The frontend defaults to the debug profile used by existing qualification records; `--release` creates a separate artifact without replacing it. Release builds require their own correctness and performance qualification. Stripping a debug-profile binary does not turn it into a release-profile build.

The root Cargo workspace covers `crates/*`. `frontend/` is a **separate workspace**, with its own lockfile and vendored vLLM dependencies. Root workspace checks do not check the frontend. Scripts discover the Cargo target directory through `cargo metadata`, so `CARGO_TARGET_DIR` is supported. Generated `.cross/` and `.deploy/` directories are not source artifacts.

## 2. Install the local toolchain

On Windows, install WSL Ubuntu and Git/OpenSSH. Run this in an elevated PowerShell if WSL is not installed, then complete Ubuntu's initial user setup:

```powershell
wsl --install -d Ubuntu
wsl -d Ubuntu
```

Inside local Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y build-essential gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
  binutils-aarch64-linux-gnu protobuf-compiler pkg-config cmake perl curl ca-certificates git
```

Install Rust through your approved rustup distribution if `rustup` is absent, then select the previously qualified version:

```bash
rustup toolchain install 1.98.1 --profile minimal --component rustfmt --component clippy
rustup default 1.98.1
rustup target add aarch64-unknown-linux-gnu --toolchain 1.98.1
```

The root manifest requires Rust **1.96 or newer**; 1.98.1 is the recorded qualified local toolchain, not a claim that every later version is qualified. The repository does not currently pin a toolchain file. A standard rustup installation uses `$HOME/.cargo/bin`; the existing development machine uses `/home/p00603624/rust/cargo/bin`. Put the appropriate directory on `PATH` and check what is actually selected:

```bash
cd /mnt/d/omni/IronPangu
rustc --version
cargo --version
rustup target list --installed
protoc --version
aarch64-linux-gnu-gcc --version
aarch64-linux-gnu-g++ --version
command -v aarch64-linux-gnu-strip
```

The frontend needs host `protoc` and builds vendored OpenSSL, so Perl, Make and both host and cross C/C++ tools must be available. No local CANN installation is needed for the Rust builds. The previously qualified cross GCC was 15.2; distro packages may supply a different version, which needs its own compatibility verification.

For faster builds, place Cargo's output on WSL's Linux filesystem rather than the Windows-mounted checkout:

```bash
export CARGO_TARGET_DIR="$HOME/.cache/inferfabric-target"
```

Keep checkout shell scripts in LF format. If necessary, enable their executable bits in WSL (`chmod +x scripts/cross-*.sh`); Cargo invokes the compiler wrappers directly even when the outer script is launched with `bash`.

## 3. Prepare the target-container sysroot

**Skip this section for host-only tests.** Cross-linking must use the deployment image's libc and C++ headers/libraries. Using the default Ubuntu cross sysroot previously produced frontend dependencies on GLIBC_2.38/2.39, while the qualified openEuler image provides glibc 2.34.

The recorded environment used openEuler 22.03 LTS, GCC 10.3.1 and CANN 9.1.0, with this image:

```text
registry-cbu.huawei.com/omniai_omniinfer_dev/ai-infra-infer-1.0.3-a3-arm-feature/vllm_0.25.1-202609021706-daily:0.0.1
```

See [the historical environment record](remote-development.md) for image identity and driver details. Do not assume a container's current image or name from that record. Select an existing ARM64 development container, or create a build-only container from the selected image. For a build-only container, no NPU device mappings are needed. These are remote-host Docker administration commands; all project commands run through Docker:

```bash
# Run on the remote host, after choosing the available image.
IMAGE='registry-cbu.huawei.com/omniai_omniinfer_dev/ai-infra-infer-1.0.3-a3-arm-feature/vllm_0.25.1-202609021706-daily:0.0.1'
docker image inspect "$IMAGE" --format '{{.Id}} {{.Architecture}}'
docker run -d --name inferfabric-build \
  -v /data/p00603624:/data/p00603624 \
  --entrypoint /bin/bash "$IMAGE" -lc 'exec sleep infinity'
docker exec inferfabric-build bash -lc \
  'mkdir -p /data/p00603624/inferfabric && uname -m && ldd --version && g++ --version'
```

Choose a fresh container name if one already exists. On local PowerShell, export and download its sysroot:

```powershell
cd D:\omni\IronPangu
./scripts/prepare-sysroot.ps1 -Remote root@7.156.99.58 -Container inferfabric-build
```

The helper exports headers, startup objects, the dynamic loader, libc and C++ runtime files **inside Docker**, then downloads the archive and extracts it locally under `.cross/sysroot`. It expects `/data/p00603624/inferfabric` to exist in the container and be backed by the same host path, because `scp` reads the archive from the host. It currently selects WSL distribution `Ubuntu` explicitly.

Verify in local WSL:

```bash
test -f .cross/sysroot/usr/include/stdio.h
test -d .cross/sysroot/usr/include/c++/10.3.1
test -f .cross/sysroot/usr/lib64/crt1.o
test -e .cross/sysroot/usr/lib64/libc.so
test -e .cross/sysroot/usr/lib64/libstdc++.so
test -e .cross/sysroot/lib/ld-linux-aarch64.so.1
```

`cross-cc.sh` and `cross-cxx.sh` force this sysroot's includes and libraries. The C++ wrapper currently hardcodes `usr/include/c++/10.3.1` and its `aarch64-linux-gnu` subdirectory. **A different image may require wrapper changes**, not just another export. Review library paths and the loader as well. `INFERFABRIC_SYSROOT=/absolute/linux/path` overrides the sysroot location for the cross-build scripts; it does not override the hardcoded C++ header version. Regenerate and requalify the sysroot when changing deployment images. Preserve symlinks by extracting with WSL tar, as the helper does.

## 4. Fetch dependencies, check and build locally

In local WSL, populate both workspaces' dependency caches while network access is available:

```bash
cargo fetch --locked
cargo fetch --manifest-path frontend/Cargo.toml --locked
cargo fetch --locked --target aarch64-unknown-linux-gnu
cargo fetch --manifest-path frontend/Cargo.toml --locked --target aarch64-unknown-linux-gnu
```

Then run host checks. These do not initialize ACL or require model weights:

```bash
cargo fmt --all --check
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo fmt --manifest-path frontend/Cargo.toml --all --check
cargo test --manifest-path frontend/Cargo.toml --locked
cargo clippy --manifest-path frontend/Cargo.toml --all-targets --locked -- -D warnings
cargo build --workspace --locked
cargo build --manifest-path frontend/Cargo.toml --locked
```

For an offline run after caches are populated, set `export CARGO_NET_OFFLINE=true`. Missing packages must be fetched before offline compilation; do not remove lockfiles to resolve a missing-cache error.

A CPU-only functional check:

```bash
mkdir -p .deploy
cargo run --locked -p inferfabric-cli -- inspect-openpangu \
  examples/openpangu/92b.reference-config.json .deploy/openpangu-audit.json
cargo run --locked -p inferfabric-cli -- plan \
  examples/physical-plan/residual-state.logical.json .deploy/demo.ifplan --dump-ir .deploy/demo-ir
cargo run --locked -p inferfabric-cli -- explain .deploy/demo.ifplan --html .deploy/demo-plan.html
```

The OpenPangu report is a metadata audit, not an executable OpenPangu model. The physical-plan example targets the implemented CPU backend. See [OpenPangu audit usage](../examples/openpangu/README.md) and [physical planning](physical-planning.md).

Build the ARM64 artifacts locally after preparing the sysroot:

```bash
bash scripts/cross-compiler.sh
bash scripts/cross-npu-probe.sh
bash scripts/cross-frontend.sh
file .deploy/inferfabric-compiler .deploy/inferfabric-npu .deploy/inferfabric-server
aarch64-linux-gnu-readelf -l .deploy/inferfabric-server | grep interpreter
aarch64-linux-gnu-readelf --version-info .deploy/inferfabric-server
sha256sum .deploy/inferfabric-compiler .deploy/inferfabric-npu .deploy/inferfabric-server
```

Expect AArch64 Linux ELF binaries, not x86-64 or Windows executables. Check the requested interpreter and GLIBC/GLIBCXX versions against the target container. Do not run an ARM64 artifact directly on x86 WSL. Inspecting an ELF does not establish runtime or hardware compatibility.

## 5. Deploy and build C++ inside Docker

This step uses the remote machine but native compilation itself does not occupy NPUs. Keep Rust compilation local. Before deploying into an existing serving directory, drain/stop this project's server or use a separate staging directory; the helper does not stop a running process and overwrites files at its fixed destination.

From local PowerShell:

```powershell
./scripts/deploy.ps1 -Remote root@7.156.99.58 -Container inferfabric-build
```

The helper uploads `.deploy/inferfabric-server` and a source archive, extracts into `/data/p00603624/inferfabric`, runs CMake inside Docker, then invokes `acl_graph_probe --abi`. The ABI-only mode does not allocate a device. This requires the same shared host/container mount used for sysroot export. The script does **not** upload either CLI artifact, start a server, or source CANN's environment script explicitly.

If the image needs an explicit CANN environment, use a staging upload followed by these commands **inside the container** instead of relying on the helper's inherited environment:

```bash
source /usr/local/Ascend/cann/set_env.sh
cd /data/p00603624/inferfabric
cmake -S native -B native-build -DCANN_ROOT=/usr/local/Ascend/cann
cmake --build native-build -j4
./native-build/acl_graph_probe --abi
```

For an explicitly optimized native build, use a distinct directory with `-DCMAKE_BUILD_TYPE=Release`, then qualify those artifacts. The default commands do not set a CMake build type. CMake requires C++17 and links `ascendcl`, `opapi`, `nnopbase`, and `hccl`. It builds `libinferfabric_acl.so`, `acl_graph_probe`, `gdr_contract_probe`, `sampler_probe`, `tp_probe`, `physical_plan_probe`, `openpangu_linear_probe`, and `openpangu_parallel_probe`.

To upload the release CLI separately from local PowerShell:

```powershell
scp .deploy/inferfabric-npu root@7.156.99.58:/data/p00603624/inferfabric/inferfabric-npu
ssh root@7.156.99.58 "docker exec inferfabric-build chmod +x /data/p00603624/inferfabric/inferfabric-npu"
```

Rebuild and deploy Rust and the native library from the same revision when changing their C ABI. Record the source revision, image digest, compiler versions and artifact hashes together. Old `libpangu_acl.so` artifacts and historical binary names are not interchangeable with current InferFabric builds.

The program backend also compiles the host API adapters in
`native/vendor/omni_opapi` into `inferfabric_owned_opapi`. They fix optional
descriptor ownership while continuing to call the installed vendor L0 operators
and kernels. This target alone uses `_GLIBCXX_USE_CXX11_ABI=0`, matching CANN's
host C++ API; the serving frontend and executor keep their existing ABI.
The source provenance, license and changes are recorded in that directory.
An incompatible vendor L0 ABI is a link error. Do not disable the adapter to
hide that error or reuse a frozen arena from a different backend library.

Uncaptured forward metadata uses an owning-stream completion epoch. Tensor
values retain descriptor leases, so values used by sampling after the forward
returns remain valid. The completion boundary destroys transient invocations
before clearing ordinary/strided descriptor caches. Decode/graph metadata and
immutable weight-view keys remain persistent. `Context::descriptor_lease` owns
metadata only; `DeviceTensor` and resident tensor values also retain their device
buffer leases. Raw `descriptor`/`strided_descriptor` results from an uncaptured
epoch must not escape its completion without an explicit owner. The audit
distinguishes transient cache entries from live leased descriptors and their
peaks. `prepared_execution_probe` verifies these lifetimes and negative capture
contracts before full-model qualification.

## 6. Runtime checks: CPU mock versus NPU execution

The mock server can run in the build-only container with no compute devices. In one container shell:

```bash
cd /data/p00603624/inferfabric
./inferfabric-server --mock frontend/fixtures/mock.inferfabric frontend/fixtures/mock-tokenizer 18080
```

In another shell in the same container:

```bash
./inferfabric-server --smoke-test http://127.0.0.1:18080
```

This exercises the vLLM Rust HTTP/tokenizer/streaming integration with synthetic results. It does not qualify model kernels.

**The following stage requires assigned NPUs and a device-enabled container.** A build-only container without device mappings cannot perform it. See [native bring-up](npu-bringup.md) and [native serving](npu-native-milestone.md) for hardware qualification. Do not run model probes or `--profile-only` while avoiding NPU occupation: profiling executes on the device too.

For the implemented Qwen TP1 path, inside the NPU container with the artifacts and checkpoint available:

```bash
source /usr/local/Ascend/cann/set_env.sh
cd /data/p00603624/inferfabric
./inferfabric-server --native examples/qwen35-2b-checkpoint.inferfabric \
  /data/p00603624/models/qwen35 native-build/libinferfabric_acl.so 0 18081 \
  --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 16 \
  --gpu-memory-utilization 0.9 --memory-profile memory-profile.json
```

The current runtime profiles memory and qualifies graph execution during startup. In a second container shell after readiness:

```bash
./inferfabric-server --native-smoke-test http://127.0.0.1:18081
./inferfabric-server --sampling-smoke-test http://127.0.0.1:18081
./inferfabric-server --batching-smoke-test http://127.0.0.1:18081
```

For TP2, use `examples/qwen35-2b-tp2.inferfabric`, assign both devices, and set `HCCL_OP_EXPANSION_MODE=HOST`, `HCCL_CONNECT_TIMEOUT=120`, and `HCCL_EXEC_TIMEOUT=120` before launch. Read [tensor parallelism](npu-tensor-parallel.md) for the qualified configuration and limitations. The server binds container loopback; publishing a Docker port alone does not expose a loopback-only listener. Arrange network access/tunneling for the actual container network configuration.

These commands use the deployed name `inferfabric-server`; the README uses the optional deployment alias `inferfabric-native-server`. They refer to the same locally built frontend binary. OpenPangu V2 92B/505B native serving remains roadmap work; a successful build or metadata audit does not imply those models can be served yet.

## 7. Common failures

| Symptom | Check / remedy |
|---|---|
| `cargo` missing or wrong compiler version | Check WSL `PATH`, rustup default and any directory override; Windows Rust is not the WSL toolchain. |
| `protoc` missing | Install `protobuf-compiler` in local WSL, where frontend build scripts run. |
| Compiler-wrapper permission denied or `/usr/bin/env: bash\r` | Ensure executable permissions and LF line endings on `scripts/cross-*.sh`. |
| Missing `crt*.o`, C++ headers or libc at link time | Check sysroot extraction, symlinks, image layout and the wrapper's hardcoded GCC header version. |
| `GLIBC_*` / `GLIBCXX_*` not found in container | Rebuild against that image's sysroot, inspect ELF version requirements and check deployed artifact identity. |
| `Exec format error` | The ARM64 executable must run in an ARM64 environment, not directly in x86 WSL. |
| CMake cannot find ACL/OPAPI/HCCL | Use the CANN development image and correct `CANN_ROOT`; source its environment before loading native libraries. |
| `scp` destination missing / export archive absent | Create the workspace inside Docker and confirm `/data/p00603624` is mounted at the same host/container path. |
| Root tests pass but frontend does not compile | Run the separate frontend workspace checks and fetch its dependencies. |
| Offline Cargo fails | Populate both workspace/target dependency caches with network access first. |
| ACL/HCCL initialization or graph capture fails | This is a hardware/runtime qualification issue, not proof of a cross-link problem. Check assigned devices, driver/CANN compatibility and the documented HCCL mode. |

Build recipes and scripts are the source of truth for commands; [validation records](validation/2026-10-15-openpangu-audit/README.md) describe what was actually tested. A reproducible production release still needs the pinned build-image/toolchain, dependency provenance and per-model qualification gates in the [roadmap](../README.md#roadmap-to-production).

## 8. OpenPangu initialization and decoder qualification

These are development qualification commands, not a 92B serving command. Build the fixture and load-plan exporters in local WSL:

```bash
bash scripts/cross-openpangu-probe.sh
```

This produces `.deploy/export_openpangu_linear`, `export_openpangu_tensor`, `export_openpangu_moe` and `export_openpangu_load_plan` as ARM64 executables. Transfer the required binaries, native sources and scripts into the remote Docker workspace. Restore executable permissions after a Windows-created archive if necessary. Within that container, from the staged repository root:

```bash
chmod +x .deploy/export_openpangu_*
bash scripts/qualify-openpangu-load.sh /models/92B \
  examples/openpangu/92b-tp8-ep8.inferfabric load-qualification
bash scripts/qualify-openpangu-attention.sh attention-qualification
```

The loader qualification occupies all eight devices from the TP8/EP8 DSL while loading and checking each rank; run it only on the assigned devices. It generates manifests from an audited checkpoint, builds the C++ loader, checks malformed manifests and compares every device upload with a readback. The attention qualification uses device 0 and checks synthetic absorbed MLA inputs against an independent CPU reference, including graph replay. The other `qualify-openpangu-{tp8,router,moe,mhc,mome}.sh` scripts document their individual arguments at their entry points. See the [loader evidence](validation/2026-10-15-openpangu-load/README.md) and [attention evidence](validation/2026-10-15-openpangu-attention/README.md) for exact scope and remaining integration work.

The exporter now emits shared `IFINIT01` initialization programs (`rank-N.bin`) and their readable `rank-N.init.json` IR alongside the model placement `plan.json`. The generic `initialization_probe` executes these programs. Old `IFLOAD01` files are rejected: regenerate plans after updating the exporter. See [initialization format and ownership](initialization-plan.md).

After successful initialization qualification, run all three full 92B generation cases through the same executor:

```bash
bash scripts/qualify-openpangu-decoder.sh /models/92B load-qualification/plan decoder-qualification
python3 scripts/compare-openpangu-decoder.py decoder-qualification/results.json \
  docs/validation/2026-10-15-openpangu-decoder/reference/results.json comparison.json
```

The saved reference applies only to the pinned checkpoint/tokenizer in the validation report. For a different checkpoint, regenerate the offline reference with `scripts/reference-openpangu-decoder.sh` before comparing. The diagnostic decoder still has the documented 512-token, single-request limits; DSL serving budgets do not override them.

## Model definition inputs

Source readers now require versioned v1 DSL or an `inferfabric.model` v1 artifact emitted by compiled code. Legacy flat key/value files are rejected. All checked-in examples and frontend fixtures are migrated. See [dual authoring and debug-plan commands](model-authoring.md). Rebuild Rust tools/frontend after this source-language change; initialization and decoder binary formats are unchanged.
## Frozen arena preparation for the 92B token backend

After local Rust cross-compilation and the native Docker build, the physical
token bundle needs hardware preparation before normal execution. Run this in
the qualified NPU container with the CANN/vendor environments loaded:

```sh
INFERFABRIC_PREPARE_ARENA=1 INFERFABRIC_DECODE_GRAPH=1 \
  python3 scripts/qualify-openpangu-decoder.py \
    --checkpoint /models/92B --plan compiled-plan \
    --binary native-build/openpangu_decode --output preparation
python3 scripts/plan-shared-workspace.py compiled-plan workspace-plan \
  preparation/arithmetic-prepared preparation/geography-prepared preparation/chinese-prepared
INFERFABRIC_PREPARE_ARENA=1 INFERFABRIC_DECODE_GRAPH=1 \
  python3 scripts/qualify-openpangu-decoder.py \
    --checkpoint /models/92B --plan workspace-plan \
    --binary native-build/openpangu_decode --output fixed-preparation
python3 scripts/bind-prepared-arena.py \
  workspace-plan fixed-preparation/arithmetic-prepared bound-plan
```

The first run discovers workspace demands. `plan-shared-workspace.py` merges their
maximum per rank and records program/backend fingerprints. The second run starts
with one fixed workspace lease before any model buffers are allocated. Only this
second, zero-growth preparation can bind an arena. Requests above the planned
maximum fail; changing runtime shapes or implementations requires discovery again.
See [shared workspace planning](shared-workspace-planning.md).

Use `bound-plan` for subsequent execution with the same native binary and runtime.
The normal executor rejects missing, stale, incomplete or exhausted arena
bindings. Rebuilding the native binary requires preparation again. The benchmark
has an explicit `--prepare-arena` switch for the initial measurement run; omit it
when verifying the bound plan against an existing qualified output fixture.
The preparation driver checks answer anchors and residency; use the benchmark's
unchanged qualified fixture for exact-output regression validation.


## Integer KV-cache kernels

Rust model compilation remains local cross-compilation. Build the native Ascend C
storage kernels inside the assigned Docker container before configuring the decoder:

```sh
source /usr/local/Ascend/cann/set_env.sh
cmake -S native/kernels -B cache-build -DSOC_VERSION=Ascend910_9391 -DCMAKE_BUILD_TYPE=Release
cmake --build cache-build -j8
cmake -S native -B native-build -DIF_CACHE_KERNEL_LIBRARY="$PWD/cache-build/lib/libinferfabric_cache.a"
cmake --build native-build --target openpangu_decode cache_quantization_probe -j8
native-build/cache_quantization_probe
```

Select the actual device SoC when using another machine. The static device library
is linked into the decoder and therefore covered by its arena fingerprint. Rebuild
and reprepare the arena after changing it. INT8/INT4 plans contain explicit
quantize-write and dequantization instructions; the default DSL remains BF16.
Full-model numerical and performance evidence is recorded separately from the
storage-kernel probe in the [cache report](validation/2026-10-16-integer-cache/README.md).
Prepared audit directories also contain `memory-rank-N.json`: device HBM before
model initialization and after generation, including checkpoint parameters,
prepared storage, graph and runtime overhead. Device-wide figures can include
other processes; use arena manifests to inspect this executor's exact bindings.

### Packed MLA C8 verification

After the Ascend C cache library and native decoder build above, build and run
`packed_cache_probe` inside the NPU container. It checks the fused vendor writer,
per-128 scales, BF16 rotary bits, packed attention and changing device metadata
under graph replay. Rust compiler changes still use the local cross-build.

```bash
cmake --build native-build --target openpangu_decode packed_cache_probe token_plan_test -j8
native-build/packed_cache_probe
./inferfabric-compiler compile-token examples/openpangu/92b-tp8-ep8-c8.inferfabric \
  /models/92B cache-c8-plan --allow-stale-index-size
native-build/token_plan_test cache-c8-plan
```

The installed Omni inference operator library must export
`aclnnAiInfraKvQuantSparseFlashAttention` and support `pertile128` in
`aclnnAiInfraKvRmsNormRopeCache`. Missing support fails; there is no host fallback.
Follow [the C8 full-model qualification procedure](validation/2026-10-17-packed-c8/README.md)
to prepare and bind the arena before ordinary execution.

### Multiple resident request states

Compile storage capacity separately from the invocation batch size:

```bash
./inferfabric-compiler compile-token examples/openpangu/92b-tp8-ep8-c8.inferfabric \
  /models/92B resident-source --allow-stale-index-size --resident-sequences 2
cmake --build native-build --target program_engine_probe page_binding_test -j4
native-build/page_binding_test
```

`--resident-sequences` accepts 1–64. Counts above one emit the `IFTOKN03` binary
header with an explicit resident capacity. Each cache gets disjoint page partitions
and each recurrent operator gets independent slots. Weights, scratch and the token
graph remain shared. The physical program and viewer expose the resulting shapes.
BF16 and packed C8 support this contract; older row-quantized cache operators reject
multi-resident compilation. This option does not enlarge context or enable packed
multi-sequence kernels.

Rebuild the native library and perform workspace discovery, fixed-workspace
preparation and arena binding for this new binary. Use
`scripts/qualify-program-engine.py --interleave --permuted-pages` at each stage to
test independent continuation and slot reuse against the immutable full-model
fixtures. The C ABI and Rust adapter expose capacity, select-resident and
reset-resident operations. The HTTP worker continuously admits and interleaves
resident requests:

```bash
INFERFABRIC_DECODE_GRAPH=1 inferfabric-server --program \
  resident-bound /models/92B "$PWD/native-build/libinferfabric_program.so" 18081 \
  --max-num-seqs 2 --max-num-batched-tokens 2 --max-model-len 512
```

`max-num-seqs` cannot exceed compiled resident capacity. `max-model-len` is prompt
plus output and cannot exceed compiled context; lowering it restricts admission
without shrinking the frozen arena. The token budget bounds input tokens
per scheduling round, with at most one invocation per resident per round and a
rotating cursor when the budget is smaller than active count. A compiled packed
prefill invocation consumes its full chunk width; decode, short tails and budgets
below that width use the declared single-token variant. The default budget is
configured residents times compiled chunk width. The queue holds
at most four times the configured resident count; excess admission fails explicitly.

### Native packed-prefill qualification

Add `--prefill-tokens 4` to `compile-token` to compile the first qualified packed
variant alongside serial decode. This emits `IFTOKN04`; widths 2, 8 and 16 are
accepted by the compiler but do not yet have the width-4 NPU qualification.
Do not reuse a frozen arena from a different chunk width or native library.
Discover workspace with the chunk-enabled probe, plan its maximum, prepare again
with that workspace, and bind the resulting arena before normal execution.

Pass `--packed-prefill` to `scripts/qualify-program-engine.py` at **every**
preparation and final qualification stage. Combine it with `--interleave` and
`--permuted-pages` to verify independent histories and noncontiguous page bindings.
The harness verifies full output and the exact packed-graph replay count; merely
compiling a chunk variant does not prove that a client uses it.

The C ABI provides `inferfabric_program_prefill_capacity` and
`inferfabric_program_prefill`; calls accept exactly the compiled number of input
tokens and sample only the final row when requested. Token tails use the existing
single-token call. The Rust HTTP adapter checks the native chunk capacity before
readiness and schedules this variant within the requested token budget. Use
`qualify-program-http.py --expect-prefill packed` to require actual packed graph
use beyond startup, or `--expect-prefill serial` with a one-token budget to verify
the serial control. See [native evidence and timings](validation/2026-10-17-packed-prefill/README.md)
and [HTTP qualification](validation/2026-10-17-packed-http/README.md).

### Shared page ownership

Add `--shared-pages` to `compile-token` for `IFTOKN05`. The manifest and native
backend must agree on this policy before HTTP readiness. Rebuild native code and
prepare a fresh frozen bundle; neither prior binary identities nor prior arenas
grant readiness for the new executable. Native preparation probes currently need
`--interleave --permuted-pages` to install their explicit page maps. The HTTP
worker uses the Rust allocator to reserve partial page maps automatically.

The shared pool retains the compiled physical capacity. Request admission reserves
prompt plus maximum output tokens across every component/rank, with recurrent
slots pinned to the resident ID. Completion and cancellation clear only owned
storage before retirement. `qualify-program-http.py` automatically verifies
cross-partition bindings and zero remaining occupied ownership for shared bundles.
This does not yet enable prefix sharing, SWA ring recycling or 64K execution.
See [shared-page qualification](validation/2026-10-17-shared-pages/README.md).

### Native slow-consumer qualification

Build the opt-in test executable in local WSL with
`bash scripts/cross-frontend-tests.sh`; it writes
`.deploy/inferfabric-server-tests`. Copy it into the NPU container and run
`scripts/qualify-program-backpressure.py` there with `--binary`, `--plan`,
`--checkpoint`, `--library`, `--cases` and a new `--output` directory. Use a frozen
shared-page plan with at least two residents and sufficient context for the
fixture prompt plus its reference output through EOS. Source the same CANN/vendor environments as
normal serving. Contexts above 512 still require the experimental opt-in.

The test uses a two-message queue to force backpressure before a short reference
reaches EOS; production queues remain capped at 64 messages. It checks peer
reference output, drains and compares the complete EOS-scoped continuation
against an unthrottled run, then cancels a full queue
and verifies slot reuse. The wrapper checks every rank's frozen arena, workspace,
device-residency counters and final ownership reclamation. This isolates native
worker backpressure; HTTP socket buffering and sustained slow-client load remain
separate qualification gates. The hardware test is ignored by ordinary local
`cargo test` runs and must not be counted as passing until explicitly executed.

### Repeated mixed-load resource qualification

For an already qualified frozen bundle, add `--load-rounds 24
--load-concurrency 4 --load-rss-growth-limit-mib 128` to the normal
`scripts/qualify-program-http.py` invocation inside the NPU container. Supply
independent `--cases` and long `--resident-cases`, `--respect-eos`,
`--check-arithmetic-history`, two active sequences and a 4096-token budget.
Each wave compares concurrent outputs and usage against the references, then
disconnects a stream and verifies recovery. This does not require rebuilding
or reprofiling an unchanged native library and frozen plan.

The optional RSS gate reads Linux `/proc/<server-pid>/status` and `fd` after
each wave. The native ranks are threads in that server process. Every later
sample is compared with the first completed wave: RSS growth must stay within
the explicitly selected limit, with no thread or FD count growth. Raw samples
are retained in `load-progress.json`; `resource-growth.json` records the gate
even when it fails. The limit is a qualification tolerance, not a production
memory budget. These samples do not measure continuous RSS peaks or slow-socket
buffer bounds; the separate per-rank audits check device allocations, workspace,
activation transfers and final shared-page reclamation.

For a failing host RSS gate, build the optional diagnostic in the Linux container:

```sh
g++ -std=c++17 -shared -fPIC -O2 -pthread -Wall -Wextra -Werror \
  native/probes/host_allocator_probe.cpp -ldl -o host_allocator_probe.so
python3 scripts/test-host-allocator-probe.py ./host_allocator_probe.so
```

Then add `--host-allocator-probe ./host_allocator_probe.so` to the HTTP
qualifier. Only the server receives `LD_PRELOAD`; the native execution library
and frozen plan remain unchanged. The diagnostic writes `host-allocator.jsonl`
every two seconds using glibc 2.33+ `mallinfo2`. It adds one host thread and one
file descriptor, which are present before the first load-wave baseline. It makes
no ACL calls or device reads. Compare allocated arena bytes, free arena bytes
and mmap bytes with timestamped post-wave RSS samples. Allocator counters are
not allocation stack traces and do not alone prove a leak or its owner. Sampling
can contend with host allocations; timing under this diagnostic is not a
production performance result. It is opt-in and rejects existing output files.

The qualifier also enables bounded sampled allocation stacks at the start of
repeated-load waves (default one allocation per 1024 per thread). It records
malloc/calloc/realloc/memalign paths; allocator-internal suballocations and other
entry points are not a complete heap census. Each recorded pointer is removed
when an intercepted free occurs; failed realloc retains its original record.
Overflow is reported and invalidates analysis. The fixed table has 65,536 slots.
Helpers and forked children do not inherit active tracking.

After each wave the client waits for an acknowledged snapshot before submitting
the next wave. Analyze these snapshots, rather than whichever periodic sample
happened to overlap prefill:

```sh
python3 scripts/summarize-host-allocations.py OUTPUT/host-allocation-stacks.txt \
  --progress OUTPUT/load-progress.json --output OUTPUT/stack-summary.json
```

The report preserves module-relative return addresses for `addr2line -f -C` and
reports unscaled sampled bytes/counts. It must not be interpreted as exact byte
attribution or proof that a growing vendor pool is unbounded. The additional
metadata and locking make this an offline diagnostic only.

For the qualified CANN 9.1 installation, the probe also interposes its exported
`op::internal::Allocate/DeAllocate` functions. `kind=cann-client` records track
objects checked out from the vendor pool; `kind=libc` records track backing
allocations. The two byte totals overlap and must never be added. The diagnostic
resolves and forwards to the installed functions without reading private block
headers or changing pool policy. This is version-specific offline tooling,
not an execution dependency or a supported vendor API contract. Source the CANN
environment and add `--cann` to `test-host-allocator-probe.py` to verify a known
pool-object allocation/return; this test does not select an NPU device.

`host_metadata_probe` is a separate small NPU executable built by CMake. Its
`ordinary`, `repeatable`, `retained` and `released` modes repeat the 4096-row
strided BF16 column-slice/Cast path for 17 batches of 128 calls, without model
weights. Ordinary executors follow vendor automatic destruction; repeatable
executors are explicitly destroyed after the existing batch stream completion.
The other modes compare retaining input/output descriptors with releasing them
after each batch. It reports glibc counters, not model correctness or production
latency. Run it only inside the NPU container with the normal CANN environment.
