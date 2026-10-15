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
| Native C++ library and probes, inside target Docker | `cmake -S native -B native-build && cmake --build native-build -j4` | `native-build/libinferfabric_acl.so` and probe executables |

Both ARM64 CLI scripts build the same `inferfabric-cli` package; their profile and copied filename differ. Despite its name, `cross-npu-probe.sh` only compiles and strips locally. It does not connect to a remote machine or execute a probe. The frontend script deliberately builds the qualified debug profile and rejects arguments such as `--release`. Stripping a debug-profile binary does not turn it into a release-profile build.

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

For an explicitly optimized native build, use a distinct directory with `-DCMAKE_BUILD_TYPE=Release`, then qualify those artifacts. The default commands do not set a CMake build type. CMake requires C++17 and links `ascendcl`, `opapi`, `nnopbase`, and `hccl`. It builds `libinferfabric_acl.so`, `acl_graph_probe`, `gdr_contract_probe`, `sampler_probe`, `tp_probe`, and `physical_plan_probe`.

To upload the release CLI separately from local PowerShell:

```powershell
scp .deploy/inferfabric-npu root@7.156.99.58:/data/p00603624/inferfabric/inferfabric-npu
ssh root@7.156.99.58 "docker exec inferfabric-build chmod +x /data/p00603624/inferfabric/inferfabric-npu"
```

Rebuild and deploy Rust and the native library from the same revision when changing their C ABI. Record the source revision, image digest, compiler versions and artifact hashes together. Old `libpangu_acl.so` artifacts and historical binary names are not interchangeable with current InferFabric builds.

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
