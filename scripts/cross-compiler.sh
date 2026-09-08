#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
export CC_aarch64_unknown_linux_gnu="$root/scripts/cross-cc.sh"
export CXX_aarch64_unknown_linux_gnu="$root/scripts/cross-cxx.sh"
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="$root/scripts/cross-cc.sh"
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-L native=${INFERFABRIC_SYSROOT:-$root/.cross/sysroot}/usr/lib64"
cargo build --manifest-path "$root/Cargo.toml" --locked --target aarch64-unknown-linux-gnu -p inferfabric-cli
mkdir -p "$root/.deploy"
target_dir=$(cargo metadata --manifest-path "$root/Cargo.toml" --format-version 1 --no-deps | sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p')
aarch64-linux-gnu-strip -o "$root/.deploy/inferfabric-compiler" "$target_dir/aarch64-unknown-linux-gnu/debug/inferfabric"
