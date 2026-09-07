#!/usr/bin/env bash
set -euo pipefail
if (( $# != 0 )); then echo 'This script builds the qualified debug profile; no arguments supported.' >&2; exit 2; fi
root=$(cd "$(dirname "$0")/.." && pwd)
export CC_aarch64_unknown_linux_gnu="$root/scripts/cross-cc.sh"
export CXX_aarch64_unknown_linux_gnu="$root/scripts/cross-cxx.sh"
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="$root/scripts/cross-cc.sh"
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-L native=${PANGU_SYSROOT:-$root/.cross/sysroot}/usr/lib64"
cargo build --manifest-path "$root/frontend/Cargo.toml" --locked --target aarch64-unknown-linux-gnu

# Copy only the ARM64 output; every Rust build remains on the local machine.
mkdir -p "$root/.deploy"
target_dir=$(cargo metadata --manifest-path "$root/frontend/Cargo.toml" --format-version 1 --no-deps | sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p')
aarch64-linux-gnu-strip -o "$root/.deploy/pangu-server" "$target_dir/aarch64-unknown-linux-gnu/debug/pangu-server"
