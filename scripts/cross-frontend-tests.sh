#!/usr/bin/env bash
# Build on local WSL. Run the resulting opt-in tests only inside the NPU container.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
export CC_aarch64_unknown_linux_gnu="$root/scripts/cross-cc.sh"
export CXX_aarch64_unknown_linux_gnu="$root/scripts/cross-cxx.sh"
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="$root/scripts/cross-cc.sh"
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-L native=${INFERFABRIC_SYSROOT:-$root/.cross/sysroot}/usr/lib64"
report=$(mktemp)
trap 'rm -f "$report"' EXIT
cargo test --manifest-path "$root/frontend/Cargo.toml" --locked --target aarch64-unknown-linux-gnu \
  --bin inferfabric-server --no-run --message-format=json > "$report"
binary=$(python3 - "$report" <<'PY'
import json,sys
records=[json.loads(line) for line in open(sys.argv[1])]
paths=[r['executable'] for r in records if r.get('reason')=='compiler-artifact'
       and r.get('target',{}).get('name')=='inferfabric-server'
       and r.get('profile',{}).get('test') and r.get('executable')]
if len(paths)!=1: raise SystemExit('expected exactly one ARM frontend test executable')
print(paths[0])
PY
)
mkdir -p "$root/.deploy"
aarch64-linux-gnu-strip -o "$root/.deploy/inferfabric-server-tests" "$binary"
