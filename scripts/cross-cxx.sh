#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
sysroot=${PANGU_SYSROOT:-$root/.cross/sysroot}
compiler=aarch64-linux-gnu-gcc
extra=()
if [[ ${0##*/} == *cxx* ]]; then
  compiler=aarch64-linux-gnu-g++
  extra=(-isystem "$sysroot/usr/include/c++/10.3.1" -isystem "$sysroot/usr/include/c++/10.3.1/aarch64-linux-gnu" -isystem "$sysroot/usr/include/c++/10.3.1/backward")
fi
exec "$compiler" --sysroot="$sysroot" -B"$sysroot/usr/lib64/" -L"$sysroot/usr/lib64" \
  -nostdinc "${extra[@]}" -isystem "$(aarch64-linux-gnu-gcc -print-file-name=include)" \
  -isystem "$sysroot/usr/include" "$@"
