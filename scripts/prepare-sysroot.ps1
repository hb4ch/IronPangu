param([string]$Remote = 'root@7.156.99.58', [string]$Container = 'inferfabric-dev')
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force "$root/.cross/sysroot" | Out-Null
$export = 'cd / && shopt -s nullglob && files=(usr/include lib/ld-linux-aarch64.so.1 usr/lib64/crt*.o usr/lib64/libc.so* usr/lib64/libc_nonshared.a usr/lib64/libm.so* usr/lib64/libpthread* usr/lib64/libdl* usr/lib64/librt* usr/lib64/libresolv* usr/lib64/libutil* usr/lib64/libgcc_s* usr/lib64/libstdc++*) && tar -czf /data/p00603624/inferfabric/sysroot.tar.gz "${files[@]}"'
& ssh $Remote "docker exec $Container bash -lc '$export'"
if ($LASTEXITCODE) { throw 'Container sysroot export failed' }
& scp "${Remote}:/data/p00603624/inferfabric/sysroot.tar.gz" "$root/.cross/sysroot.tar.gz"
if ($LASTEXITCODE) { throw 'Sysroot download failed' }
$linuxRoot = (& wsl -d Ubuntu -- wslpath -u $root).Trim()
& wsl -d Ubuntu -- bash -lc "cd '$linuxRoot' && tar -xzf .cross/sysroot.tar.gz -C .cross/sysroot && ln -sfn usr/lib64 .cross/sysroot/lib64 && ln -sfn libstdc++.so.6 .cross/sysroot/usr/lib64/libstdc++.so && ln -sfn libgcc_s.so.1 .cross/sysroot/usr/lib64/libgcc_s.so"
if ($LASTEXITCODE) { throw 'Sysroot extraction failed' }
