param([string]$Remote = 'root@7.156.99.58', [string]$Container = 'inferfabric-dev')
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
if (!(Test-Path "$root/.deploy/inferfabric-server")) { throw 'Run scripts/cross-frontend.sh locally first' }
Push-Location $root
try {
    & tar -czf .deploy/source.tar.gz Cargo.toml Cargo.lock crates native frontend/src frontend/Cargo.toml frontend/Cargo.lock frontend/fixtures examples docs scripts README.md vendor/rust vendor/LICENSE vendor/README.md
    if ($LASTEXITCODE) { throw 'Source archive failed' }
    & ssh $Remote "docker exec $Container mkdir -p /data/p00603624/inferfabric"
    if ($LASTEXITCODE) { throw 'Container workspace preparation failed' }
    & scp .deploy/inferfabric-server .deploy/source.tar.gz "${Remote}:/data/p00603624/inferfabric/"
    if ($LASTEXITCODE) { throw 'Upload failed' }
    & ssh $Remote "docker exec $Container bash -lc 'cd /data/p00603624/inferfabric && tar -xzf source.tar.gz && chmod +x inferfabric-server && cmake -S native -B native-build && cmake --build native-build -j4 && ./native-build/acl_graph_probe --abi'"
    if ($LASTEXITCODE) { throw 'Container deployment or native ABI check failed' }
} finally { Pop-Location }
