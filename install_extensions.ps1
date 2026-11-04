$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = (Get-Command python -ErrorAction Stop).Source
$modules = @(
    'simple-knn',
    'diff-gaussian-rasterization-reduced',
    'absolute-depth-3dgs',
    'absolute-depth-3dgs-onesh',
    'mesh-gs',
    'fast-mesh-gs',
    'fast-mesh-gs-variable'
)
foreach ($module in $modules) {
    $path = Join-Path $root (Join-Path 'submodules' $module)
    if (-not (Test-Path -LiteralPath (Join-Path $path 'setup.py') -PathType Leaf)) {
        throw "Missing CUDA extension source: $path"
    }
    & $python -m pip install --no-build-isolation -e $path
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install $module (exit $LASTEXITCODE)"
    }
}
