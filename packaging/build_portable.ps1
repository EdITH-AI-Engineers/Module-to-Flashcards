[CmdletBinding()]
param(
    [string]$TesseractDir = "C:\Program Files\Tesseract-OCR",
    [string]$PythonExe = "",
    [switch]$SkipGpuPreflight,
    [switch]$SkipAssetPreparation
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$BuildRoot = [IO.Path]::GetFullPath((Join-Path $RepoRoot "build"))
$DistRoot = [IO.Path]::GetFullPath((Join-Path $RepoRoot "dist"))
$AssetsRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "assets"))
$FrozenDist = [IO.Path]::GetFullPath((Join-Path $BuildRoot "pyinstaller-dist"))
$PyInstallerWork = [IO.Path]::GetFullPath((Join-Path $BuildRoot "pyinstaller-work"))
$BundleDir = [IO.Path]::GetFullPath((Join-Path $DistRoot "ModuleToFlashcards"))
$SpecPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "ModuleToFlashcards.spec"))
Set-Location -LiteralPath $RepoRoot

function Assert-ChildPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Parent,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $resolvedPath = [IO.Path]::GetFullPath($Path)
    $resolvedParent = [IO.Path]::GetFullPath($Parent).TrimEnd([IO.Path]::DirectorySeparatorChar)
    $prefix = $resolvedParent + [IO.Path]::DirectorySeparatorChar
    if (-not $resolvedPath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label must be inside $resolvedParent; received $resolvedPath"
    }
}

Assert-ChildPath -Path $BuildRoot -Parent $RepoRoot -Label "Build directory"
Assert-ChildPath -Path $DistRoot -Parent $RepoRoot -Label "Distribution directory"
Assert-ChildPath -Path $AssetsRoot -Parent $RepoRoot -Label "Asset directory"
Assert-ChildPath -Path $FrozenDist -Parent $BuildRoot -Label "PyInstaller distribution"
Assert-ChildPath -Path $PyInstallerWork -Parent $BuildRoot -Label "PyInstaller work directory"
Assert-ChildPath -Path $BundleDir -Parent $DistRoot -Label "Portable bundle"

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        $PythonExe = $venvPython
    }
    else {
        $PythonExe = (Get-Command python -ErrorAction Stop).Source
    }
}
$PythonExe = [IO.Path]::GetFullPath($PythonExe)
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable not found: $PythonExe"
}

New-Item -ItemType Directory -Force -Path $BuildRoot, $DistRoot | Out-Null

& $PythonExe -c "import fastapi, huggingface_hub, llama_cpp, PIL, pymupdf, pytesseract, sentencepiece, torch, transformers, uvicorn"
if ($LASTEXITCODE -ne 0) {
    throw "Runtime dependency preflight failed. Install requirements.txt before building."
}

if (-not $SkipGpuPreflight) {
    & $PythonExe -c "import llama_cpp, torch; ok = torch.cuda.is_available() and llama_cpp.llama_supports_gpu_offload(); print('CUDA preflight:', ok); raise SystemExit(0 if ok else 1)"
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA preflight failed. Build the release in a CUDA-enabled environment or use -SkipGpuPreflight only for configuration smoke checks."
    }
}

if (-not $SkipAssetPreparation) {
    if (-not (Test-Path -LiteralPath $TesseractDir -PathType Container)) {
        throw "Tesseract directory not found: $TesseractDir"
    }
    & $PythonExe (Join-Path $PSScriptRoot "prepare_assets.py") --repo-root $RepoRoot --tesseract-dir $TesseractDir
    if ($LASTEXITCODE -ne 0) {
        throw "Locked asset preparation failed."
    }
}

& $PythonExe -m PyInstaller --noconfirm --clean --distpath $FrozenDist --workpath $PyInstallerWork $SpecPath
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller freeze failed."
}

$FrozenApp = Join-Path $FrozenDist "ModuleToFlashcards"
& $PythonExe (Join-Path $PSScriptRoot "assemble_portable.py") --repo-root $RepoRoot --frozen-app $FrozenApp --assets $AssetsRoot --output-dir $BundleDir
if ($LASTEXITCODE -ne 0) {
    throw "Portable bundle assembly failed."
}

& (Join-Path $BundleDir "ModuleToFlashcards.exe") --verify
if ($LASTEXITCODE -ne 0) {
    throw "Frozen bundle verification failed."
}

$Version = (& $PythonExe -c "from version import __version__; print(__version__)" | Select-Object -Last 1).Trim()
$Archive = [IO.Path]::GetFullPath((Join-Path $DistRoot "ModuleToFlashcards-$Version-windows-x64.zip"))
Assert-ChildPath -Path $Archive -Parent $DistRoot -Label "Release archive"
if (Test-Path -LiteralPath $Archive -PathType Leaf) {
    Remove-Item -LiteralPath $Archive -Force
}
Compress-Archive -LiteralPath $BundleDir -DestinationPath $Archive -CompressionLevel Optimal

Write-Host "Portable release: $BundleDir"
Write-Host "Archive: $Archive"
