[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [string]$TesseractDir = "",
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

& $PythonExe -c "import fastapi, multipart, pymupdf, uvicorn"
if ($LASTEXITCODE -ne 0) {
    throw "Runtime dependency preflight failed. Install requirements.txt before building."
}

Write-Host "Building the Mistral API client; no local models or CUDA assets are needed."
if ([string]::IsNullOrWhiteSpace($TesseractDir)) {
    $ocrCommand = Get-Command tesseract.exe -ErrorAction SilentlyContinue
    if ($ocrCommand) { $TesseractDir = Split-Path -Parent $ocrCommand.Source }
    elseif ($env:TESSERACT_CMD) { $TesseractDir = Split-Path -Parent $env:TESSERACT_CMD }
    else { $TesseractDir = Join-Path $env:ProgramFiles "Tesseract-OCR" }
}
if (-not (Test-Path -LiteralPath (Join-Path $TesseractDir "tesseract.exe"))) {
    throw "Install Tesseract OCR or pass -TesseractDir before building the portable app."
}

& $PythonExe -m PyInstaller --noconfirm --clean --distpath $FrozenDist --workpath $PyInstallerWork $SpecPath
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller freeze failed."
}

$FrozenApp = Join-Path $FrozenDist "ModuleToFlashcards"
& $PythonExe (Join-Path $PSScriptRoot "assemble_portable.py") --repo-root $RepoRoot --frozen-app $FrozenApp --output-dir $BundleDir --tesseract-dir $TesseractDir
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
& $PythonExe (Join-Path $PSScriptRoot "create_archive.py") --source-dir $BundleDir --output $Archive
if ($LASTEXITCODE -ne 0) {
    throw "Portable archive creation failed."
}

Write-Host "Portable release: $BundleDir"
Write-Host "Archive: $Archive"
