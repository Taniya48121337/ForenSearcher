$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$DefaultCodexPython = 'C:\Users\Zya\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$Python = if ($env:CLAIMUNIT_PYTHON) {
    $env:CLAIMUNIT_PYTHON
} elseif (Test-Path $DefaultCodexPython) {
    $DefaultCodexPython
} else {
    (Get-Command python -ErrorAction Stop).Source
}

$Unedited = Join-Path $ProjectRoot 'data\source\1004_unedited.mp4'
$Processed = Join-Path $ProjectRoot 'data\edited\1004_processed.mp4'
$Extractor = Join-Path $ProjectRoot 'tools\remote_zip_extract.py'

if (-not (Test-Path $Unedited)) {
    & $Python $Extractor 'https://github.com/adobe-research/VideoSham-dataset/releases/download/v1.0/Unedited-Part1.zip' --match '(^|/)1004_unedited\.mp4$' --output-dir (Split-Path -Parent $Unedited)
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
if (-not (Test-Path $Processed)) {
    & $Python $Extractor 'https://github.com/adobe-research/VideoSham-dataset/releases/download/v1.0/Processed-Part1.zip' --match '(^|/)1004_processed\.mp4$' --output-dir (Split-Path -Parent $Processed)
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$Vendor = Join-Path $ProjectRoot 'vendor'
if (-not (Test-Path (Join-Path $Vendor 'av\__init__.py'))) {
    & $Python -m pip install --disable-pip-version-check --target $Vendor 'av==15.1.0'
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& (Join-Path $ProjectRoot 'run.ps1')
exit $LASTEXITCODE
