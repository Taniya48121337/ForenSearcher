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
$env:PYTHONPATH = Join-Path $ProjectRoot 'vendor'
& $Python (Join-Path $ProjectRoot 'src\pipeline.py') --validate
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python (Join-Path $ProjectRoot 'src\trajectory_rubric.py')
exit $LASTEXITCODE
