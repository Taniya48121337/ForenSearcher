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
& $Python (Join-Path $ProjectRoot 'src\cli_teacher_rollout.py') --promote
exit $LASTEXITCODE
