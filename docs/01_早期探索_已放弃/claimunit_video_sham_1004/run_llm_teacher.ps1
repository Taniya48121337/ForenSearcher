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
$KeyWasPrompted = $false
if ([string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) {
    $SecureKey = Read-Host 'OpenAI API key (input is hidden)' -AsSecureString
    $env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new('', $SecureKey).Password
    $KeyWasPrompted = $true
}

$TeacherExitCode = 1
try {
    & $Python (Join-Path $ProjectRoot 'src\llm_teacher_rollout.py') --promote
    $TeacherExitCode = $LASTEXITCODE
} finally {
    if ($KeyWasPrompted) {
        Remove-Item Env:\OPENAI_API_KEY -ErrorAction SilentlyContinue
    }
}
exit $TeacherExitCode
