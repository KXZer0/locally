# start.ps1 — locally launcher
# Runs the venv's Python directly. locally.py prints its own
# device-detection, per-model loading progress, and the "locally ready"
# banner with the URL — the launcher does not poll /health or auto-open
# a browser. Connect a client to the API URL in the banner.
#
# Args are set by install.ps1 in the generated start.ps1.

param(
    [string]$ServerArgs = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Direct invocation works in Windows PowerShell 5.1 and PowerShell 7.
$PythonPath = if ($env:OS -eq 'Windows_NT') { Join-Path $ScriptDir 'venv/Scripts/python.exe' } else { Join-Path $ScriptDir 'venv/bin/python' }
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Run install.ps1 first to create the server environment.' }

$AllArgs = @((Join-Path $ScriptDir "locally.py"))
if ($ServerArgs) {
    $AllArgs += $ServerArgs.Split(" ", [StringSplitOptions]::RemoveEmptyEntries)
}
if ($ExtraArgs) {
    $AllArgs += $ExtraArgs  # user overrides from the start.ps1 command line, e.g. --port 8091
}

& $PythonPath @AllArgs
exit $LASTEXITCODE
