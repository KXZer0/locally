#requires -Version 7.0
# start.ps1 — locally launcher
# Activates the venv and runs locally.py. locally.py prints its own
# device-detection, per-model loading progress, and the "locally ready"
# banner with the URL — the launcher does not poll /health or auto-open
# the browser. Open the URL from the banner yourself.
#
# Args are set by install.ps1 in the generated start.ps1.

param(
    [string]$ServerArgs = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Activate venv (Scripts on Windows, bin on POSIX)
$VenvBinDir = if ($IsWindows) { "Scripts" } else { "bin" }
& (Join-Path $ScriptDir "venv" $VenvBinDir "Activate.ps1")

$AllArgs = @((Join-Path $ScriptDir "locally.py"))
if ($ServerArgs) {
    $AllArgs += $ServerArgs.Split(" ", [StringSplitOptions]::RemoveEmptyEntries)
}
if ($ExtraArgs) {
    $AllArgs += $ExtraArgs  # user overrides from the start.ps1 command line, e.g. --port 8091
}

& python @AllArgs
