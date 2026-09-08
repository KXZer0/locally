<# Start the configured API in this terminal. Ctrl+C stops it. #>
param(
    [int]$Port = 8000,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = 'Stop'
$start = if ($env:LOCALLY_START_SCRIPT) { $env:LOCALLY_START_SCRIPT } else { Join-Path $PSScriptRoot 'start.ps1' }
if (-not (Test-Path -LiteralPath $start)) {
    throw 'No start.ps1 was found. Run install.ps1 to configure the API launcher.'
}
# `--port` works with both installer-generated start.ps1 files (which forward
# server arguments) and this checkout's start.ps1 (which delegates to the
# configured launcher). The last argparse value wins in either case.
& $start '--port' "$Port" @ExtraArgs
exit $LASTEXITCODE
