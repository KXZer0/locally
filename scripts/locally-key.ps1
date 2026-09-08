# Hardware-key hot path: start or inspect the API by default, with chat as a
# selectable mode for users who want the key to open a prompt instead.
#
# There is no window to raise any more -- locally is a terminal program -- so
# The script first checks whether a server is ALREADY listening. API mode
# starts the configured foreground launcher only when needed. Chat mode
# attaches to a listener or starts a server that chat session owns.
#
# Started with --start the server takes the default model/ link and picks its
# own device. A configured server (install.ps1's start.ps1, with its device
# and idle-timeout flags) should be started that way once; the key then finds
# it listening and attaches, which is the faster path anyway.
#
# Runs under Windows PowerShell 5.1 from System32, which is what locally.lnk
# uses and what Smart App Control never looks twice at (TODONT.md).
param(
    [ValidateSet('chat', 'api')]
    [string]$Mode = 'api',
    [int]$Port = 8000,
    # Prints the selected action without opening a window. Used by tests and
    # handy when checking a key-remapper command.
    [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = if ($env:LOCALLY_PYTHON) { $env:LOCALLY_PYTHON } elseif ($env:OS -eq 'Windows_NT') { Join-Path $RepoRoot 'venv\Scripts\python.exe' } else { Join-Path $RepoRoot 'venv/bin/python' }
$Entry = Join-Path $RepoRoot 'locally.py'

# A TCP connect to loopback answers in about a millisecond. On this machine a
# connection to a CLOSED loopback port is dropped rather than refused, so the
# wait is bounded explicitly instead of relying on a fast failure.
$Client = [System.Net.Sockets.TcpClient]::new()
try { $Serving = $Client.ConnectAsync('127.0.0.1', $Port).Wait(250) -and $Client.Connected }
catch { $Serving = $false }
finally { $Client.Dispose() }

$Quote = { param($Value) "'" + ($Value -replace "'", "''") + "'" }
$Action = $null
if ($Mode -eq 'chat') {
    $Inner = "& $(& $Quote $Python) $(& $Quote $Entry) chat --port $Port"
    if ($Serving) { $Action = 'attach-chat' }
    else { $Inner += ' --start'; $Action = 'start-owned-chat' }
} else {
    if ($Serving) {
        $Inner = "& $(& $Quote $Python) $(& $Quote $Entry) status --port $Port"
        $Action = 'show-api-status'
    } else {
        $Launcher = Join-Path $RepoRoot 'api.ps1'
        $Inner = "& $(& $Quote $Launcher) -Port $Port"
        $Action = 'start-api'
    }
}

if ($WhatIf) {
    [pscustomobject]@{ Mode = $Mode; Port = $Port; Serving = $Serving; Action = $Action; Command = $Inner } |
        ConvertTo-Json -Compress
    return
}

# The key gives no console to inherit, so open one. -NoExit keeps the window
# up after the client exits, so a startup error stays readable.
if (-not (Test-Path -LiteralPath $Python)) { throw 'Run install.ps1 first to create the server environment.' }
$Shell = Get-Command pwsh -ErrorAction SilentlyContinue
if (-not $Shell) { $Shell = Get-Command powershell -ErrorAction SilentlyContinue }
if (-not $Shell) { throw 'No PowerShell host was found to open the locally terminal.' }
$Encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Inner))
Start-Process -FilePath $Shell.Source -WindowStyle Normal `
    -ArgumentList '-NoProfile', '-NoExit', '-EncodedCommand', $Encoded
