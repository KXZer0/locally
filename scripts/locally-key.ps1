# Hardware-key hot path: put a chat prompt in front of the user.
#
# There is no window to raise any more -- locally is a terminal program -- so
# the only question this script answers is whether a server is ALREADY
# listening. If one is, attach to it: starting a second one wants the same
# port, and locally exits with "Port 8000 is already in use" into a window the
# user has to read to find that out. If nothing is there, start a server this
# chat session owns, which stops again when the session ends.
#
# Started with --start the server takes the default model/ link and picks its
# own device. A configured server (install.ps1's start.ps1, with its device
# and idle-timeout flags) should be started that way once; the key then finds
# it listening and attaches, which is the faster path anyway.
#
# Runs under Windows PowerShell 5.1 from System32, which is what locally.lnk
# uses and what Smart App Control never looks twice at (TODONT.md).
param([int]$Port = 8000)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = if ($env:OS -eq 'Windows_NT') { Join-Path $RepoRoot 'venv\Scripts\python.exe' } else { Join-Path $RepoRoot 'venv/bin/python' }
if (-not (Test-Path -LiteralPath $Python)) { throw 'Run install.ps1 first to create the server environment.' }
$Entry = Join-Path $RepoRoot 'locally.py'

# A TCP connect to loopback answers in about a millisecond. On this machine a
# connection to a CLOSED loopback port is dropped rather than refused, so the
# wait is bounded explicitly instead of relying on a fast failure.
$Client = [System.Net.Sockets.TcpClient]::new()
try { $Serving = $Client.ConnectAsync('127.0.0.1', $Port).Wait(250) -and $Client.Connected }
catch { $Serving = $false }
finally { $Client.Dispose() }

$Quote = { param($Value) "'" + ($Value -replace "'", "''") + "'" }
$Inner = "& $(& $Quote $Python) $(& $Quote $Entry) chat --port $Port"
if (-not $Serving) { $Inner += ' --start' }

# The key gives no console to inherit, so open one. -NoExit keeps the window
# up after the client exits, so a startup error stays readable.
$Shell = Get-Command pwsh -ErrorAction SilentlyContinue
if (-not $Shell) { $Shell = Get-Command powershell -ErrorAction SilentlyContinue }
if (-not $Shell) { throw 'No PowerShell host was found to open the locally terminal.' }
Start-Process -FilePath $Shell.Source -WindowStyle Normal `
    -ArgumentList "-NoProfile -NoExit -Command `"$Inner`""
