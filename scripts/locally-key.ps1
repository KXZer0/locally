<#
.SYNOPSIS
    Hot path for the locally hardware key — raise the window, open one, or
    hand off to the launcher.

.DESCRIPTION
    Replaces locallyKey.exe, which Smart App Control blocks (2026-08-17: SAC
    left evaluation mode and started enforcing, so an unsigned locally-built
    binary stopped loading — see TODONT.md). A self-signed certificate cannot
    fix that; SAC judges against the Microsoft trusted root program and ignores
    local trust stores. So the answer is to ship no binary at all and let a
    Microsoft-signed host run the logic.

    THREE CASES, and only the last one is allowed to be slow:

      window up + server up   raise it and exit                (~50 ms of work)
      window gone + server up open the app window RIGHT HERE   (no pwsh spawn)
      server down             hand off to locally-launch.ps1   (it must boot a model)

    The middle case is the one that was costing ~3-4 s. This script used to
    treat "no window" as a cold start and hand off to locally-launch.ps1,
    which meant spawning pwsh 7 (~235-600 ms) and paying that script's
    Add-Type (~336 ms) purely to run a single Start-Process. Closing the app
    window is the normal way to put locally away, so this was the common
    case, not the rare one. %LOCALAPPDATA%\locally\launch.log showed every
    recent press logging "server already running: True" — i.e. the entire
    launcher ran to do nothing but open a browser.

    The Win32 calls live in locally-win32.ps1, shared with the launcher.
    They used to be duplicated, which is how locally-launch.ps1 kept its
    Add-Type long after this script stopped using one.

    Run with Windows PowerShell 5.1, from System32:
      %SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe
    It is Microsoft-signed and in System32, so SAC never looks twice. (It is
    no longer *required* — the shared helper now emits on pwsh 7 too — but
    5.1 needs no separate install and this is the path locally.lnk uses.)

.EXAMPLE
    powershell.exe -NoProfile -WindowStyle Hidden -File scripts\locally-key.ps1
#>
param(
    [int] $Port = 8000,
    # Anything else is forwarded verbatim to locally-launch.ps1 on the cold path.
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Rest = @()
)

$ErrorActionPreference = 'Stop'
$sw = [Diagnostics.Stopwatch]::StartNew()

# A hardware key gives no console to read, so leave a trail — the same log
# locally-launch.ps1 writes. Without this, "the key felt slow" is unfalsifiable.
# [IO.*] rather than Test-Path/Add-Content throughout: the first Test-Path in
# a PS 5.1 session costs 144 ms of provider init (measured; later calls are
# free), and Add-Content costs ~41 ms. Both are cmdlet overhead, not I/O, and
# this script's whole budget is a few hundred milliseconds.
$logDir = Join-Path $env:LOCALAPPDATA 'locally'
if (-not [IO.Directory]::Exists($logDir)) { [void][IO.Directory]::CreateDirectory($logDir) }
$keyLog = Join-Path $logDir 'launch.log'
function Write-Log($msg) {
    [IO.File]::AppendAllText($keyLog,
        "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  key: $msg ($($sw.ElapsedMilliseconds) ms)$([Environment]::NewLine)")
}

$win32 = Join-Path $PSScriptRoot 'locally-win32.ps1'
if (-not [IO.File]::Exists($win32)) {
    [Console]::Error.WriteLine("locally-win32.ps1 not found next to this script")
    exit 2
}
. $win32

# A TCP connect to localhost answers in about a millisecond. A visible window
# is NOT evidence of a live server — a stale window over a dead one is exactly
# the "failed to fetch" case, and it has to fall through to the cold path.
function Test-Up {
    $c = [System.Net.Sockets.TcpClient]::new()
    try { return $c.ConnectAsync('127.0.0.1', $Port).Wait(250) -and $c.Connected }
    catch { return $false } finally { $c.Dispose() }
}

$serverUp = Test-Up
# Raise whatever is there even if the server died — that is still where the
# user wants to be looking, and the page will show its own error.
$raised   = Show-locallyWindow

# ---- Case 1: everything already up. Nothing to do. ----
if ($raised -and $serverUp) {
    Write-Log "raised existing window (pid $($raised.Id))"
    exit 0
}

# ---- Case 2: server is fine, the window was just closed. ----
# Open it here rather than waking pwsh 7 and the whole launcher for one
# Start-Process. Only the app-mode path is inlined; anything more exotic
# (no Chromium installed) still deserves the launcher's fallback chain.
if ($serverUp -and -not $raised) {
    $opened = Open-locallyApp "http://127.0.0.1:$Port"
    if ($opened) {
        Write-Log "opened $opened, no launcher"
        exit 0
    } else {
        Write-Log "installed PWA / Chromium app mode failed; handing off for fallbacks"
    }
}

# ---- Case 3: cold path. The server needs starting (or app mode failed). ----
# locally-launch.ps1 is #requires -Version 7.0, so it needs pwsh — the
# WindowsApps alias rather than whatever Get-Command resolves to, since a Store
# install pins the version in its path and breaks on the next update.
$script = Join-Path $PSScriptRoot 'locally-launch.ps1'
if (-not [IO.File]::Exists($script)) {
    [Console]::Error.WriteLine("locally-launch.ps1 not found next to this script")
    exit 2
}
$pwsh = "$env:LOCALAPPDATA\Microsoft\WindowsApps\pwsh.exe"
if (-not [IO.File]::Exists($pwsh)) {
    $pwsh = (Get-Command pwsh.exe -ErrorAction SilentlyContinue).Source
}
if (-not $pwsh) {
    [Console]::Error.WriteLine("pwsh.exe not found; locally-launch.ps1 requires PowerShell 7")
    exit 3
}

# One quoted string rather than an array: Start-Process puts ValidateNotNullOrEmpty
# on each ArgumentList element, so a single empty argument fails the whole launch
# — and `-NpuModel ''` is a documented, meaningful value to locally-launch.ps1
# ("falls back to the GPU model below"). The old exe was no better here; joining
# on spaces dropped the empty argument silently instead of erroring.
function Quote($s) {
    if ($null -eq $s -or $s -eq '' -or $s -match '[\s"]') {
        return '"' + ($s -replace '"', '\"') + '"'
    }
    return $s
}
$launchArgs = (@('-NoProfile', '-NonInteractive', '-WindowStyle', 'Hidden',
                 '-File', $script, '-Port', $Port) + $Rest |
                ForEach-Object { Quote $_ }) -join ' '
try {
    Start-Process -FilePath $pwsh -ArgumentList $launchArgs -WindowStyle Hidden
    Write-Log "server down; handed off to locally-launch.ps1"
} catch {
    [Console]::Error.WriteLine("failed to launch: $($_.Exception.Message)")
    exit 3
}
exit 0
