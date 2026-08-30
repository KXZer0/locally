#requires -Version 7.0
<#
.SYNOPSIS
    Start locally in proxy mode against a local Ollama — the Machine B setup.

.DESCRIPTION
    locally is OpenVINO, and OpenVINO's GPU plugin is Intel-only: detect_devices()
    filters non-Intel GPUs out, because the plugin enumerates any OpenCL device
    but its kernels only run on Intel silicon. So on a box with an NVIDIA card
    there is no device to place a generative model on, and everything else the
    app is — the utilities, the memory HUD, the voice path, the whole web UI —
    was unreachable there for want of a chat model.

    --proxy-url fixes that by pointing the primary slot at another
    OpenAI-compatible server. Ollama drives the 4070 with CUDA; locally relays.
    One program, one URL, one UI on both machines.

    This script starts that arrangement. It checks Ollama first, because the
    failure it prevents is the one that wastes the most time: locally's
    ProxySlot probes the upstream at load and refuses to start without it, but
    it does so 20 lines into a startup log you may not be watching.

    Companion doc: docs/ODYSSEUS.md.

.EXAMPLE
    .\scripts\locally-proxy.ps1

.EXAMPLE
    .\scripts\locally-proxy.ps1 -ProxyModel 'qwen3-coder:30b' -Port 8000

.EXAMPLE
    # Ollama on another box on the tailnet.
    .\scripts\locally-proxy.ps1 -ProxyUrl 'http://desktop.tailXXXX.ts.net:11434'
#>
param(
    # The upstream ROOT, not a /v1 path. ProxySlot appends /v1/models and
    # /v1/chat/completions itself, so a trailing /v1 here gives you a 404 on
    # /v1/v1/models — which reads as "wrong model" rather than "wrong URL".
    #
    # Note this is the opposite convention from what you give ODYSSEUS, which
    # does want the /v1 suffix. Different layers.
    [string] $ProxyUrl = 'http://localhost:11434',

    # Optional. Left empty, ProxySlot takes the first model the upstream
    # advertises — which is what a single-model Ollama host means by "the
    # model". Named explicitly, it is validated at startup against the
    # upstream's real list and the error names what IS available, so a typo
    # costs seconds rather than a debugging session.
    [string] $ProxyModel = '',

    # Bearer token for the upstream. Ollama needs none; this exists for a vLLM
    # or llama.cpp behind auth. Leave empty unless the upstream asks.
    [string] $ProxyKey = '',

    # locally's OpenAI API. This is the port Odysseus points at:
    #     http://host.docker.internal:8000/v1
    [int] $Port = 8000,

    # 0 disables locally's own Ollama-compatible shim, and on this machine that
    # is not optional: the REAL Ollama owns 11434. Two things claiming to be
    # Ollama on one port means locally fails to bind and warns, and whichever
    # client you point at 11434 gets whichever one won the race.
    #
    # Set this to a free port (e.g. 11435) only if you specifically want
    # locally's shim as well — see docs/ODYSSEUS.md sec. 4.
    [int] $OllamaPort = 0,

    # 0 = never unload. Right for an assistant: it is queried in bursts all day,
    # and an idle unload puts a cold reload inside the user's first message
    # after every quiet stretch. On a proxy slot the unload is a no-op anyway
    # (this process holds no weights), but the flag also governs the utility
    # and audio slots, and 0 auto-enables --prewarm.
    [int] $IdleTimeout = 0,

    # Utility models. 'auto' is right here and needs no thought: with no Intel
    # accelerator on this machine it falls back to the CPU, which OpenVINO runs
    # on any x86. That gets OCR, layout, upscale, matting, detect and reranked
    # search on a box whose GPU locally cannot touch — and those utilities are
    # the reason to run locally here at all, since Ollama is doing the
    # generating. Slower than the NPU, fast enough to read a document.
    [ValidateSet('auto', 'npu', 'gpu', 'cpu', 'npu,gpu')]
    [string] $UtilEngines = 'auto',

    # Whisper / Kokoro / Silero run fine on a plain CPU via OpenVINO — these
    # are not Intel-accelerator-only paths. Passed only when the directory
    # exists, so a missing model degrades to "that tab is off", not a crash.
    [string] $Whisper = '',
    [string] $Tts = '',
    [string] $Vad = '',
    [string] $WhisperDevice = 'CPU',

    # Trim a client's tool list server-side, keeping only these names. Not
    # needed for Odysseus's assistant tools against a proxy slot — a REMOTE
    # slot has no schema budget, that ceiling is an NPU constraint — but this
    # is the escape hatch when a client sends thirty schemas and prefill is the
    # whole wait. Try "Bash,Edit,Read,Write,Glob,Grep".
    [string] $AgentTools = '',

    # Print the command and exit without starting anything.
    [switch] $WhatIfOnly,

    # Run in this console instead of a hidden window, so you can watch the
    # startup probe and Ctrl+C out.
    [switch] $Foreground
)

$ErrorActionPreference = 'Stop'

$root = Split-Path $PSScriptRoot -Parent
$python = if ($IsWindows) {
    Join-Path $root 'venv\Scripts\python.exe'
} else {
    Join-Path $root 'venv/bin/python'
}
if (-not (Test-Path $python)) {
    throw "venv python not found at $python. Create the venv first (see README)."
}

# -----------------------------------------------------------------------------
# Is Ollama actually there?
# -----------------------------------------------------------------------------
# Two checks, deliberately, because they fail differently and the fixes are
# different. A TCP connect answers "is anything listening" in microseconds and
# cannot be confused by a slow model load; the HTTP call answers "is it Ollama,
# and what does it hold". Doing only the first would let you start against a
# port owned by something else entirely.

$uri = [System.Uri] $ProxyUrl
$probeHost = $uri.Host
$probePort = if ($uri.Port -gt 0) { $uri.Port } else { 11434 }

function Test-PortOpen($h, $p, $ms = 800) {
    $c = [System.Net.Sockets.TcpClient]::new()
    try {
        return $c.ConnectAsync($h, $p).Wait($ms) -and $c.Connected
    } catch { return $false } finally { $c.Dispose() }
}

Write-Host "Checking upstream at $ProxyUrl ... " -NoNewline

if (-not (Test-PortOpen $probeHost $probePort)) {
    Write-Host "nothing listening." -ForegroundColor Red
    Write-Host ""
    Write-Host "  Nothing is listening on ${probeHost}:${probePort}, so locally would refuse"
    Write-Host "  to start: ProxySlot.load() probes the upstream and fails with"
    Write-Host "  'proxy: cannot reach $ProxyUrl ... Is the server running?'"
    Write-Host ""
    Write-Host "  Start Ollama, then run this again:" -ForegroundColor Yellow
    if ($IsWindows) {
        Write-Host "      ollama serve            # or start the Ollama service / tray app"
    } else {
        Write-Host "      ollama serve            # or: systemctl --user start ollama"
    }
    Write-Host ""
    Write-Host "  If Ollama is on ANOTHER machine, it must listen outside its own"
    Write-Host "  loopback and you must say so here:"
    Write-Host "      OLLAMA_HOST=0.0.0.0:11434 ollama serve"
    Write-Host "      .\scripts\locally-proxy.ps1 -ProxyUrl 'http://<host>:11434'"
    Write-Host ""
    Write-Host "  If you meant a local model on Intel silicon instead of a proxy,"
    Write-Host "  this is the wrong script — use scripts\locally-launch.ps1."
    exit 1
}

# Reachable. Now find out what it holds, so a --proxy-model typo is caught here
# — with the real list in front of you — instead of in a startup log.
$upstreamModels = @()
try {
    $tags = Invoke-RestMethod -Uri "$($ProxyUrl.TrimEnd('/'))/api/tags" -TimeoutSec 8
    $upstreamModels = @($tags.models | ForEach-Object { $_.name })
    Write-Host "ok." -ForegroundColor Green
} catch {
    # Not fatal. Something is listening and it may well be a vLLM or llama.cpp
    # that has no /api/tags — ProxySlot probes /v1/models, which is the shape
    # that actually matters. Say so and carry on rather than blocking.
    Write-Host "reachable, but not Ollama's /api/tags." -ForegroundColor Yellow
    Write-Host "  ($($_.Exception.Message))"
    Write-Host "  Fine if the upstream is vLLM or llama.cpp — locally probes"
    Write-Host "  /v1/models, not /api/tags. Continuing."
}

if ($upstreamModels.Count -gt 0) {
    Write-Host "  Upstream models: $($upstreamModels -join ', ')"
    if ($ProxyModel -and ($upstreamModels -notcontains $ProxyModel)) {
        Write-Host ""
        Write-Host "  '$ProxyModel' is not one of them." -ForegroundColor Red
        Write-Host "  locally would refuse at startup with the same list. Either pull it:"
        Write-Host "      ollama pull $ProxyModel"
        Write-Host "  or pass -ProxyModel with one of the names above, or drop the"
        Write-Host "  flag entirely to take the first the upstream advertises."
        exit 1
    }
    if (-not $ProxyModel) {
        Write-Host "  No -ProxyModel given; locally will take '$($upstreamModels[0])'."
    }
} elseif (-not $ProxyModel) {
    # ProxySlot errors out on this exact case, so warn before spending the
    # startup time: "upstream advertises no models and --proxy-model was not given".
    Write-Host "  Upstream advertised no models and no -ProxyModel was given." -ForegroundColor Yellow
    Write-Host "  locally will refuse to start. Pass -ProxyModel explicitly."
}

# -----------------------------------------------------------------------------
# Port 8000 free?
# -----------------------------------------------------------------------------
if (Test-PortOpen '127.0.0.1' $Port 400) {
    Write-Host ""
    Write-Host "Something already owns 127.0.0.1:$Port." -ForegroundColor Yellow
    Write-Host "  Probably locally is already running — check http://127.0.0.1:$Port/health"
    Write-Host "  Pass -Port to use a different one (and update Odysseus's endpoint)."
    exit 1
}

# -----------------------------------------------------------------------------
# Build the command
# -----------------------------------------------------------------------------
$serverArgs = @(
    (Join-Path $root 'locally.py')
    '--proxy-url'; $ProxyUrl
    '--port'; "$Port"
    # 0 keeps the real Ollama's 11434 to itself. See the param comment.
    '--ollama-port'; "$OllamaPort"
    '--idle-timeout'; "$IdleTimeout"
    '--util-engines'; $UtilEngines
)
# --model-dir is deliberately NOT passed: locally skips the model-directory
# check entirely when --proxy-url is set, and naming a directory that holds no
# OpenVINO IR would only be misleading in the logs.

if ($ProxyModel) { $serverArgs += '--proxy-model'; $serverArgs += $ProxyModel }
if ($ProxyKey)   { $serverArgs += '--proxy-key';   $serverArgs += $ProxyKey }
if ($AgentTools) { $serverArgs += '--agent-tools'; $serverArgs += $AgentTools }

if ($Whisper -and (Test-Path $Whisper)) {
    $serverArgs += '--whisper-dir'; $serverArgs += $Whisper
    $serverArgs += '--whisper-device'; $serverArgs += $WhisperDevice
}
if ($Tts -and (Test-Path $Tts)) {
    $serverArgs += '--tts-dir'; $serverArgs += $Tts
    $serverArgs += '--tts-device'; $serverArgs += 'CPU'
}
if ($Vad -and (Test-Path $Vad)) {
    $serverArgs += '--vad-dir'; $serverArgs += $Vad
    $serverArgs += '--vad-device'; $serverArgs += 'CPU'
}

# The upstream key must not land in a log or a console scrollback.
$shown = $serverArgs -join ' '
if ($ProxyKey) { $shown = $shown.Replace($ProxyKey, '<redacted>') }

Write-Host ""
Write-Host "  $python $shown"
Write-Host ""

if ($WhatIfOnly) { return }

# -----------------------------------------------------------------------------
# Go
# -----------------------------------------------------------------------------
if ($Foreground) {
    & $python @serverArgs
    return
}

$logDir = if ($IsWindows) {
    Join-Path $env:LOCALAPPDATA 'locally'
} else {
    Join-Path $HOME '.local/share/locally'
}
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$startArgs = @{
    FilePath               = $python
    ArgumentList           = $serverArgs
    WorkingDirectory       = $root
    RedirectStandardOutput = (Join-Path $logDir 'proxy-server.log')
    RedirectStandardError  = (Join-Path $logDir 'proxy-server.err')
}
# -WindowStyle is a Windows-only parameter; passing it on Linux/macOS throws.
if ($IsWindows) { $startArgs.WindowStyle = 'Hidden' }

Start-Process @startArgs | Out-Null

# Wait for the socket to bind, not for the upstream probe to finish. Flask
# listens within a couple of seconds while ProxySlot probes and warms on a
# background thread, and /health reports per-slot state — so reporting early
# gives feedback instead of a blank stare.
$sw = [Diagnostics.Stopwatch]::StartNew()
while ($sw.ElapsedMilliseconds -lt 30000 -and -not (Test-PortOpen '127.0.0.1' $Port 300)) {
    Start-Sleep -Milliseconds 100
}

if (Test-PortOpen '127.0.0.1' $Port 400) {
    Write-Host "locally is up on http://127.0.0.1:$Port  (bound after $($sw.ElapsedMilliseconds) ms)" -ForegroundColor Green
    Write-Host "  Logs:   $logDir"
    Write-Host "  Health: http://127.0.0.1:$Port/health"
    Write-Host "  Models: http://127.0.0.1:$Port/v1/models"
    Write-Host ""
    Write-Host "  The proxy slot warms with one real token before it reports ready;"
    Write-Host "  /v1/models stays empty until then. A proxy slot advertises as"
    Write-Host "  '<model>@REMOTE' — give Odysseus that exact id."
    Write-Host ""
    Write-Host "  Point Odysseus at:  http://host.docker.internal:$Port/v1" -ForegroundColor Cyan
    Write-Host "  (host.docker.internal, NOT localhost — see docs/ODYSSEUS.md sec. 7)"
} else {
    Write-Host "Port $Port never bound within 30s. Check $logDir\proxy-server.err" -ForegroundColor Red
    exit 1
}
