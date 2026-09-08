<#
.SYNOPSIS
    Foreground API launcher: this machine's configuration, in one place.
    Ctrl+C stops the server and releases its models.

.DESCRIPTION
    Not a browser launcher any more, and it raises no window: locally is a
    terminal program. What it still is -- and what very nearly went out with
    the web UI -- is the CONFIGURATION. Which model lands on which device,
    which voice models load and where, the KV precision and context budget a
    27B needs in order to fit, and the SearXNG port that keeps locally's
    search independent of Odysseus's are all measured decisions (CLAUDE.md,
    TODONT.md). A bare `python locally.py` starts a server that runs and is
    configured differently: no ASR, no TTS, no VAD, f16 KV, a 2 GB pool.

    Runs on Windows PowerShell 5.1 and PowerShell 7 alike. Anything after the
    named parameters is forwarded to locally.py verbatim, and argparse takes
    the last flag, so extra arguments override what is set here.

.EXAMPLE
    .\scripts\locally-launch.ps1
    .\scripts\locally-launch.ps1 -IdleTimeout 0 --no-prompt-cache
#>
param(
    # ONE chat model at a time, Ollama-style. The NPU has no memory of its own
    # -- it allocates from the same RAM as the GPU -- so a second resident
    # model costs real RAM and heat for a model you cannot talk to
    # concurrently anyway. Swap with POST /v1/models/load or /load in terminal
    # chat, which unloads the old one first, so peak memory stays one model.
    #
    # Placement is the SERVER's call (--device auto): _choose_device prefers
    # the NPU, and rules it out for vision and for group-quantized int4 by
    # reading the IR's rt_info -- which a shell script cannot do, and which is
    # why naming a device here was only ever a guess about a file.
    #
    # A PREFERENCE ORDER, not a requirement: the first name that is still on
    # disk wins, and if none of them are, the launcher asks locally.py what IS
    # there (--list-models) rather than insisting on a model you deleted.
    # Names are resolved under -ModelsDir; an absolute path also works.
    [string[]] $ChatModels = @('Qwen3-8B-abliterated-int4-cw-ov',
                               'Qwen3-8B-int4-cw-ov'),
    [string] $ModelsDir = "$env:USERPROFILE\models",
    [string] $Whisper   = "$env:USERPROFILE\models\whisper-small-int8-ov",
    [string] $Tts       = "$env:USERPROFILE\models\Kokoro-82M-int8-ov",
    [string] $Speaker   = "$env:USERPROFILE\models\speaker",
    # Silero voice-activity model -- one 1.3 MB file, and it is what lets the
    # audio socket take turns on its own. Take the openvino_16k build: the
    # stock silero_vad.onnx has a sample-rate branch OpenVINO cannot convert.
    [string] $Vad       = "$env:USERPROFILE\models\silero-vad",
    [int]    $Port      = 8000,
    # 'auto' is right for almost everyone: the server computes the smallest
    # ratio that makes the model fit, and stays off when it already does.
    [string] $OffloadRatio = 'auto',
    # The KV pool IS the context window, and agent clients are what make it
    # matter: Claude Code alone sends ~35k tokens before you type.
    [string] $ContextTokens = '40000',
    # 8081, not 8080. Odysseus publishes its own SearXNG on 127.0.0.1:8080, so
    # on a machine running both, whichever starts first takes the port and the
    # other silently uses its neighbour's search engine.
    [int]    $SearxPort = 8081,
    # int8 KV halves cache bytes per token. Measured on the B390: same prefill
    # rate, correct output. Pass "" to use f16.
    [string] $KvPrecision = 'u8',
    # ASR sits in the voice critical path and is ~3x faster on the GPU.
    [ValidateSet('GPU', 'CPU', 'NPU')]
    [string] $WhisperDevice = 'GPU',
    # Unload models after this many seconds idle (0 = never).
    [int]    $IdleTimeout = 900,
    # Regenerate locally.lnk and install it as a Windows app entry, then exit.
    # The shortcut points at locally-key.ps1, which opens a chat terminal.
    [switch] $CreateShortcut,
    [switch] $Desktop,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = if ($env:OS -eq 'Windows_NT') { Join-Path $root 'venv\Scripts\python.exe' } else { Join-Path $root 'venv/bin/python' }

if ($CreateShortcut) {
    if ($env:OS -ne 'Windows_NT') { throw '-CreateShortcut writes a Windows shortcut; there is nothing to write here.' }
    $lnk = Join-Path $root 'locally.lnk'
    # System32's Windows PowerShell, deliberately, and NOT pwsh: it is
    # Microsoft-signed and in System32, so Smart App Control never looks twice
    # (SAC enforcing is what killed locallyKey.exe -- TODONT.md). It points at
    # locally-key.ps1, which attaches a chat terminal to a running server and
    # only starts one when nothing answers.
    $target = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
    $key = Join-Path $PSScriptRoot 'locally-key.ps1'
    $ws = New-Object -ComObject WScript.Shell
    $s = $ws.CreateShortcut($lnk)
    $s.TargetPath = $target
    $s.Arguments = '-NoProfile -WindowStyle Hidden -File "{0}" -Port {1}' -f $key, $Port
    $s.WorkingDirectory = $root
    $s.WindowStyle = 7          # minimised: no console flash on key press
    $s.Description = 'Open a locally chat terminal'
    $icon = Join-Path $root 'static\icons\locally.ico'
    $s.IconLocation = if (Test-Path $icon) { "$icon,0" } else { "$env:SystemRoot\System32\SHELL32.dll,13" }
    $s.Save()
    Write-Host "Created $lnk"

    # A .lnk in Start Menu\Programs is what Windows counts as an installed
    # app: searchable from Start, pinnable to the taskbar, and it is what a
    # key remapper points at. Creation and lookup used to disagree about where
    # the app lives, which was the whole "why isn't locally an app" problem.
    $startMenu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
    $installed = Join-Path $startMenu 'locally.lnk'
    try {
        if (-not (Test-Path $startMenu)) { New-Item -ItemType Directory -Force $startMenu | Out-Null }
        Copy-Item -LiteralPath $lnk -Destination $installed -Force
        Write-Host "Installed to Start menu: $installed"
    } catch {
        Write-Host "Could not write the Start menu entry: $($_.Exception.Message)"
    }
    if ($Desktop) {
        try {
            $desktopLnk = Join-Path ([Environment]::GetFolderPath('Desktop')) 'locally.lnk'
            Copy-Item -LiteralPath $lnk -Destination $desktopLnk -Force
            Write-Host "Installed to Desktop: $desktopLnk"
        } catch {
            Write-Host "Could not write the Desktop shortcut: $($_.Exception.Message)"
        }
    }
    # Windows' own Settings -> Personalization -> Text input -> "Customize
    # Copilot key" lists only signed MSIX apps, so it cannot target a .lnk. On
    # this machine NewPilot (MSIX) forwards the press to the shortcut.
    Write-Host ""
    Write-Host "Hardware key: point NewPilot (or your vendor's key utility) at the"
    Write-Host "shortcut above. Windows' own 'Customize Copilot key' setting only"
    Write-Host "accepts signed MSIX apps, so it cannot target this."
    return
}

if (-not (Test-Path -LiteralPath $python)) { throw "Run install.ps1 first: no venv Python at $python" }

# A TCP connect, not an HTTP request: /health is served by the same process
# that may be busy compiling a 14 GB model, so it can take seconds to answer,
# and an HTTP timeout here once reported "not running" for a live server and
# started a SECOND one on the same port.
$client = [System.Net.Sockets.TcpClient]::new()
try { $serving = $client.ConnectAsync('127.0.0.1', $Port).Wait(400) -and $client.Connected }
catch { $serving = $false }
finally { $client.Dispose() }
if ($serving) {
    Write-Host "Something is already listening on port $Port."
    Write-Host "Attach a terminal to it:  python locally.py chat --port $Port"
    Write-Host "Or start another server with -Port <n>."
    exit 1
}

# Exactly one chat model, and no name is load-bearing. Preferences first,
# then whatever locally.py can actually see -- deleting a model is the user's
# business and must not change the launcher into something that starts a
# server for a directory that is not there.
#
# The device is left to the server in every case. `--device` is a preference,
# and _choose_device already rules the NPU out for vision and for
# group-quantized int4 by reading the IR, which a shell script cannot do.
$chatModel = $null
foreach ($name in $ChatModels) {
    if (-not $name) { continue }
    $candidate = if ([IO.Path]::IsPathRooted($name)) { $name } else { Join-Path $ModelsDir $name }
    if (Test-Path -LiteralPath $candidate) { $chatModel = $candidate; break }
}
if (-not $chatModel) {
    $searchIn = if (Test-Path -LiteralPath $ModelsDir) { $ModelsDir } else { $root }
    $found = & $python (Join-Path $root 'locally.py') '--list-models' $searchIn 2>$null
    # path<TAB>name<TAB>llm|vlm. Prefer a text model: it is the one the NPU can
    # take, and a VLM gets no prefix cache.
    $rows = @($found | Where-Object { $_ -match "`t" } | ForEach-Object { , ($_ -split "`t") })
    $pick = @($rows | Where-Object { $_[2] -eq 'llm' })[0]
    if (-not $pick) { $pick = @($rows)[0] }
    if ($pick) {
        $chatModel = $pick[0]
        Write-Host "Using the only chat model found: $($pick[1])"
    }
}
if (-not $chatModel) {
    # A clean clone has no model yet. The API still starts, and a model can be
    # loaded into it later (POST /v1/models/load, or /load in terminal chat).
    $chatModel = Join-Path $root 'model'
    Write-Host "No chat model found; starting the API without one."
}
$chatDevice = 'auto'
$modelsRoot = if (Test-Path -LiteralPath $ModelsDir) { $ModelsDir } else { Split-Path $chatModel -Parent }

$serverArgs = @(
    (Join-Path $root 'locally.py')
    '--device'; $chatDevice
    '--model-dir'; $chatModel
    '--models-dir'; $modelsRoot
    '--port'; "$Port"
    '--idle-timeout'; "$IdleTimeout"
    '--vscode-compat'
)
# Passed unconditionally: the slot itself skips KV_CACHE_PRECISION on the NPU
# (device.py), so the launcher does not need to know where the model landed.
if ($KvPrecision) { $serverArgs += @('--kv-precision', $KvPrecision) }
# Web search is opt-in and degrades to off: with nothing installed and nothing
# listening, the flag is simply not passed and the server makes no outbound
# connections at all. Passed whenever SearXNG is INSTALLED, not only when it
# is up -- the server starts it on demand and stops it when idle.
$searxRoot = Join-Path (Split-Path $root -Parent) 'searxng-src'
$searxUp = $false
try {
    $probe = [System.Net.Sockets.TcpClient]::new()
    $searxUp = $probe.ConnectAsync('127.0.0.1', $SearxPort).Wait(250) -and $probe.Connected
    $probe.Dispose()
} catch { }
if ($searxUp -or (Test-Path $searxRoot)) {
    $serverArgs += @('--search-url', "http://localhost:$SearxPort")
    if (Test-Path $searxRoot) { $serverArgs += @('--searxng-root', $searxRoot) }
}
# No --no-prompt-cache here any more, and that is not a loss: PROMPT_CACHE is
# tri-state and its default IS per-device (on for CPU, off for GPU -- prefix
# caching is a CPU win and a GPU trap on this box). The launcher used to force
# it from a device it had guessed; the server knows which device the model
# actually landed on. Same for --offload-ratio, whose default is already auto.
# --context-tokens is passed everywhere: on a GPU with no prefix cache no pool
# is built and the number is simply unused.
if ($ContextTokens) { $serverArgs += @('--context-tokens', $ContextTokens) }
if ($OffloadRatio -and $OffloadRatio -ne 'auto') {
    $serverArgs += @('--offload-ratio', "$OffloadRatio")
}
# ASR on the GPU: measured 0.3 s against ~1.05 s on CPU for the same clip, and
# it sits directly in the voice turn's critical path. 245 MB next to a
# multi-GB chat model is noise.
if (Test-Path $Whisper) { $serverArgs += @('--whisper-dir', $Whisper, '--whisper-device', $WhisperDevice) }
if (Test-Path $Tts)     { $serverArgs += @('--tts-dir', $Tts, '--tts-device', 'CPU') }
# VAD on the CPU deliberately: ~1M parameters, running on every 32 ms frame
# for as long as the mic is open. That belongs nowhere near the NPU, which is
# answering, or the GPU, which is doing ASR.
if (Test-Path $Vad)     { $serverArgs += @('--vad-dir', $Vad, '--vad-device', 'CPU') }
if (Test-Path $Speaker) { $serverArgs += @('--speaker-dir', $Speaker, '--speaker-device', 'CPU') }
$serverArgs += $ExtraArgs

& $python @serverArgs
exit $LASTEXITCODE
