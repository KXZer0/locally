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
# PositionalBinding=$false is load-bearing, not tidiness. `api.ps1` forwards
# `--port 8000` GNU-style, because the installer-generated start.ps1 hands its
# arguments straight to locally.py. PowerShell does not recognise `--port` as a
# parameter name, so with default binding it fills the POSITIONAL parameters
# instead: `$ChatModels = '--port'` and `$ModelsDir = '8000'`. Measured -- that
# is every model on this machine except the two symlinks in the repo root
# vanishing from `/v1/models`, which is what "not all models are showing" was.
# `--port` never reached python either, so `api.ps1 -Port 9000` served 8000.
# Both symptoms are silent: the launcher starts, and the wrong answer looks
# like a correct one. With positional binding off, an unrecognised flag lands
# in $ExtraArgs and is forwarded, which is where it was always meant to go.
[CmdletBinding(PositionalBinding = $false)]
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
    [string[]] $ChatModels = @('gemma-4-E4B-it-int4-ov',
                               'Qwen3-8B-abliterated-int4-cw-ov',
                               'Qwen3-8B-int4-cw-ov'),
    [string] $ModelsDir = "$env:USERPROFILE\models",
    # Optional models stay off unless named. API startup should not compile
    # audio engines merely because conventional folders happen to exist.
    [string] $Whisper   = '',
    # Speech-to-text by folder: a Parakeet TDT export (FluidInference/
    # parakeet-tdt-0.6b-v3-ov) or a Whisper one; the server reads which. Takes
    # precedence over -Whisper, since one ASR model is served at a time.
    [string] $Asr       = '',
    [string] $Tts       = '',
    [string] $Speaker   = '',
    # Silero voice-activity model -- one 1.3 MB file, and it is what lets the
    # audio socket take turns on its own. Take the openvino_16k build: the
    # stock silero_vad.onnx has a sample-rate branch OpenVINO cannot convert.
    [string] $Vad       = '',
    # Utility models are NOT like the audio ones and must not be treated like
    # them. Audio engines compile eagerly and sit in the voice critical path;
    # utility slots wait on the chat model's load event and then compile on
    # first use, so the only startup cost is OCR's ~160 MB. Measured
    # 2026-09-10 on this box: chat-ready at 9.7 s with utilities on, against
    # 9.6 s with them off.
    #
    # So the default is the conventional directory, gated on the OCR pair
    # actually being there. Requiring a flag is what left `/health.util` null
    # on an ordinary start, with document reading, OCR, local search and
    # upscaling silently dead while the models sat on disk -- reported as
    # "does it still support images and documents". Pass '' to turn them off.
    [string] $UtilModels = "$env:USERPROFILE\models\util",
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
    # Parakeet device. CPU by default: the chat model usually holds the GPU,
    # and the NPU path is static-shape and not yet verified here -- run
    # scripts/parakeet-check.py on this machine before choosing NPU.
    [ValidateSet('GPU', 'CPU', 'NPU')]
    [string] $AsrDevice = 'CPU',
    # Kokoro text-to-speech device. CPU keeps the GPU free; GPU is faster for voice apps (Pumpkin Trivia).
    [ValidateSet('GPU', 'CPU')]
    [string] $TtsDevice = 'CPU',
    # Unload models after this many seconds idle (0 = never).
    [int]    $IdleTimeout = 0,
    # Regenerate locally.lnk and install it as a Windows app entry, then exit.
    # The shortcut points at locally-key.ps1; API is the default action and
    # -ShortcutMode chat makes it open the optional terminal client instead.
    [switch] $CreateShortcut,
    [switch] $Desktop,
    [ValidateSet('chat', 'api')]
    [Alias('Mode')]
    [string] $ShortcutMode = 'api',
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
    $s.Arguments = '-NoProfile -WindowStyle Hidden -File "{0}" -Mode {1} -Port {2}' -f $key, $ShortcutMode, $Port
    $s.WorkingDirectory = $root
    $s.WindowStyle = 7          # minimised: no console flash on key press
    $s.Description = if ($ShortcutMode -eq 'chat') { 'Open a Locally chat terminal' } else { 'Start or inspect the Locally API' }
    $icon = Join-Path $root 'static\icons\locally.ico'
    $s.IconLocation = if (Test-Path $icon) { "$icon,0" } else { "$env:SystemRoot\System32\SHELL32.dll,13" }
    $s.Save()
    Write-Host "Created $lnk"
    Write-Host "Shortcut mode: $ShortcutMode"

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

# A forwarded `--port N` reaches argparse and wins there, so the SERVER lands on
# the right port either way. What it does not do on its own is tell this script,
# which then pre-checks the wrong port and can refuse to start over a listener
# that has nothing to do with the port being asked for.
for ($i = 0; $i -lt $ExtraArgs.Count - 1; $i++) {
    if ($ExtraArgs[$i] -eq '--port') {
        $forwarded = 0
        if ([int]::TryParse($ExtraArgs[$i + 1], [ref]$forwarded)) { $Port = $forwarded }
    }
}

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
if ($Asr -and (Test-Path $Asr)) {
    $serverArgs += @('--asr-dir', $Asr, '--asr-device', $AsrDevice)
} elseif ($Whisper -and (Test-Path $Whisper)) {
    $serverArgs += @('--whisper-dir', $Whisper, '--whisper-device', $WhisperDevice)
}
if ($Tts -and (Test-Path $Tts))         { $serverArgs += @('--tts-dir', $Tts, '--tts-device', $TtsDevice) }
# VAD on the CPU deliberately: ~1M parameters, running on every 32 ms frame
# for as long as the mic is open. That belongs nowhere near the NPU, which is
# answering, or the GPU, which is doing ASR.
if ($Vad -and (Test-Path $Vad))         { $serverArgs += @('--vad-dir', $Vad, '--vad-device', 'CPU') }
if ($Speaker -and (Test-Path $Speaker)) { $serverArgs += @('--speaker-dir', $Speaker, '--speaker-device', 'CPU') }
# OCR is the minimum the utilities need, so the detector/recognizer pair is
# the sentinel: present means the models were downloaded on purpose, absent
# means there is nothing to turn on. Only a directory the user NAMED earns a
# warning -- warning about the default on a machine that simply has no utility
# models would fire on every start of a clean install.
if ($UtilModels) {
    $ocrDet = Join-Path $UtilModels 'ocr-det\inference.onnx'
    $ocrRec = Join-Path $UtilModels 'ocr-rec\inference.onnx'
    if ((Test-Path -LiteralPath $ocrDet) -and (Test-Path -LiteralPath $ocrRec)) {
        $serverArgs += @('--util-models-dir', $UtilModels)
    } elseif ($PSBoundParameters.ContainsKey('UtilModels')) {
        Write-Warning "Utilities were requested, but $UtilModels does not contain the OCR detector and recognizer (ocr-det\inference.onnx, ocr-rec\inference.onnx). Utilities remain off."
    }
}
$serverArgs += $ExtraArgs

& $python @serverArgs
exit $LASTEXITCODE
