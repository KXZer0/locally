#requires -Version 7.0
<#
.SYNOPSIS
    One-key launcher for locally — start it if needed, then open the UI.

.DESCRIPTION
    Written for a hardware key (the Copilot key remapped via PowerToys
    Keyboard Manager) or a desktop shortcut. Idempotent by design: pressing
    the key twice must not start a second server, so it checks /health first
    and only launches when nothing is listening.

    If the server is already up it just focuses the browser, which makes the
    key behave like "show me locally" rather than "restart locally".

.EXAMPLE
    pwsh -NoProfile -WindowStyle Hidden -File scripts\locally-launch.ps1
#>
param(
    # ONE chat model at a time, Ollama-style. The NPU has no memory of its own
    # — it allocates from the same 31.5 GB as the GPU — so a second resident
    # model costs real RAM and heat while you can only ever talk to one of
    # them. Swap models from the UI (or POST /v1/models/load) instead; that
    # unloads the old one first, so peak memory stays at one model.
    #
    # The NPU wins by default, because the key is pressed for a quick question
    # far more often than for a coding session, and that is the case the NPU is
    # built for: lowest power, no fans, and it leaves the GPU free. Qwen3-8B is
    # channel-wise int4 — the only quantization the vpux compiler accepts
    # (group-quantized int4 crashes it), and verified on this NPU.
    #
    # What the NPU cannot do, and why the swap exists: no vision (text only),
    # no tool calling, and a hard MAX_PROMPT_LEN of 4096 tokens. So Claude Code
    # and any agent client need the GPU coder — load it from the settings panel
    # (or POST /v1/models/load), which unloads this one first.
    #
    # -NpuModel '' falls back to the GPU model below.
    [string] $NpuModel  = "$env:USERPROFILE\models\Qwen3-8B-int4-cw-ov",
    [string] $GpuModel  = "$env:USERPROFILE\models\Qwen3.6-27B-int4-ov",
    [string] $Whisper   = "$env:USERPROFILE\models\whisper-small-int8-ov",
    [string] $Tts       = "$env:USERPROFILE\models\Kokoro-82M-int8-ov",
    [string] $Speaker   = "$env:USERPROFILE\models\speaker",
    # Silero voice-activity model — this is what makes the Voice tab take turns
    # on its own instead of needing the mic button held. Without it the tab
    # still works, but push-to-talk is the only mode. One 1.3 MB file:
    #   curl -L --create-dirs -o "$env:USERPROFILE\models\silero-vad\silero_vad_openvino_16k.onnx" `
    #     https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad_openvino_16k.onnx
    # Take the openvino_16k build: the stock silero_vad.onnx has a sample-rate
    # branch OpenVINO's ONNX frontend cannot convert.
    [string] $Vad       = "$env:USERPROFILE\models\silero-vad",
    [int]    $Port      = 8000,
    # 'auto' is the right answer for almost everyone: the server computes the
    # smallest ratio that makes the model fit (weights + KV pool vs the device
    # budget) and stays off entirely when it already fits. A pinned number is
    # either wasted speed or a load that OOMs. Pass a number to override, 0
    # to disable.
    [string] $OffloadRatio = 'auto',
    # The KV pool IS the context window, and agent clients are what make it
    # matter: Claude Code alone sends ~35k tokens before you type. The old
    # 2 GB default (~21k tokens on a 30B) fails such a turn mid-request.
    # 40k costs ~4 GB and still leaves the 30B needing no offload at all.
    [string] $ContextTokens = '40000',
    [int]    $SearxPort = 8080,
    # int8 KV halves cache bytes/token (96 KB -> ~48 KB on the 30B). At f16 a
    # 100k-token coding session needs 9.6 GB of KV on top of 15.2 GB of
    # weights and simply will not fit; at u8 it needs 4.8. Measured on the
    # B390: same prefill rate (2,088 tok/s), correct output. Quality cost is
    # far below the int4 weights already in use. Pass '' to use f16.
    [string] $KvPrecision = 'u8',
    # ASR sits in the voice critical path and is 3x faster on the GPU; set to
    # CPU if the GPU model is so large that the extra 245 MB won't fit.
    [ValidateSet('GPU', 'CPU', 'NPU')]
    [string] $WhisperDevice = 'GPU',
    # Unload models after this many seconds idle (0 = never). Keeps RAM free
    # when the key is used for a quick question and then forgotten.
    [int]    $IdleTimeout = 900,
    [switch] $NoBrowser,
    # Regenerate locally.lnk next to the project, then exit. The shortcut is
    # what a hardware key / PowerToys "Open app" points at, since those only
    # take a path and no arguments.
    [switch] $CreateShortcut
)

$ErrorActionPreference = 'Stop'
$root   = Split-Path $PSScriptRoot -Parent
$python = Join-Path $root 'venv\Scripts\python.exe'
$url    = "http://127.0.0.1:$Port"

# A hardware key gives no console to read, so leave a trail. When the key
# "does nothing", this log is the only way to tell whether the script ran at
# all, failed early, or ran fine and the browser step was the problem.
$logDir = Join-Path $env:LOCALAPPDATA 'locally'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$launchLog = Join-Path $logDir 'launch.log'
function Write-Log($msg) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" |
        Add-Content -Path $launchLog -Encoding utf8
}
Write-Log "--- invoked (user=$env:USERNAME, cwd=$PWD) ---"

# Win32 window handling: pressing the key a second time should raise the
# window that's already showing locally, not pile up another tab.
#
# Sourced, not redeclared. This block used to call Add-Type, which shells out
# to the C# compiler at runtime (measured 336 ms here) -- the exact cost the
# Reflection.Emit rewrite removed from locally-key.ps1 and never removed from
# this file. Sharing one file is what stops that drift recurring.
$win32 = Join-Path $PSScriptRoot 'locally-win32.ps1'
if (Test-Path $win32) { . $win32 }

function Show-ExistingWindow {
    # The app-mode window titles itself "locally" (from the page <title>), so
    # it's findable without talking to the browser. Plain tabs inside a big
    # browser window can't be raised individually from outside -- that's why
    # the launcher prefers app mode below.
    if (-not (Get-Command Show-locallyWindow -ErrorAction SilentlyContinue)) {
        Write-Log "locally-win32.ps1 missing; cannot raise an existing window"
        return $false
    }
    $p = Show-locallyWindow
    if ($p) {
        Write-Log "raised existing window (pid $($p.Id), '$($p.MainWindowTitle)')"
        return $true
    }
    return $false
}

# Open-locallyApp now lives in locally-win32.ps1, shared with the key script.

function Open-Url($u) {
    if (Show-ExistingWindow) { return }

    $opened = Open-locallyApp $u
    if ($opened) {
        Write-Log "opened ${opened}: $u"
        return
    }
    # Fallbacks: explorer resolves the default browser in any context;
    # Start-Process needs a usable association, which a key-remapper's
    # environment doesn't always have.
    try {
        Start-Process -FilePath 'explorer.exe' -ArgumentList $u -ErrorAction Stop
        Write-Log "opened via explorer.exe: $u"
        return
    } catch { Write-Log "explorer.exe failed: $($_.Exception.Message)" }
    try {
        Start-Process $u -ErrorAction Stop
        Write-Log "opened via shell association: $u"
    } catch { Write-Log "FAILED to open $u : $($_.Exception.Message)" }
}

if ($CreateShortcut) {
    $lnk = Join-Path $root 'locally.lnk'
    # System32's Windows PowerShell, deliberately, and NOT pwsh:
    #   - it is Microsoft-signed and in System32, so Smart App Control never
    #     looks twice (SAC enforcing is what killed locallyKey.exe — TODONT.md);
    #   - it is already present and starts faster here. The shared Win32 helper
    #     also works on pwsh 7 via AssemblyBuilder.DefineDynamicAssembly, but
    #     there is no benefit in waking the larger host for this hot path.
    # It points at locally-key.ps1, not at this script: that is the hot path
    # (~720 ms to raise an existing window) and it hands off to this one only
    # when there is real work to do.
    $target = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
    $key    = Join-Path $PSScriptRoot 'locally-key.ps1'
    $ws = New-Object -ComObject WScript.Shell
    $s  = $ws.CreateShortcut($lnk)
    $s.TargetPath       = $target
    $s.Arguments        = '-NoProfile -WindowStyle Hidden -File "{0}" -Port {1}' -f $key, $Port
    $s.WorkingDirectory = $root
    $s.WindowStyle      = 7          # minimised: no console flash on key press
    $s.Description      = 'Launch locally and open the web UI'
    $icon = Join-Path $root 'static\icons\locally.ico'
    $s.IconLocation     = if (Test-Path $icon) { "$icon,0" } else { "$env:SystemRoot\System32\SHELL32.dll,13" }
    $s.Save()
    Write-Host "Created $lnk"
    Write-Host "Point PowerToys 'Open app' (or a desktop/taskbar shortcut) at that file."
    return
}

# Deliberately a TCP connect, not an HTTP request. /health is served by the
# same process that is busy compiling a 14 GB model, so it can take many
# seconds to answer — an HTTP timeout here reported "not running" for a server
# that was perfectly alive and launched a SECOND one on the same port. Whether
# something owns the port is the actual question, and TCP answers it in
# microseconds.
function Test-locallyUp {
    $c = [System.Net.Sockets.TcpClient]::new()
    try {
        $ok = $c.ConnectAsync('127.0.0.1', $Port).Wait(400)
        return $ok -and $c.Connected
    } catch { return $false } finally { $c.Dispose() }
}

$alreadyUp = Test-locallyUp
Write-Log "server already running: $alreadyUp"

if (-not $alreadyUp) {
    if (-not (Test-Path $python)) {
        Write-Log "FATAL: venv python not found at $python"
        throw "venv python not found at $python"
    }

    # Exactly one chat model. -NpuModel wins if given (low-power text only),
    # otherwise the GPU model (vision + tools).
    if ($NpuModel -and (Test-Path $NpuModel)) {
        $chatModel = $NpuModel; $chatDevice = 'NPU'
    } elseif (Test-Path $GpuModel) {
        $chatModel = $GpuModel; $chatDevice = 'GPU'
    } else {
        # A clean clone has no model yet by definition. Passing the ordinary
        # default path (even though it does not exist) lets locally start in
        # setup-only mode and open the guided onboarding instead of making the
        # launcher itself the first-run installer. main() also recognises this
        # exact default on later launches and applies the saved setup choice —
        # including a non-default OpenVINO model or an Ollama proxy on NVIDIA.
        $chatModel = Join-Path $root 'model'; $chatDevice = 'auto'
        Write-Log "no chat model found; starting guided setup"
    }

    $serverArgs = @(
        (Join-Path $root 'locally.py')
        '--device'; $chatDevice
        '--model-dir'; $chatModel
        '--models-dir'; (Split-Path $chatModel -Parent)
        '--port'; "$Port"
        '--idle-timeout'; "$IdleTimeout"
        '--vscode-compat'
    )
    # Prefix caching is a CPU win and a GPU trap on this box: on the B390 the
    # continuous-batching backend prefills ~10x slower than the plain pipeline
    # and hangs outright when consecutive prompts share a near-total prefix
    # (which is exactly what an agent client sends). Measured: Claude Code's
    # first turn 75s -> 8.7s with it off. See TODONT.md.
    if ($KvPrecision -and $chatDevice -ne 'NPU') {
        $serverArgs += '--kv-precision'; $serverArgs += $KvPrecision
    }
    # Web search is opt-in and degrades to off: if nothing is listening on the
    # SearXNG port we simply do not pass the flag, and the server makes no
    # outbound connections at all. Start it with scripts\searxng.ps1.
    # Pass --search-url whenever SearXNG is INSTALLED, not only when it happens
    # to be listening: the server starts it on demand and stops it when idle,
    # so requiring it to be up at launch would defeat the whole point.
    $searxRoot = Join-Path (Split-Path $root -Parent) 'searxng-src'
    $searxUp = $false
    try {
        $c = [System.Net.Sockets.TcpClient]::new()
        $searxUp = $c.ConnectAsync('127.0.0.1', $SearxPort).Wait(250) -and $c.Connected
        $c.Dispose()
    } catch { }
    if ($searxUp -or (Test-Path $searxRoot)) {
        $serverArgs += '--search-url'; $serverArgs += "http://localhost:$SearxPort"
        if (Test-Path $searxRoot) { $serverArgs += '--searxng-root'; $serverArgs += $searxRoot }
        Write-Log ("web search enabled (running={0}, on-demand={1})" -f $searxUp, (Test-Path $searxRoot))
    } else {
        Write-Log "no SearXNG installed or running - web search stays off"
    }
    if ($chatDevice -eq 'GPU') { $serverArgs += '--no-prompt-cache' }
    else { if ($ContextTokens) { $serverArgs += '--context-tokens'; $serverArgs += $ContextTokens } }
    if ($chatDevice -eq 'GPU' -and $OffloadRatio) {
        $serverArgs += '--offload-ratio'; $serverArgs += "$OffloadRatio"
    }
    # ASR on the GPU: measured 0.3 s vs ~1.05 s on CPU for the same clip, and
    # transcription sits directly in the voice turn's critical path. It's a
    # 245 MB model next to a multi-GB chat model, so the memory cost is noise.
    if (Test-Path $Whisper) { $serverArgs += '--whisper-dir'; $serverArgs += $Whisper
                              $serverArgs += '--whisper-device'; $serverArgs += $WhisperDevice }
    if (Test-Path $Tts)     { $serverArgs += '--tts-dir'; $serverArgs += $Tts
                              $serverArgs += '--tts-device'; $serverArgs += 'CPU' }
    # CPU deliberately: ~1M parameters, and it runs on every 32 ms frame for as
    # long as the mic is open. That belongs nowhere near the NPU, which is
    # busy answering, or the GPU, which is doing ASR.
    if (Test-Path $Vad)     { $serverArgs += '--vad-dir'; $serverArgs += $Vad
                              $serverArgs += '--vad-device'; $serverArgs += 'CPU' }
    if (Test-Path $Speaker) { $serverArgs += '--speaker-dir'; $serverArgs += $Speaker
                              $serverArgs += '--speaker-device'; $serverArgs += 'CPU' }

    Write-Log "starting server: $python $($serverArgs -join ' ')"
    Start-Process -FilePath $python -ArgumentList $serverArgs `
        -WorkingDirectory $root -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logDir 'server.log') `
        -RedirectStandardError  (Join-Path $logDir 'server.err') | Out-Null

    # Wait only for the socket to bind, not for models to load. Flask listens
    # within a couple of seconds while the models compile on background
    # threads, and the UI polls /health and shows per-device loading state —
    # so opening early gives instant feedback instead of a dead key press.
    $sw = [Diagnostics.Stopwatch]::StartNew()
    while ($sw.ElapsedMilliseconds -lt 30000 -and -not (Test-locallyUp)) {
        Start-Sleep -Milliseconds 100
    }
    Write-Log "port bound after $($sw.ElapsedMilliseconds)ms"
}

if (-not $NoBrowser) { Open-Url $url } else { Write-Log 'browser suppressed (-NoBrowser)' }
Write-Log "done"
