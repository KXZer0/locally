#requires -Version 7.0
<#
.SYNOPSIS
    Start Claude Code against the local locally server instead of the cloud.

.DESCRIPTION
    Claude Code talks to whatever speaks the Anthropic Messages API, and
    locally does (POST /v1/messages). So this sets the three environment
    variables the CLI reads and launches it — no proxy, no router, no account.

    The CLI itself ships inside the Claude desktop app, which is why it is
    usually not on PATH: it lives in
    %APPDATA%\Claude\claude-code\<version>\claude.exe. This finds it there
    when `claude` isn't on PATH, newest version first.

    Run it directly, or press the button in locally's settings panel — that
    button runs this same script, so there is one path to debug, not two.

.EXAMPLE
    pwsh -File scripts\claude-local.ps1 -ProjectDir C:\Projects\MyApp
#>
param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [string]$Model = "",
    [string]$ProjectDir = ".",
    # Launch VS Code instead of the terminal CLI. The Claude Code extension
    # shells out to the same binary, so it obeys the same three variables —
    # but only if it inherits them, which means VS Code must be started from
    # a process that has them set. Setting them user-wide would silently
    # redirect every Claude Code session on the machine, including ones meant
    # for the cloud, so this scopes them to the window it opens.
    [switch]$VSCode,
    # Claude Code ships 30 tool schemas: 99,044 of a 141,959-char request
    # (measured). On an iGPU that prefill is most of your wait, so the
    # launcher defaults to the six a coding agent actually needs. Pass ""
    # for the full set.
    [string]$Tools = "Bash,Edit,Read,Write,Glob,Grep"
)

$ErrorActionPreference = 'Stop'

function Find-ClaudeCli {
    $onPath = Get-Command claude -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }

    $roots = @()
    if ($IsWindows) {
        $roots += Join-Path $env:APPDATA 'Claude\claude-code'
    }
    $roots += Join-Path $HOME '.claude\local'
    $roots += Join-Path $HOME '.local\bin'

    foreach ($root in $roots) {
        if (-not (Test-Path $root)) { continue }
        $exe = if ($IsWindows) { 'claude.exe' } else { 'claude' }
        # Version directories sort as text ("2.1.9" > "2.1.10"), so compare
        # them as numbers or a point release will silently win over a newer one.
        $found = Get-ChildItem $root -Directory -ErrorAction SilentlyContinue |
            Sort-Object { try { [version]($_.Name -replace '[^0-9.]', '') }
                          catch { [version]'0.0.0' } } -Descending |
            ForEach-Object { Join-Path $_.FullName $exe } |
            Where-Object { Test-Path $_ } |
            Select-Object -First 1
        if ($found) { return $found }
        $direct = Join-Path $root $exe
        if (Test-Path $direct) { return $direct }
    }
    return $null
}

$cli = Find-ClaudeCli
if (-not $cli -and -not $VSCode) {
    Write-Host "Claude Code CLI not found." -ForegroundColor Red
    Write-Host "  Looked on PATH and in the Claude desktop app's bundle."
    Write-Host "  Install it with:  npm install -g @anthropic-ai/claude-code"
    exit 1
}

if (-not (Test-Path $ProjectDir)) {
    Write-Host "No such directory: $ProjectDir" -ForegroundColor Red
    exit 1
}

# Ask the server what it is serving, so the model id is right without the
# caller having to know it. Not fatal if the server is down — Claude Code
# will say so more clearly than we can.
if (-not $Model) {
    try {
        $health = Invoke-RestMethod -Uri "$BaseUrl/health" -TimeoutSec 5
        $slot = $health.devices.PSObject.Properties.Value |
            Where-Object { $_.type -in @('llm', 'vlm') } | Select-Object -First 1
        if ($slot) { $Model = $slot.model }
    } catch {
        Write-Host "Warning: $BaseUrl is not answering — is locally running?" `
            -ForegroundColor Yellow
    }
}
if (-not $Model) { $Model = 'local' }

$env:ANTHROPIC_BASE_URL = $BaseUrl
$env:ANTHROPIC_AUTH_TOKEN = 'local'   # required by the CLI, ignored by locally
$env:ANTHROPIC_MODEL = $Model

Write-Host ""
Write-Host "  Claude Code -> locally" -ForegroundColor Cyan
Write-Host "    endpoint : $BaseUrl/v1/messages"
Write-Host "    model    : $Model"
Write-Host "    project  : $((Resolve-Path $ProjectDir).Path)"
if ($Tools) { Write-Host "    tools    : $Tools" }
Write-Host ""

Set-Location $ProjectDir

if ($VSCode) {
    $code = Get-Command code -ErrorAction SilentlyContinue
    if (-not $code) {
        Write-Host "VS Code's 'code' command is not on PATH." -ForegroundColor Red
        Write-Host "  In VS Code: Ctrl+Shift+P -> 'Shell Command: Install code in PATH'"
        exit 1
    }
    Write-Host "  Opening VS Code — its Claude Code panel will use the local model."
    Write-Host "  Only this window: other VS Code windows still use the cloud."
    & $code.Source $ProjectDir
    exit 0
}

$ccArgs = @()
if ($Tools) { $ccArgs += @('--tools', $Tools) }
& $cli @ccArgs
