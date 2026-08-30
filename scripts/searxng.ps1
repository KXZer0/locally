<#
.SYNOPSIS
    Start the local SearXNG that backs locally's web search.

.DESCRIPTION
    locally's --search-url talks to a SearXNG instance. This starts one on
    127.0.0.1:8080, loopback only, with JSON output enabled.

    Two things about running SearXNG on Windows, both discovered the hard way:

    * It has ONE Unix-only dependency: searx/valkeydb.py imports `pwd` at module
      scope. It is only used to build a valkey unix-socket path, and this
      instance runs without valkey, so a stub module in the venv satisfies the
      import and nothing calls into it. The stub lives in the venv's
      site-packages rather than in SearXNG's source tree, so `git pull` in
      searxng-src stays clean.

    * `git clone` cannot check out four files whose names contain a colon
      (nginx/apache/uwsgi deployment templates). They are irrelevant to running
      it; the clone is otherwise complete.

.EXAMPLE
    .\scripts\searxng.ps1
    .\scripts\searxng.ps1 -Install       # first-time setup
    .\scripts\searxng.ps1 -Update        # fast-forward + refresh dependencies
#>
#requires -Version 7.0
param(
    [string] $SearxRoot = (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'searxng-src'),
    [int]    $Port      = 8080,
    [switch] $Install,
    [switch] $Update
)

$ErrorActionPreference = 'Stop'
$settings = Join-Path $PSScriptRoot '..' 'searxng-settings.yml' | Resolve-Path -ErrorAction SilentlyContinue
# Accept the pre-rename filename too, so an existing SearXNG install
# keeps working without being touched.
if (-not $settings) {
    $settings = @('settings-locally.yml', 'settings-nollama.yml') |
        ForEach-Object { Join-Path $SearxRoot $_ } |
        Where-Object { Test-Path $_ } | Select-Object -First 1
}
$py = Join-Path $SearxRoot '.venv\Scripts\python.exe'

if ($Install -or $Update) {
    if (-not (Test-Path $SearxRoot)) {
        Write-Host "Cloning SearXNG to $SearxRoot ..."
        git clone --depth 1 https://github.com/searxng/searxng.git $SearxRoot
        # Four deployment templates have a colon in the filename and cannot
        # exist on NTFS. Restore everything else.
        Push-Location $SearxRoot; git restore --source=HEAD :/ 2>$null; Pop-Location
    } elseif ($Update) {
        Write-Host "Updating SearXNG in $SearxRoot ..."
        git -C $SearxRoot pull --ff-only
    }
    if (-not (Test-Path $py)) {
        Write-Host "Creating venv ..."
        python -m venv (Join-Path $SearxRoot '.venv')
    }
    Write-Host "Installing dependencies ..."
    & $py -m pip install --quiet --upgrade pip
    & $py -m pip install --quiet -r (Join-Path $SearxRoot 'requirements.txt')

    $site = & $py -c "import site;print(site.getsitepackages()[-1])"
    $shim = Join-Path $site 'pwd.py'
    if (-not (Test-Path $shim)) {
        Write-Host "Adding the Windows pwd stub ..."
        @'
"""Windows stub for the Unix-only `pwd` module. See scripts/searxng.ps1."""
def getpwuid(_uid):
    raise KeyError("pwd is unavailable on Windows; this instance runs without valkey")
def getpwnam(_name):
    return getpwuid(None)
def getpwall():
    return []
'@ | Set-Content -Path $shim -Encoding utf8
    }
    Write-Host "Done. Run this script again without -Install to start it."
    return
}

if (-not (Test-Path $py)) {
    throw "SearXNG is not installed at $SearxRoot. Run: .\scripts\searxng.ps1 -Install"
}
if (-not (Test-Path $settings)) {
    throw "Settings file not found: $settings"
}

$env:SEARXNG_SETTINGS_PATH = (Resolve-Path $settings).Path
Write-Host "SearXNG on http://127.0.0.1:$Port  (settings: $env:SEARXNG_SETTINGS_PATH)"
Write-Host "Point locally at it with:  --search-url http://localhost:$Port"
Push-Location $SearxRoot
try { & $py -m searx.webapp } finally { Pop-Location }
