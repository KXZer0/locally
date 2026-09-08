<#
.SYNOPSIS
    Why Odysseus cannot see locally: runs the reachability ladder and names the
    layer that is broken, instead of leaving you to guess between five of them.

.DESCRIPTION
    Odysseus reports "offline" and lists no models for a handful of unrelated
    reasons -- the server is not running, it bound loopback only, the WSL
    gateway address moved, a firewall drops the port, or .env points somewhere
    else -- and every one of them looks identical from inside Odysseus. This
    walks them in order and prints one verdict.

    The trick the whole script rests on: a failed connection is either DROPPED
    (silence, so curl burns the full timeout) or RESET (an answer, in
    milliseconds). Only a firewall drops. So the elapsed time, not the status
    code, is what separates "a firewall is eating this" from "nothing is
    listening". Documented in docs/ODYSSEUS.md section 7.

    Read-only: it changes no firewall rules and starts no servers.

.EXAMPLE
    pwsh -File scripts/diagnose-odysseus-link.ps1
.EXAMPLE
    pwsh -File scripts/diagnose-odysseus-link.ps1 -Port 8000 -Container odysseus_odysseus_1
#>
[CmdletBinding()]
param(
    [int]$Port = 8000,
    [string]$Container = 'odysseus_odysseus_1',
    [int]$TimeoutSec = 6
)

$ErrorActionPreference = 'Continue'
$problems = @()
$notes    = @()
$serving  = $null

function Say([string]$label, [string]$verdict, [string]$colour) {
    Write-Host ('  {0,-32} ' -f $label) -NoNewline
    Write-Host $verdict -ForegroundColor $colour
}

Write-Host ''
Write-Host 'locally -> Odysseus link check' -ForegroundColor Cyan
Write-Host ''

# --- 1. the gateway address, discovered rather than assumed -----------------
# 172.17.96.1 is stable across restarts but NOT across a WSL virtual-switch
# recreation, so hardcoding it is how this breaks silently months later.
$gw = (Get-NetIPAddress -AddressFamily IPv4 -EA SilentlyContinue |
        Where-Object { $_.InterfaceAlias -like '*WSL*' } | Select-Object -First 1).IPAddress
if (-not $gw) {
    Say 'WSL gateway' 'not found - is WSL running?' Red
    $problems += 'No vEthernet (WSL) adapter. Start the podman machine, then re-run.'
    $gw = '172.17.96.1'
} else {
    Say 'WSL gateway' $gw Green
}

# --- 2. is anything listening, and on what interface? ----------------------
$listen = Get-NetTCPConnection -State Listen -LocalPort $Port -EA SilentlyContinue
if (-not $listen) {
    Say ('listening on :' + $Port) 'NOTHING' Red
    $problems += "No process is listening on port $Port. Start the API with api.ps1 (not chat.ps1 --start)."
} else {
    $addrs = ($listen.LocalAddress | Sort-Object -Unique) -join ', '
    if ($listen.LocalAddress -contains '0.0.0.0') {
        Say ('listening on :' + $Port) '0.0.0.0 (all interfaces)' Green
    } else {
        Say ('listening on :' + $Port) ($addrs + ' - LOOPBACK ONLY') Red
        # core/terminal.py passes --host 127.0.0.1, so a chat-owned server can
        # never be reached from a container however the firewall is set.
        $problems += "Bound to $addrs only; a container can never reach that. chat.ps1 --start binds 127.0.0.1 - use api.ps1."
    }
    # .Path is the loaded IMAGE, which is what firewall rules match. It is not
    # the command line: venv\Scripts\python.exe redirects to the base
    # interpreter, so Win32_Process shows the venv and the firewall sees the
    # system Python. Reading the command line here hid a block rule for a whole
    # session -- see docs/ODYSSEUS.md section 7, trap 1.
    $owner = Get-Process -Id ($listen[0].OwningProcess) -EA SilentlyContinue
    if ($owner -and $owner.Path) {
        $serving = $owner.Path
        $notes += "serving process: $serving"
    }
}

# --- 3. does the host answer, on loopback and on the gateway? -------------
$targets = @(
    @{ name = 'host via 127.0.0.1';   addr = '127.0.0.1' },
    @{ name = 'host via WSL gateway'; addr = $gw }
)
foreach ($t in $targets) {
    try {
        $r = Invoke-WebRequest -Uri ('http://{0}:{1}/v1/models' -f $t.addr, $Port) -TimeoutSec $TimeoutSec -UseBasicParsing
        Say $t.name ('HTTP ' + $r.StatusCode) Green
    } catch {
        Say $t.name 'no answer' Red
        $problems += ('The host itself cannot reach http://{0}:{1}/v1/models.' -f $t.addr, $Port)
    }
}

# --- 4 and 5. the VM and the container -- timing is the diagnosis ---------
function Probe([string]$label, [string[]]$argv) {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $argv[0] @($argv[1..($argv.Count - 1)]) 2>&1 | Out-Null
    $code = $LASTEXITCODE
    $sw.Stop()
    $secs = $sw.Elapsed.TotalSeconds
    if ($code -eq 0) {
        Say $label ('HTTP 200 in {0:N2}s' -f $secs) Green
        return 'ok'
    }
    # curl exit 28 is its own timeout. Silence for the full window means the
    # packet was dropped, and only a firewall drops without answering.
    if ($code -eq 28 -or $secs -ge ($TimeoutSec - 0.5)) {
        Say $label ('DROPPED - timed out after {0:N1}s' -f $secs) Red
        return 'dropped'
    }
    Say $label ('reset in {0:N3}s (curl exit {1})' -f $secs, $code) Yellow
    return 'reset'
}

$url = 'http://{0}:{1}/v1/models' -f $gw, $Port
if (-not (Get-Command podman -EA SilentlyContinue)) {
    Say 'podman' 'not on PATH - VM/container rungs skipped' Yellow
} else {
    # Run the ssh rung from TEMP: podman's ssh writes a known_hosts fragment to
    # a file literally named NUL in the working directory, and a stray NUL in
    # the repo root shows up in git status forever after.
    Push-Location $env:TEMP
    try {
        $vm = Probe 'podman VM -> host' @('podman', 'machine', 'ssh', "curl -s -o /dev/null -m $TimeoutSec $url")
    } finally {
        Pop-Location
    }
    $ct = Probe 'Odysseus container -> host' @('podman', 'exec', $Container, 'curl', '-s', '-o', '/dev/null', '-m', "$TimeoutSec", $url)

    if ($vm -eq 'dropped') {
        $problems += "Port $Port is DROPPED between the WSL VM and the host. That is a firewall, not routing and not Odysseus. See docs/ODYSSEUS.md section 7."
    } elseif ($vm -eq 'reset' -and $ct -ne 'ok') {
        $problems += 'The packet reaches the host but is reset, so the path is open. Suspect the server or the port, not the firewall.'
    }
}

# --- 5b. reachable is not the same as advertising a model ----------------
# /v1/models lists only slots whose status is "ready". An idle-unloaded slot
# still SERVES -- the chat path reloads it on demand -- but it advertises
# nothing, so a client that discovers models by polling this endpoint sees an
# empty list and calls the backend offline. That is a second, independent cause
# of the same symptom, and it appears only after the idle timeout has elapsed.
try {
    $models = Invoke-RestMethod -Uri ('http://127.0.0.1:{0}/v1/models' -f $Port) -TimeoutSec $TimeoutSec
    $count = @($models.data).Count
    if ($count -gt 0) {
        Say 'models advertised' (($models.data.id | Select-Object -First 3) -join ', ') Green
    } else {
        Say 'models advertised' 'NONE - list is empty' Red
        $idle = $false
        try {
            $h = Invoke-RestMethod -Uri ('http://127.0.0.1:{0}/health' -f $Port) -TimeoutSec $TimeoutSec
            $idle = @($h.devices.PSObject.Properties.Value | Where-Object { $_.status -eq 'idle_unloaded' }).Count -gt 0
        } catch { }
        if ($idle) {
            $problems += 'A slot is idle_unloaded, so /v1/models is empty and Odysseus reads that as offline. It would still answer a chat request. Restart with -IdleTimeout 0 to keep the model resident and advertised.'
        } else {
            $problems += 'No model is loaded, so /v1/models is empty. Check the launcher actually loaded one.'
        }
    }
} catch {
    Say 'models advertised' 'could not read /v1/models' Yellow
}

# --- 6. does .env agree with the address we just proved? ------------------
$envFile = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'odysseus\.env'
if (Test-Path -LiteralPath $envFile) {
    $line = Select-String -Path $envFile -Pattern '^\s*OLLAMA_BASE_URL=(.+)$' | Select-Object -First 1
    if ($line) {
        $configured = $line.Matches[0].Groups[1].Value.Trim()
        if ($configured -match [regex]::Escape($gw) -and $configured -match (':' + $Port) -and $configured -match '/v1') {
            Say 'odysseus/.env base URL' 'matches' Green
        } else {
            Say 'odysseus/.env base URL' $configured Red
            $problems += ('odysseus/.env has OLLAMA_BASE_URL=' + $configured + ' but the proven address is http://{0}:{1}/v1 - the /v1 suffix is required.' -f $gw, $Port)
        }
    }
} else {
    Say 'odysseus/.env' 'not found' Yellow
}

# --- 7. the firewall rules most likely to be at fault --------------------
# Block beats Allow in Windows Firewall, and rules match the exact image path,
# so a block rule naming the wrong python still kills a correct allow rule.
if ($serving) {
    $blocking = @(Get-NetFirewallRule -Enabled True -Direction Inbound -Action Block -EA SilentlyContinue |
        Where-Object { ($_ | Get-NetFirewallApplicationFilter -EA SilentlyContinue).Program -eq $serving })
    if ($blocking.Count -gt 0) {
        Say 'block rule on serving exe' ([string]$blocking.Count + ' found') Red
        $problems += "A firewall Block rule names $serving. Block wins over Allow - remove it."
    } else {
        Say 'block rule on serving exe' 'none' Green
    }
}
$allow = @(Get-NetFirewallRule -PolicyStore ActiveStore -Enabled True -Direction Inbound -Action Allow -EA SilentlyContinue |
    Where-Object { ($_ | Get-NetFirewallPortFilter -EA SilentlyContinue).LocalPort -contains "$Port" })
if ($allow.Count -gt 0) {
    Say ('active allow rule for :' + $Port) (($allow.DisplayName | Select-Object -First 3) -join '; ') Green
} else {
    Say ('active allow rule for :' + $Port) 'NONE in ActiveStore' Red
    $problems += "No enforced allow rule for port $Port. A rule sitting in PersistentStore that never reached ActiveStore is not enforced."
}

# --- verdict --------------------------------------------------------------
Write-Host ''
foreach ($n in $notes) { Write-Host ('  note: ' + $n) -ForegroundColor DarkGray }
if ($problems.Count -eq 0) {
    Write-Host ''
    Write-Host 'Link looks healthy. If Odysseus still shows offline, the model id is the next suspect (docs/ODYSSEUS.md section 7).' -ForegroundColor Green
    Write-Host ''
    exit 0
}
Write-Host ''
Write-Host 'Problems, most likely first:' -ForegroundColor Yellow
$i = 1
foreach ($p in $problems) {
    Write-Host ('  ' + $i + '. ' + $p)
    $i++
}
Write-Host ''
exit 1
