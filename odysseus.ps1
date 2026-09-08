<# Start Locally and Odysseus when needed, then open Odysseus in the default browser. #>
param(
    [int]$Port = 7000,
    [int]$ApiPort = 8000,
    [switch]$CreateShortcut,
    [switch]$Desktop,
    [switch]$FromShortcut,
    [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'
$odysseusUrl = "http://127.0.0.1:$Port"

function Test-LocalPort([int]$Number) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try { return $client.ConnectAsync('127.0.0.1', $Number).Wait(350) -and $client.Connected }
    catch { return $false }
    finally { $client.Dispose() }
}

function Show-LaunchError([string]$Message) {
    if ($FromShortcut) {
        $shell = New-Object -ComObject WScript.Shell
        [void]$shell.Popup($Message, 0, 'Odysseus could not start', 16)
    } else {
        Write-Error $Message
    }
}

if ($CreateShortcut) {
    if ($env:OS -ne 'Windows_NT') { throw '-CreateShortcut is available on Windows.' }
    $shortcutPath = Join-Path $PSScriptRoot 'odysseus.lnk'
    $target = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
    $shortcutShell = New-Object -ComObject WScript.Shell
    $shortcut = $shortcutShell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $target
    $shortcut.Arguments = '-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}" -FromShortcut -Port {1} -ApiPort {2}' -f $PSCommandPath, $Port, $ApiPort
    $shortcut.WorkingDirectory = $PSScriptRoot
    $shortcut.WindowStyle = 7
    $shortcut.Description = 'Start Locally and open Odysseus'
    $icon = Join-Path $PSScriptRoot 'static\icons\locally.ico'
    if (Test-Path -LiteralPath $icon) { $shortcut.IconLocation = "$icon,0" }
    $shortcut.Save()

    $startMenu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\Odysseus.lnk'
    Copy-Item -LiteralPath $shortcutPath -Destination $startMenu -Force
    Write-Host "Installed Start-menu shortcut: $startMenu"
    if ($Desktop) {
        $desktopPath = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Odysseus.lnk'
        Copy-Item -LiteralPath $shortcutPath -Destination $desktopPath -Force
        Write-Host "Installed Desktop shortcut: $desktopPath"
    }
    return
}

$apiRunning = Test-LocalPort $ApiPort
$odysseusRunning = Test-LocalPort $Port
if ($WhatIf) {
    [pscustomobject]@{
        ApiRunning = $apiRunning
        OdysseusRunning = $odysseusRunning
        StartApi = -not $apiRunning
        StartOdysseus = -not $odysseusRunning
        Open = $odysseusUrl
    } | ConvertTo-Json -Compress
    return
}

if (-not $apiRunning) {
    $apiScript = Join-Path $PSScriptRoot 'api.ps1'
    if (-not (Test-Path -LiteralPath $apiScript)) {
        Show-LaunchError "Locally's api.ps1 was not found in $PSScriptRoot."
        exit 1
    }
    $hostCommand = Get-Command pwsh -ErrorAction SilentlyContinue
    if (-not $hostCommand) { $hostCommand = Get-Command powershell -ErrorAction SilentlyContinue }
    if (-not $hostCommand) {
        Show-LaunchError 'PowerShell was not found, so Locally could not start.'
        exit 1
    }
    $quotedApi = "'" + ($apiScript -replace "'", "''") + "'"
    $command = "& $quotedApi -Port $ApiPort"
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
    Start-Process -FilePath $hostCommand.Source -WindowStyle Normal `
        -ArgumentList "-NoProfile -EncodedCommand $encoded"

    $deadline = [DateTime]::UtcNow.AddSeconds(90)
    do {
        Start-Sleep -Milliseconds 250
        $apiRunning = Test-LocalPort $ApiPort
    } until ($apiRunning -or [DateTime]::UtcNow -ge $deadline)
    if (-not $apiRunning) {
        Show-LaunchError "Locally did not open port $ApiPort within 90 seconds. Check its API terminal."
        exit 1
    }
}

if (-not $odysseusRunning) {
    $python = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
    $entry = Join-Path $PSScriptRoot 'locally.py'
    if (-not (Test-Path -LiteralPath $python)) {
        Show-LaunchError 'Run install.ps1 first so Locally can start Odysseus.'
        exit 1
    }
    try {
        $output = @(& $python $entry odysseus start --port "$Port" --timeout 240 2>&1)
        if ($LASTEXITCODE -ne 0) {
            $detail = $output -join "`n"
            try {
                $parsed = $detail | ConvertFrom-Json
                if ($parsed.detail) { $detail = $parsed.detail }
            } catch { }
            throw $detail
        }
        $result = (($output -join "`n") | ConvertFrom-Json)
        $odysseusRunning = [bool]$result.running
    } catch {
        $detail = $_.Exception.Message
        Show-LaunchError "Odysseus did not start.`n`n$detail"
        exit 1
    }
}

if (-not $odysseusRunning) {
    Show-LaunchError "Odysseus did not answer on port $Port."
    exit 1
}

Start-Process $odysseusUrl
Write-Host "Opened $odysseusUrl"
