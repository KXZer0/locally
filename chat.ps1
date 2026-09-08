<# Start the terminal chat. Attach when the API is up; otherwise own a server
   for this session so /exit or Ctrl+C shuts down only that owned server. #>
param(
    [int]$Port = 8000,
    [string]$Url = '',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = 'Stop'
$python = if ($env:LOCALLY_PYTHON) { $env:LOCALLY_PYTHON } else { Join-Path $PSScriptRoot 'venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $python)) { throw 'Run install.ps1 first to create the server environment.' }

if ($ExtraArgs -contains '--url') {
    throw 'Use chat.ps1 -Url http://host:port instead of forwarding --url.'
}

$serving = $false
if (-not $Url) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try { $serving = $client.ConnectAsync('127.0.0.1', $Port).Wait(250) -and $client.Connected }
    catch { $serving = $false }
    finally { $client.Dispose() }
}

$arguments = @((Join-Path $PSScriptRoot 'locally.py'), 'chat')
if ($Url) { $arguments += @('--url', $Url) }
else {
    $arguments += @('--port', "$Port")
    if (-not $serving) { $arguments += '--start' }
}
$arguments += $ExtraArgs
& $python @arguments
exit $LASTEXITCODE
