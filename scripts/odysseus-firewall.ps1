#requires -Version 7.0
#requires -RunAsAdministrator
<#
.SYNOPSIS
    Allow the local Odysseus container to reach locally's OpenAI API.

.DESCRIPTION
    Creates one inbound Windows Firewall rule restricted to the Python process
    currently listening on locally's port, the WSL virtual adapter, and that
    adapter's private subnet. It does not open the port on Wi-Fi or Ethernet.
#>
param([int] $Port = 8000)

$ErrorActionPreference = 'Stop'
$ruleName = 'locally (Podman to API, scoped)'
$oldRuleName = 'locally (WSL/Podman)'
$adapterName = 'vEthernet (WSL (Hyper-V firewall))'

$adapter = Get-NetIPAddress -InterfaceAlias $adapterName -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike '169.254.*' } |
    Select-Object -First 1
if (-not $adapter) {
    throw "The WSL virtual adapter is not available. Start the Podman machine first."
}

$bytes = ([Net.IPAddress]::Parse($adapter.IPAddress)).GetAddressBytes()
$bits = [int]$adapter.PrefixLength
for ($i = 0; $i -lt $bytes.Length; $i++) {
    $remaining = [Math]::Max(0, [Math]::Min(8, $bits - ($i * 8)))
    $mask = if ($remaining -eq 0) { 0 } else { (0xff -shl (8 - $remaining)) -band 0xff }
    $bytes[$i] = $bytes[$i] -band $mask
}
$subnet = "$([Net.IPAddress]::new($bytes))/$($adapter.PrefixLength)"

$listener = Get-NetTCPConnection -LocalPort $Port -State Listen |
    Select-Object -First 1
if (-not $listener) {
    throw "Nothing is listening on TCP port $Port. Start locally first."
}
$program = (Get-Process -Id $listener.OwningProcess).Path
if (-not $program) {
    throw "Could not resolve the executable listening on TCP port $Port."
}

# A prior "Cancel" on Python's Windows Firewall prompt creates an explicit
# program-wide Block rule. Windows gives Block precedence over every Allow, so
# our narrow rule cannot work while that TCP rule is enabled. Disable only the
# matching TCP rule; the firewall's default inbound block remains in force and
# the scoped rule below is still the sole exception. Leave UDP untouched.
$disabledBlocks = @()
Get-NetFirewallRule -Direction Inbound -Action Block -Enabled True |
    ForEach-Object {
        $candidate = $_
        $candidateProgram = ($candidate | Get-NetFirewallApplicationFilter).Program
        $candidateProtocol = ($candidate | Get-NetFirewallPortFilter).Protocol
        if ($candidateProgram -ieq $program -and $candidateProtocol -eq 'TCP') {
            $candidate | Set-NetFirewallRule -Enabled False
            $disabledBlocks += $candidate.Name
        }
    }

Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue |
    Remove-NetFirewallRule
# Remove the older port-only rule if an earlier setup attempt created it.
Get-NetFirewallRule -DisplayName $oldRuleName -ErrorAction SilentlyContinue |
    Remove-NetFirewallRule

New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow `
    -Program $program -Protocol TCP -LocalPort $Port -RemoteAddress $subnet `
    -Profile Any -InterfaceAlias $adapterName | Out-Null

Write-Output "Created '$ruleName'"
Write-Output "  Program: $program"
Write-Output "  Port:    $Port/tcp"
Write-Output "  Source:  $subnet through $adapterName"
if ($disabledBlocks.Count) {
    Write-Output "  Disabled conflicting TCP block: $($disabledBlocks -join ', ')"
}
