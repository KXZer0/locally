<#
.SYNOPSIS
    Three user32 calls, emitted with no C# compiler. Shared by the launcher
    and the hardware-key hot path.

.DESCRIPTION
    This lives in its own file because the two callers drifted: the
    Reflection.Emit rewrite landed in locally-key.ps1 and locally-launch.ps1
    was left calling Add-Type, still paying the runtime C# compiler that the
    rewrite existed to remove (measured 336 ms on this box, pwsh 7). One copy
    cannot drift from itself.

    Add-Type shells out to csc.exe. Reflection.Emit declares the same
    P/Invokes in-process. Measured here: Add-Type 336 ms, emit ~125 ms cold.

    WORKS ON BOTH HOSTS. .NET Core removed DefineDynamicAssembly from
    AppDomain, so the 5.1 entry point throws on pwsh 7 — but the static
    AssemblyBuilder.DefineDynamicAssembly is there, and DefinePInvokeMethod
    itself works fine on Core. Verified on pwsh 7: AppDomain path FAILED,
    static path OK, emitted IsIconic callable. This is why locally-key.ps1
    no longer has to warn that pwsh 7 can only AppActivate.

    PreserveSig is NOT optional and its absence is silent. Without it the CLR
    treats each BOOL return as an HRESULT and rewrites the signature, so the
    value is garbage: measured on a genuinely minimised window, Add-Type's
    IsIconic said True and the emitted one said False, which made the restore
    step quietly never run. Any bool-returning P/Invoke added here needs it.

.OUTPUTS
    The emitted type, or $null if emit is unavailable (caller falls back to
    the WScript.Shell COM AppActivate, which focuses but cannot un-minimise).
#>

function Get-Win32 {
    try {
        $an = New-Object Reflection.AssemblyName 'locallyKeyWin'
        try {
            # Windows PowerShell 5.1
            $asm = [AppDomain]::CurrentDomain.DefineDynamicAssembly(
                       $an, [Reflection.Emit.AssemblyBuilderAccess]::Run)
        } catch {
            # pwsh 7 / .NET Core: same emit, different entry point.
            $asm = [Reflection.Emit.AssemblyBuilder]::DefineDynamicAssembly(
                       $an, [Reflection.Emit.AssemblyBuilderAccess]::Run)
        }
        $tb = $asm.DefineDynamicModule('locallyKeyWin').DefineType('U32', 'Public, Class')
        $methods = @()
        foreach ($m in 'SetForegroundWindow', 'IsIconic') {
            $methods += $tb.DefinePInvokeMethod($m, 'user32.dll', 'Public, Static',
                [Reflection.CallingConventions]::Standard, [bool], @([IntPtr]),
                [Runtime.InteropServices.CallingConvention]::Winapi,
                [Runtime.InteropServices.CharSet]::Auto)
        }
        $methods += $tb.DefinePInvokeMethod('ShowWindow', 'user32.dll', 'Public, Static',
            [Reflection.CallingConventions]::Standard, [bool], @([IntPtr], [int]),
            [Runtime.InteropServices.CallingConvention]::Winapi,
            [Runtime.InteropServices.CharSet]::Auto)
        foreach ($m in $methods) {
            $m.SetImplementationFlags($m.GetMethodImplementationFlags() -bor
                [Reflection.MethodImplAttributes]::PreserveSig)
        }
        return $tb.CreateType()
    } catch {
        return $null
    }
}

# ALL matches, not the first: a Chromium app window is reported by more than
# one of the browser's processes, so "first match" picks a different handle
# run to run — measured, the old exe grabbed a handle whose IsIconic said
# False while the real window sat minimised.
#
# [Process]::GetProcesses() rather than the Get-Process cmdlet: measured
# 15 ms vs 38 ms here, because the cmdlet builds a decorated PSObject for
# every process on the box.
function Find-locallyWindows {
    $hits = @()
    foreach ($p in [System.Diagnostics.Process]::GetProcesses()) {
        try {
            if ($p.MainWindowHandle -eq [IntPtr]::Zero) { continue }
            if ($p.MainWindowTitle -like '*locally*') { $hits += $p }
        } catch { }   # process exited mid-enumeration
    }
    return $hits
}

# Raise every candidate until one takes. Returns the process that accepted
# focus, or $null. Restore-if-minimised first so pressing the key while
# already focused feels like toggle-to-front rather than a no-op.
function Show-locallyWindow {
    $procs = Find-locallyWindows
    if (-not $procs.Count) { return $null }
    $u32 = Get-Win32
    foreach ($p in $procs) {
        $h = $p.MainWindowHandle
        if ($u32) {
            if ($u32::IsIconic($h)) { [void]$u32::ShowWindow($h, 9) }   # SW_RESTORE
            if ($u32::SetForegroundWindow($h)) { return $p }
        } else {
            try { if ((New-Object -ComObject WScript.Shell).AppActivate($p.Id)) { return $p } } catch { }
        }
    }
    return $null
}

# App mode gives locally its own window with its own title and taskbar entry,
# which is what makes "press the key again to come back" possible at all. A
# plain tab inside a big browser window cannot be raised individually from
# outside the browser.
#
# [IO.File]::Exists, NOT Test-Path. Measured on PS 5.1: the FIRST Test-Path in
# a session costs 144 ms of provider initialisation and every later one costs
# 0 ms, while [IO.File]::Exists costs 2 ms cold. The four disk checks were
# never the expense. These are absolute filesystem paths, so nothing of value
# is lost by skipping the provider layer.
function Get-AppModeBrowser {
    foreach ($p in @(
        "$env:ProgramFiles\Google\Chrome\Application\chrome.exe"
        "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe"
        "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe"
        "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
    )) {
        if ([IO.File]::Exists($p)) { return $p }
    }
    return $null
}

# An installed PWA is the actual Windows app identity: its shortcut points at
# msedge_proxy.exe with Edge's generated --app-id, which is what gives the
# window the locally icon, Start-menu identity and its own taskbar grouping.
# A raw `msedge.exe --app=http://...` looks similar but is only an anonymous
# browser window. Edge normally creates the first path below; the others cover
# users who chose a Start-menu shortcut instead of a Desktop one.
function Get-InstalledlocallyShortcut {
    $desktop = [Environment]::GetFolderPath('Desktop')
    $start = [Environment]::GetFolderPath('StartMenu')
    foreach ($p in @(
        (Join-Path $desktop 'locally.lnk')
        (Join-Path $start 'Programs\locally.lnk')
        (Join-Path $start 'Programs\Edge Apps\locally.lnk')
    )) {
        if ([IO.File]::Exists($p)) { return $p }
    }
    return $null
}

# Real app window, no browser at all: locally_app.py wraps pywebview, which
# drives the WebView2 runtime directly. Preferred over every Edge app-mode
# path below -- no tabs, no address bar, no browser profile, and the taskbar
# entry is "locally", not "Microsoft​ Edge". Optional by design: pywebview
# is not in requirements.txt (only the native window needs it), so a fast
# `python -c "import webview"` probe decides whether to even try, rather
# than paying a failed Start-Process + waiting on a window that never
# appears. ~175 ms measured on this box -- cheap next to the rest of the
# launch path.
#
# $PSScriptRoot here is THIS file's directory (scripts\), not the caller's,
# even when dot-sourced -- verified empirically, since it is what every
# other path in this file already relies on implicitly. That gives the repo
# root without a parameter every caller would otherwise have to thread
# through (locally-key.ps1's Open-locallyApp call takes one argument today;
# adding a required $Root would mean touching that hot path too).
$locallyRoot = Split-Path $PSScriptRoot -Parent

function Test-PywebviewAvailable([string] $Python) {
    if (-not [IO.File]::Exists($Python)) { return $false }
    & $Python -c 'import webview' *> $null
    return $LASTEXITCODE -eq 0
}

# python.exe is a console-subsystem binary: launching it flashes/keeps a
# console window behind the pywebview window even under -WindowStyle Hidden.
# pythonw.exe is the same interpreter built against the GUI subsystem --
# byte-identical behaviour, no console ever created. Only the probe above
# still needs python.exe (it reads $LASTEXITCODE, which needs a real stdio
# handshake).
#
# Renaming that exe (venv\Scripts\pythonw.exe -> ...\locally.exe) to fix
# Task Manager's "python" label was tried and reverted -- it runs, but
# doesn't fix anything. On this box's Python install (the newer per-version
# "pythoncore" layout under %LocalAppData%\Python), venv\Scripts\pythonw.exe
# is a thin redirector that spawns a CHILD process from the real interpreter
# at pythoncore-3.14-64\pythonw.exe, and that child -- not the renamed
# stub -- is what actually owns the window and runs locally_app.py.
# Verified: launching the renamed copy produced two processes
# (locally.exe, idle, no window) -> child (pythonw.exe, ParentProcessId
# pointing at the renamed one, MainWindowTitle "locally"), confirmed via
# Get-CimInstance Win32_Process's ExecutablePath/ParentProcessId. Task
# Manager would still read "pythonw.exe" off the working process. Renaming
# only relabels the inert parent.
function Open-locallyNativeWindow([string] $Url) {
    $python = Join-Path $locallyRoot 'venv\Scripts\python.exe'
    if (-not (Test-PywebviewAvailable $python)) { return $null }
    $app = Join-Path $locallyRoot 'locally_app.py'
    if (-not ([IO.File]::Exists($app))) { return $null }
    $pythonw = Join-Path $locallyRoot 'venv\Scripts\pythonw.exe'
    $launcher = if ([IO.File]::Exists($pythonw)) { $pythonw } else { $python }
    try {
        Start-Process -FilePath $launcher -ArgumentList $app, '--url', $Url `
            -WorkingDirectory $locallyRoot -WindowStyle Hidden -ErrorAction Stop
        return 'pywebview native window'
    } catch { return $null }
}

# Open the installed app when it matches the manifest's fixed localhost URL;
# otherwise retain the anonymous Chromium app-window fallback. Returns a short
# description for the launch log, or $null when neither path could open.
function Open-locallyApp([string] $Url) {
    $native = Open-locallyNativeWindow $Url
    if ($native) { return $native }

    # Both spellings of the loopback host: the callers now pass 127.0.0.1 so
    # the native window skips the IPv6 stall 'localhost' can cause, but the
    # PWA manifest is registered against the literal localhost URL, so this
    # branch has to recognise either or the installed-app path silently
    # stops being reachable.
    if ($Url -match '^http://(localhost|127\.0\.0\.1):8000/?$') {
        $installed = Get-InstalledlocallyShortcut
        if ($installed) {
            try {
                Start-Process -FilePath $installed -ErrorAction Stop
                return "installed PWA ($(Split-Path $installed -Leaf))"
            } catch { }
        }
    }

    $browser = Get-AppModeBrowser
    if ($browser) {
        try {
            Start-Process -FilePath $browser `
                -ArgumentList "--app=$Url", "--window-size=1280,860" `
                -ErrorAction Stop
            return "$(Split-Path $browser -Leaf) app window"
        } catch { }
    }
    return $null
}
