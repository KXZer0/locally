"""Native app-shell window for locally, replacing "open it in Edge app-mode".

Thin wrapper around pywebview (which drives the already-installed WebView2
runtime): open a chromeless window pointed at a URL that is *already*
serving. This script never starts the server itself -- locally-launch.ps1
owns that, and starting a second server from here would defeat the
one-server-per-port idempotency that script is built around.

Runtime flags over hardcoded config, per CLAUDE.md: URL and window size are
CLI args, not constants, so the same script serves the throwaway-server
verification path and the real one without editing.

Window title: "locally", exactly the string locally-win32.ps1's
Find-locallyWindows matches on (`$p.MainWindowTitle -like '*locally*'`). That
function is what lets the launcher/hardware-key raise an already-open window
instead of piling up a second one, and it does that purely by process
enumeration -- it never talks to the browser -- so the title is the entire
contract. templates/index.html's own <title> is also "locally"; matching it
here means a page that later renames its own <title> via JS would still be
found (pywebview's window title stays what we set, not what the DOM does),
which is the more predictable behaviour for a launcher script.
"""
import argparse
import atexit
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request

WINDOW_TITLE = "locally"

# static/icons/locally.ico sits next to this script regardless of the
# caller's cwd (locally-win32.ps1 sets -WorkingDirectory to the repo root,
# but don't rely on that holding forever).
ICON_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "static", "icons", "locally.ico"
)


def _set_app_identity() -> None:
    """Give the process its own taskbar identity before any window exists.

    Without an explicit AppUserModelID, Windows buckets every pythonw.exe
    window under a generic "Python" group (and can silently revert to the
    Python icon for it), which is also what blocks pinning this window as
    its own app rather than as "Python". Must run before window creation --
    the shell reads this once, at first-window time. No-op (silently) off
    Windows, since shell32/AppUserModelID is a Windows-only concept.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("locally.app")
    except Exception:
        pass  # cosmetic only -- never block the window over this


def _url_is_up(url: str, timeout: float) -> bool:
    """Best-effort reachability check. GET, not just a socket connect: an
    HTML file server on the port answers with a real response body we can
    show verbatim; a bare TCP check would pass for anything listening at
    all, and locally-launch.ps1 already owns the "is locally itself up"
    question -- this script only needs to know whether IT has something to
    render before pywebview draws a blank/error page.

    Polls rather than asking once. The launcher starts the server and opens
    this window in the same breath, so a single check races the server's
    startup and loses: the fallback page then sticks forever, because
    nothing re-checks after the window has loaded. That is the "localhost
    is up but the app says it isn't" report -- the server was genuinely
    listening, just not yet when we asked.

    127.0.0.1 rather than localhost matters for the same reason: on Windows
    'localhost' can resolve to ::1 first, and if the server is bound to IPv4
    only, each attempt stalls on the IPv6 connect before falling back.
    """
    deadline = time.monotonic() + max(timeout, 0.0)
    attempt_timeout = min(2.0, max(timeout, 0.5))
    while True:
        try:
            with urllib.request.urlopen(url, timeout=attempt_timeout):
                return True
        except (urllib.error.URLError, OSError):
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.25)


def _not_serving_page(url: str) -> str:
    """Write the "nothing is listening" fallback to a temp file and return a
    file:// URL to it, so the window loads it through the ordinary
    load_url() -> Source= navigation path -- the one already proven to
    render (verified against a real HTTP server, see the repo's launch
    verification notes). Two things were tried and rejected first:
    create_window(url="data:text/html,...") -- pywebview's edgechromium
    backend routes url= through its own local bottle server, which 404'd
    trying to resolve the data: string as a path (confirmed: the window
    showed bottle's own 404 page); and create_window(html=...)
    (NavigateToString) -- the window painted blank on screen here, even
    though pywebview's own event fired (DOM content was verifiable via
    evaluate_js, so the navigation itself may have been fine and this may
    be a capture/paint quirk of NavigateToString specifically rather than a
    real bug -- inconclusive either way). The url= + real navigation path
    below is the one with an unambiguous, reproducible pass, so it's what
    ships. The temp file is small and one-shot; the OS reclaims %TEMP%
    eventually, so no cleanup ceremony is needed beyond best-effort atexit.
    """
    html = (
        "<html><head><title>{title}</title>"
        "<style>body{{background:#111;color:#eee;font:15px system-ui;"
        "display:flex;align-items:center;justify-content:center;height:100vh;"
        "margin:0;text-align:center}}"
        "div{{max-width:32em}}code{{color:#8cf}}</style></head>"
        "<body><div><h2>locally isn't running</h2>"
        "<p>Nothing answered at <code>{url}</code>.</p>"
        "<p>Start the server, then reopen this window.</p>"
        "</div></body></html>"
    ).format(title=WINDOW_TITLE, url=url)

    fd, path = tempfile.mkstemp(prefix="locally-app-not-running-", suffix=".html")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(html)
    atexit.register(lambda: os.path.exists(path) and os.remove(path))
    return "file:///" + path.replace("\\", "/")


_WINDOW = None


def _own_visible_window(user32):
    """Our process's visible top-level window handle, or 0.

    Found by process id rather than by title, because the title is exactly
    what frameless takes away -- looking it up by name is the bug this
    function exists to fix.
    """
    import ctypes
    from ctypes import wintypes

    found = []
    me = os.getpid()

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _visit(hwnd, _lparam):
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value == me and user32.IsWindowVisible(hwnd):
            found.append(hwnd)
            return False  # stop at the first match
        return True

    user32.EnumWindows(_visit, 0)
    return found[0] if found else 0


def _kill_dwm_frame(hwnd) -> None:
    """Give the frameless window a clean rounded shape with no border line.

    WS_THICKFRAME is restored above so the window can be resized, and DWM
    draws a frame for it. The window style itself is clean (measured
    0x160F0000 -- no WS_CAPTION, no WS_BORDER), so anything visible at the
    edge is DWM, and it has to be turned off through DWM, not through styles.

    Do NOT reach for NCRENDERING_POLICY = DISABLED here. It was tried: it
    stops DWM compositing the frame, which exposes the raw resize border as
    a hard white stroke and squares the corners. The window ended up looking
    worse than the translucent strip it was meant to remove.

    What actually works is telling DWM what to draw rather than to stop:
      - WINDOW_CORNER_PREFERENCE = ROUND keeps the Windows 11 corner radius,
        which is the shape a native app has.
      - BORDER_COLOR = COLOR_NONE removes the border line on every edge
        while leaving the frame composited, so the rounding and the drop
        shadow survive and nothing is stroked.

    Best-effort: a failure here is cosmetic and must never stop the window
    opening, hence the swallowed exception. Both attributes need Windows 11
    (build 22000+); on Windows 10 they are ignored, which is the right
    degradation -- a square window, not a broken one.
    """
    import ctypes

    DWMWA_USE_IMMERSIVE_DARK_MODE = 20
    DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND = 33, 2
    DWMWA_BORDER_COLOR, DWMWA_COLOR_NONE = 34, 0xFFFFFFFE
    DWMWA_SYSTEMBACKDROP_TYPE, DWMSBT_NONE = 38, 1
    try:
        dwm = ctypes.windll.dwmapi
        for attr, value in (
            # Order matters only in that dark mode should land before the
            # colours: it changes which palette DWM composites the rest from.
            (DWMWA_USE_IMMERSIVE_DARK_MODE, 1),
            (DWMWA_SYSTEMBACKDROP_TYPE, DWMSBT_NONE),
            (DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND),
            (DWMWA_BORDER_COLOR, DWMWA_COLOR_NONE),
        ):
            v = ctypes.c_uint(value)
            dwm.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(v),
                                      ctypes.sizeof(v))
    except Exception:
        pass


def _relayout_webview(user32, hwnd) -> None:
    """Force WinForms to re-dock the WebView2 child control onto the
    now-correct client rect, closing a solid black strip left at the TOP
    edge only.

    Measured (screenshot pixel-scanned against a white desktop, see repo
    verification notes): after _restore_window_affordances adds
    WS_THICKFRAME and calls SetWindowPos(..., SWP_FRAMECHANGED |
    SWP_NOMOVE | SWP_NOSIZE), the LEFT/RIGHT/BOTTOM edges are clean --
    DWM's shadow gradient blends directly into real page pixels (17,17,17).
    The TOP edge instead showed shadow, then 10 rows of solid (0,0,0) --
    not the page's own near-black background -- then real content. That
    0,0,0 is the bare WinForms Form showing through: the Form's
    WM_NCCALCSIZE-computed client rect just grew (SWP_FRAMECHANGED asked
    Windows to recompute the non-client area), but a frame-change alone
    does not fire WM_SIZE, and it's WM_SIZE that WinForms' docked-child
    layout (the WebView2 control filling the form) reacts to. So the
    control kept its pre-frame size/position -- correct for every side
    except top, where the new frame geometry happens to be asymmetric
    (WM_NCCALCSIZE's default top inset differs from its left/right/bottom
    one once WS_CAPTION is absent but WS_THICKFRAME is present) -- and a
    sliver of unrendered form background was exposed there.
    SWP_FRAMECHANGED recomputes hit-testing/NC geometry bookkeeping; it is
    explicitly documented to NOT imply a resize notification, hence this
    second, separate nudge.

    Fix: force an actual WM_SIZE by changing the outer size and changing
    it back. A same-size SetWindowPos is not reliably enough to trigger
    WinForms' OnResize/PerformLayout; a genuine +1/-1 px round trip is.
    One pixel is imperceptible on screen and this runs once at startup,
    before the user is looking at a steady-state window.
    """
    import ctypes
    from ctypes import wintypes

    class _RECT(ctypes.Structure):
        _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                    ("right", wintypes.LONG), ("bottom", wintypes.LONG)]

    SWP_NOMOVE, SWP_NOZORDER, SWP_NOACTIVATE = 0x0002, 0x0004, 0x0010
    rect = _RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return
    w = rect.right - rect.left
    h = rect.bottom - rect.top
    flags = SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE
    user32.SetWindowPos(hwnd, 0, 0, 0, w + 1, h, flags)
    user32.SetWindowPos(hwnd, 0, 0, 0, w, h, flags)


def _restore_window_affordances() -> None:
    """Put back the window styles frameless strips but the page cannot redraw.

    Measured on this box with GetWindowLong before this ran: frameless gives
    WS_CAPTION False (intended -- the page draws its own bar) but ALSO
    WS_THICKFRAME False and WS_SYSMENU False, and those two are not cosmetic:

      - WS_SYSMENU is what makes a window answer WM_CLOSE. Without it the
        process SURVIVES Alt+F4 and any external close -- verified: the
        window stayed alive with 107 MB resident long after CloseMainWindow.
        For an app whose whole point is releasing memory when it closes,
        that is the worst possible failure.

    WS_SYSMENU draws nothing while WS_CAPTION is off, so restoring it costs
    no pixels. WS_MINIMIZEBOX/WS_MAXIMIZEBOX go back for the taskbar's
    minimise and restore animations, which the custom buttons drive.

    **WS_THICKFRAME is deliberately NOT restored, and must not be.** It is
    the resize border, and it is not free: measured with GetWindowRect
    against GetClientRect while it was set, the client area was inset
    6px/6px/6px/7px, and that inset is a visible strip around the page --
    the "bar at the top" this window kept being sent back for. The strip is
    the frame itself, so no amount of DWM tuning removes it; three attempts
    were made and all failed (NCRENDERING_POLICY=DISABLED turned it into a
    hard white stroke with square corners, BORDER_COLOR=COLOR_NONE and
    IMMERSIVE_DARK_MODE left it visible). Dropping the style takes the inset
    to 0 on every edge, which is what "truly frameless" means here.

    The cost is that the window CANNOT be resized by dragging its edges --
    an explicit, informed trade the owner asked for ("i dont mind no upper
    resize really i just want that ugly bar at the top gone"). Maximise and
    restore still work, because those go through WindowState rather than the
    sizing border. If drag-resize is ever wanted back, the answer is NOT to
    re-add this style: it is a WM_NCCALCSIZE window-proc subclass that
    collapses the non-client area to zero while keeping the style, which is
    how Electron does it.

    Runs as webview.start(func) -- i.e. after the window exists. The retry
    loop is because "the window exists" and "the OS handle is findable by
    title" are not the same instant.
    """
    if not sys.platform.startswith("win"):
        return
    import ctypes

    GWL_STYLE, SWP_FRAMECHANGED = -16, 0x0020
    SWP_NOMOVE, SWP_NOSIZE, SWP_NOZORDER = 0x0002, 0x0001, 0x0004
    WS_SYSMENU = 0x00080000
    WS_MINIMIZEBOX, WS_MAXIMIZEBOX = 0x00020000, 0x00010000

    user32 = ctypes.windll.user32
    for _ in range(40):
        hwnd = _own_visible_window(user32)
        if hwnd:
            # Frameless also blanks the window TEXT -- measured:
            # FindWindowW(None, "locally") returned 0 and MainWindowTitle was
            # '' for every process. That is not cosmetic either:
            # scripts/locally-win32.ps1 raises an already-open window by
            # matching MainWindowTitle -like '*locally*', so without the text
            # a second key press launches a SECOND instance (another ~420 MB
            # and a second WebView2 tree) instead of raising the first.
            user32.SetWindowTextW(hwnd, WINDOW_TITLE)
            style = user32.GetWindowLongW(hwnd, GWL_STYLE)
            user32.SetWindowLongW(hwnd, GWL_STYLE, style | WS_SYSMENU
                                  | WS_MINIMIZEBOX | WS_MAXIMIZEBOX)
            # Without SWP_FRAMECHANGED the new styles sit in the window's
            # data but the frame is never recalculated, so hit-testing keeps
            # using the old one and nothing appears to change.
            user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, SWP_FRAMECHANGED
                                | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER)
            _kill_dwm_frame(hwnd)
            _relayout_webview(user32, hwnd)
            return
        time.sleep(0.05)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Native app-shell window for a locally-compatible web UI "
        "already serving at --url. Does not start the server."
    )
    parser.add_argument(
        # 127.0.0.1, not localhost: see _url_is_up on the IPv6 stall.
        "--url", default="http://127.0.0.1:8000",
        help="URL to display (default: http://127.0.0.1:8000)",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=860)
    parser.add_argument(
        # 20s, not 2s: the launcher starts the server and this window
        # together, and Flask does not bind instantly. 2s lost that race and
        # pinned the fallback page over a server that came up moments later.
        "--wait-secs", type=float, default=20.0,
        help="how long to keep polling --url before falling back to the "
        "not-running page (default: 20)",
    )
    args = parser.parse_args()

    # Before any window exists, not just before webview.start() -- see
    # _set_app_identity's docstring.
    _set_app_identity()

    try:
        import webview
    except ImportError:
        print(
            "pywebview is not installed. Install it with:\n"
            "  venv\\Scripts\\python.exe -m pip install pywebview\n"
            "(it is optional -- the server itself does not need it)",
            file=sys.stderr,
        )
        return 1

    # The window is created NOW, not after the server answers.
    #
    # This used to be `_url_is_up(url, 20.0)` on the main thread before
    # create_window, so pressing the hardware key on a cold machine produced
    # NOTHING for as long as the server took to bind -- no window, no taskbar
    # button, no feedback of any kind. The rational response to that is to
    # press the key again, which is exactly what people did.
    #
    # So: one fast probe, and if the server is not up yet we open on the
    # waiting page immediately and swap to the real URL from a background
    # thread the moment it binds. The taskbar button exists from the first
    # frame either way, and the models keep loading behind it -- the server
    # serves the page long before the model is ready.
    already_up = _url_is_up(args.url, 0.35)
    target = args.url if already_up else _not_serving_page(args.url)

    # The OS frame is gone (frameless), so the page draws its own title bar and
    # window buttons -- see .window-controls in templates/index.html, revealed
    # only when shell.js finds window.pywebview. The page calls back through
    # this object; every method is defensive because the JS side ships whether
    # or not it is running inside the shell.
    #
    # easy_drag is OFF on purpose. It makes the WHOLE window a drag handle,
    # which in a frameless app means click-drag anywhere -- including across
    # the chat transcript -- moves the window instead of selecting text. The
    # page marks its top bar .pywebview-drag-region instead, so only that
    # strip drags.
    # The window is held in a module global, NOT as an attribute on Api.
    # pywebview's JS bridge introspects the js_api object's attributes to
    # decide what to expose to JavaScript, and an attribute holding the
    # Window makes it walk window.native.browser.webview.* from the bridge
    # thread -- where every WebView2 COM property throws "can only be
    # accessed from the UI thread" and the window never renders. Cost me a
    # blank launch; the global costs nothing.
    state = {"maximized": False}

    class Api:
        def minimize(self):
            if _WINDOW:
                _WINDOW.minimize()

        def toggle_maximize(self):
            # pywebview exposes no is-maximized query, so the state is
            # tracked here. Both sides start from "not maximized", which is
            # how the window is created, so the page's optimistic icon swap
            # and this stay in step.
            if not _WINDOW:
                return False
            _WINDOW.restore() if state["maximized"] else _WINDOW.maximize()
            state["maximized"] = not state["maximized"]
            return state["maximized"]

        def close(self):
            if _WINDOW:
                _WINDOW.destroy()

    global _WINDOW
    _WINDOW = webview.create_window(
        WINDOW_TITLE, target, width=args.width, height=args.height,
        js_api=Api(), frameless=True, easy_drag=False,
    )
    # gui='edgechromium' pins the Windows backend to WebView2 explicitly
    # rather than letting pywebview probe and possibly fall back to the
    # legacy MSHTML (IE11) renderer, which cannot render a modern SPA.
    #
    # icon: pywebview 6.2.1's own docstring claims this is "Supported only
    # on GTK/QT" -- that's stale. Traced it in site-packages: edgechromium.py
    # doesn't create its own Form, it reuses winforms.py's BrowserForm
    # (Chromium.EdgeChrome(self, window, cache_dir) is instantiated FROM
    # BrowserForm.__init__), and BrowserForm unconditionally sets
    # self.Icon = Icon(_state['icon']) before branching on which renderer to
    # embed. Verified empirically too: launched a real window with this
    # icon= set, then read the icon Windows actually holds for that hwnd via
    # WM_GETICON (the exact call the taskbar itself uses to draw the
    # button) -- it returned this ICO's artwork, not the Python default. So
    # the Win32 WM_SETICON fallback this was expected to need turned out to
    # be dead code; not adding it.
    def _on_start():
        # Runs after the window exists. Restore the frame affordances first --
        # that is what this callback was always for -- then, if we opened on
        # the waiting page, watch for the server and navigate when it answers.
        _restore_window_affordances()
        if already_up:
            return
        if _url_is_up(args.url, args.wait_secs) and _WINDOW:
            try:
                _WINDOW.load_url(args.url)
            except Exception:
                # A window closed while we were waiting is the normal way to
                # cancel this, not an error worth a traceback on stderr.
                pass

    # private_mode=False is REQUIRED, and its absence is silent.
    #
    # pywebview defaults `private_mode=True` (verified against the installed
    # library, not the docs: inspect.signature(webview.start) reports
    # private_mode=True, storage_path=None). In that mode WebView2 keeps no
    # profile, so localStorage and cookies are discarded when the window
    # closes. Every preference this UI owns lives in localStorage -- the system
    # prompt, the voice, the Odysseus address, the turn-taking settings, and
    # the `locally-onboarding-complete` flag.
    #
    # The visible symptom was not "storage is broken", which is why this
    # survived: it was "first-run setup opens every single time" and "my
    # choices do not save". The server was remembering correctly the whole
    # time -- setup.json had the right answers and /v1/setup reported
    # first_run=false -- and the browser threw its half away on every close.
    #
    # storage_path is set explicitly rather than left to pywebview's default so
    # the profile lands beside the launch log locally already owns, and so the
    # location is greppable when someone next asks where the settings went.
    storage = os.path.join(
        os.environ.get("LOCALAPPDATA")
        or os.path.expanduser("~/.local/share"), "locally", "webview")
    try:
        os.makedirs(storage, exist_ok=True)
    except OSError:
        storage = None          # unwritable profile dir must not stop the app

    webview.start(_on_start, gui="edgechromium",
                  private_mode=False, storage_path=storage,
                  icon=ICON_PATH if os.path.isfile(ICON_PATH) else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
