"""Windows-only Shift+Enter shortcut for the foreground API console."""

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import subprocess
import sys
import threading


_STD_INPUT_HANDLE = -10
_KEY_EVENT = 0x0001
_VK_RETURN = 0x0D
_SHIFT_PRESSED = 0x0010
_CREATE_NEW_CONSOLE = 0x00000010


class _KeyEvent(ctypes.Structure):
    _fields_ = [
        ("key_down", ctypes.c_int),
        ("repeat_count", ctypes.c_ushort),
        ("virtual_key", ctypes.c_ushort),
        ("virtual_scan", ctypes.c_ushort),
        ("char", ctypes.c_wchar),
        ("control_key_state", ctypes.c_uint),
    ]


class _EventUnion(ctypes.Union):
    _fields_ = [("key_event", _KeyEvent), ("padding", ctypes.c_byte * 16)]


class _InputRecord(ctypes.Structure):
    _fields_ = [("event_type", ctypes.c_ushort), ("event", _EventUnion)]


def _is_shift_enter(record):
    """Whether a console record is the first key-down for Shift+Enter."""
    key = record.event.key_event
    return (record.event_type == _KEY_EVENT and bool(key.key_down)
            and key.virtual_key == _VK_RETURN
            and bool(key.control_key_state & _SHIFT_PRESSED))


def _open_chat(port, popen=subprocess.Popen):
    """Open an attached chat client in its own console; never owns this API."""
    entry = Path(__file__).resolve().parents[2] / "locally.py"
    return popen(
        [sys.executable, str(entry), "chat", "--url", f"http://127.0.0.1:{port}"],
        creationflags=_CREATE_NEW_CONSOLE,
    )


def start_shift_enter_chat(port):
    """Listen for Shift+Enter in a Windows console without touching its mode.

    A daemon is intentional: Ctrl+C keeps its usual process-wide shutdown
    behavior, and the listener ends with the API process.
    """
    if os.name != "nt" or not getattr(sys.stdin, "isatty", lambda: False)():
        return False

    kernel32 = ctypes.windll.kernel32
    kernel32.GetStdHandle.argtypes = (wintypes.DWORD,)
    kernel32.GetStdHandle.restype = wintypes.HANDLE
    kernel32.GetConsoleMode.argtypes = (wintypes.HANDLE,
                                        ctypes.POINTER(wintypes.DWORD))
    kernel32.GetConsoleMode.restype = wintypes.BOOL
    kernel32.ReadConsoleInputW.argtypes = (wintypes.HANDLE,
                                           ctypes.POINTER(_InputRecord),
                                           wintypes.DWORD,
                                           ctypes.POINTER(wintypes.DWORD))
    kernel32.ReadConsoleInputW.restype = wintypes.BOOL
    handle = kernel32.GetStdHandle(_STD_INPUT_HANDLE)
    invalid_handle = ctypes.c_void_p(-1).value
    if handle in (0, invalid_handle):
        return False
    mode = wintypes.DWORD()
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        return False

    def listen():
        enter_held = False
        while True:
            record = _InputRecord()
            read = wintypes.DWORD()
            if not kernel32.ReadConsoleInputW(handle, ctypes.byref(record), 1,
                                              ctypes.byref(read)):
                return
            if not read.value or record.event_type != _KEY_EVENT:
                continue
            key = record.event.key_event
            if key.virtual_key != _VK_RETURN:
                continue
            if not key.key_down:
                enter_held = False
                continue
            if _is_shift_enter(record) and not enter_held:
                enter_held = True
                try:
                    _open_chat(port)
                except OSError as error:
                    print(f"\n  Could not open terminal chat: {error}", flush=True)

    threading.Thread(target=listen, name="locally-shift-enter", daemon=True).start()
    return True
