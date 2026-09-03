"""Opt-in background update checks."""

import concurrent.futures
import json
import os
import subprocess
import threading
import time
import urllib.request

from flask import jsonify, request

from core import config

# ---------------------------------------------------------------------------
# Update checks
#
# Opt-in (--check-updates), and off by default on purpose. This is the only
# code in the server that contacts anything on the internet without the user
# asking for it in that request, and a tool whose pitch is "your data never
# leaves the machine" does not get to quietly poll four services on boot.
# Everything here also has to be unable to hurt a start: the work runs on a
# background thread, every failure is swallowed, and the cached answer is what
# the endpoint serves. On a plane it returns "unknown" and nothing waits.
# ---------------------------------------------------------------------------

_UPDATE_TTL = 86400          # a day; nobody needs to know sooner
_UPDATE_TIMEOUT = 6          # sources run together, so four dead ones cost ~6 s
_updates_cache = {"checked_at": 0.0, "sources": {}, "refreshing": False}
_updates_lock = threading.Lock()

# name -> (kind, locator, human URL). Kept as data so adding a project is a
# line, not a branch.
_UPDATE_SOURCES = {
    # locally does not need a release to exist: compare this checkout's HEAD
    # with public main. That keeps update notices useful between releases.
    "locally":  ("github_commit", "KXZer0/locally", "https://github.com/KXZer0/locally"),
    "odysseus": ("github_release", "odysseus-dev/odysseus", "https://github.com/odysseus-dev/odysseus"),
    "searxng":  ("github_release", "searxng/searxng", "https://github.com/searxng/searxng"),
    "opencode": ("npm", "opencode-ai", "https://github.com/anomalyco/opencode"),
}


def _fetch_latest(kind, locator):
    """Newest published version for one project, or None. Never raises."""
    try:
        if kind == "npm":
            url = f"https://registry.npmjs.org/{locator}/latest"
        elif kind == "github_commit":
            url = f"https://api.github.com/repos/{locator}/commits/main"
        else:
            url = f"https://api.github.com/repos/{locator}/releases/latest"
        req = urllib.request.Request(url, headers={
            # GitHub rejects a missing UA, and naming ourselves is the polite
            # thing when we are the ones doing the polling.
            "User-Agent": "locally-update-check",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=_UPDATE_TIMEOUT) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
        if kind == "npm":
            return data.get("version")
        if kind == "github_commit":
            return data.get("sha")
        return (data.get("tag_name") or data.get("name") or "").lstrip("v") or None
    except Exception:
        return None


def _local_version(name):
    """What is installed here, or None when we cannot tell.

    Deliberately returns None rather than guessing: reporting "0.0.0" would
    make every project look out of date forever, which trains people to ignore
    the notice — the exact failure this feature exists to avoid.
    """
    try:
        if name == "locally":
            out = subprocess.run(["git", "-C", config.SCRIPT_DIR, "rev-parse", "HEAD"],
                                 capture_output=True, text=True, timeout=5)
            return (out.stdout or "").strip() or None
        if name == "opencode":
            out = subprocess.run(["opencode", "--version"], capture_output=True,
                                 text=True, timeout=10, shell=(os.name == "nt"))
            return (out.stdout or "").strip().split()[0].lstrip("v") or None
        if name in ("odysseus", "searxng"):
            env_name = "ODYSSEUS_DIR" if name == "odysseus" else "SEARXNG_ROOT"
            candidates = [os.environ.get(env_name)]
            if name == "odysseus":
                candidates += [os.path.join(os.path.dirname(config.SCRIPT_DIR), "odysseus"),
                               os.path.join(os.path.expanduser("~"), "odysseus")]
            else:
                candidates += [config.SEARXNG_ROOT,
                               os.path.join(os.path.dirname(config.SCRIPT_DIR), "searxng-src")]
            for path in candidates:
                if not path or not os.path.isdir(os.path.join(path, ".git")):
                    continue
                out = subprocess.run(["git", "-C", path, "describe", "--tags",
                                      "--always"], capture_output=True, text=True,
                                     timeout=5)
                value = (out.stdout or "").strip().lstrip("v")
                if value:
                    return value
    except Exception:
        pass
    return None


def _version_differs(kind, installed, latest):
    """True only where both values are comparable, never merely different."""
    if not installed or not latest:
        return False
    if kind == "github_commit":
        return not (installed.startswith(latest) or latest.startswith(installed))
    # A dirty/tag-distance git describe is not comparable to a release tag.
    if "-g" in installed:
        return False
    return installed.lstrip("v") != latest.lstrip("v")


def _refresh_updates():
    """Re-check every source. Runs on a background thread; never raises."""
    def one(item):
        name, (kind, locator, page) = item
        latest = _fetch_latest(kind, locator)
        here = _local_version(name)
        return name, {
            "latest": latest, "installed": here, "url": page,
            # Only true when the forms are genuinely comparable. Unknown is
            # "cannot tell", never "out of date".
            "update_available": _version_differs(kind, here, latest),
        }

    try:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(_UPDATE_SOURCES)) as pool:
            sources = dict(pool.map(one, _UPDATE_SOURCES.items()))
    except Exception:
        sources = {}
    finally:
        with _updates_lock:
            _updates_cache["sources"] = sources
            _updates_cache["checked_at"] = time.time()
            _updates_cache["refreshing"] = False


def _schedule_update_refresh(force=False):
    """Start one refresh if needed. Returns whether a worker was started."""
    with _updates_lock:
        age = time.time() - _updates_cache["checked_at"]
        if _updates_cache["refreshing"] or (not force and age <= _UPDATE_TTL):
            return False
        _updates_cache["refreshing"] = True
    threading.Thread(target=_refresh_updates, daemon=True,
                     name="locally-update-check").start()
    return True


def updates():
    """What is newer than what is installed. Cached for a day.

    Never blocks: a stale or empty cache is returned immediately and the
    refresh happens behind it. `?refresh=1` forces one, still asynchronously.
    """
    if not config.CHECK_UPDATES:
        return jsonify({"enabled": False, "sources": {},
                        "reason": "Update checks are off. Start with "
                                  "--check-updates to enable them."})
    _schedule_update_refresh(force=bool(request.args.get("refresh")))
    with _updates_lock:
        payload = dict(_updates_cache)
    return jsonify({"enabled": True, "checked_at": payload["checked_at"],
                    "refreshing": payload["refreshing"],
                    "sources": payload["sources"]})
