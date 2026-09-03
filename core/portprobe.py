"""A liveness probe that never blocks the caller.

`/health` is polled by the web UI, so anything it calls is on a hot path. Two
"is it running?" probes were added to it (OpenCode's web server, Odysseus) and
each does a real TCP/HTTP round trip. Measured on the 358H, 2026-08-30, with
neither service running: `/health` cost **1.86 s per call**, every call, against
1.5 ms for `/v1/models` and 5 ms for `/v1/memory`. The whole UI polls that.

The reason it is so expensive is worth writing down, because it is not what you
would predict: on this machine a connection to a CLOSED loopback port is not
refused, it is dropped, so both probes sat until their own timeout expired
(1508 ms and 360 ms -- each one's configured limit, near exactly). A "fast fail"
that assumes ECONNREFUSED is not portable; something in the local security stack
can and does swallow the reset.

So the probe is inverted. Readers never wait for the network: they get the last
known answer immediately, and a background thread refreshes it when it goes
stale. The initial value is False -- "not running" is both the safe default and
the true answer almost always, since the common case is that the service was
never started.

The cost of this is that the answer can be up to `ttl` seconds out of date. That
is the right trade for "is another program up": the UI polls continuously, so a
state change is picked up on the next tick, and nothing here is a decision that
cannot be re-made a moment later. Anything that must be current (starting a
service, then confirming it came up) calls `refresh_now()` and pays the wait
deliberately.
"""

import threading
import time


class CachedProbe:
    """Background-refreshed boolean. `get()` is always instant."""

    def __init__(self, fn, ttl=5.0):
        self._fn = fn
        self._ttl = float(ttl)
        self._value = False
        self._checked = 0.0
        self._lock = threading.Lock()
        self._running = False

    def get(self):
        """The last known answer. Never performs I/O on the caller's thread."""
        now = time.time()
        with self._lock:
            stale = (now - self._checked) > self._ttl
            start = stale and not self._running
            if start:
                self._running = True
            value = self._value
        if start:
            threading.Thread(target=self._refresh, daemon=True).start()
        return value

    def _refresh(self):
        try:
            result = bool(self._fn())
        except Exception:
            result = False
        with self._lock:
            self._value = result
            self._checked = time.time()
            self._running = False

    def refresh_now(self):
        """Probe synchronously and return it. For callers that must be current."""
        self._refresh()
        with self._lock:
            return self._value

    def set(self, value):
        """Record an answer we just learned for free (e.g. we started it)."""
        with self._lock:
            self._value = bool(value)
            self._checked = time.time()
