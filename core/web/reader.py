"""Fetching a public web page and reducing it to readable Markdown.

Scrapling was tried and dropped: scrapling.fetchers cannot import without
Playwright -- 223 MB of browser automation for a path that never opens a
browser, against 1.1 MB here. curl_cffi keeps the half that matters, a real
Chrome TLS fingerprint.

_guard_public_url is a security boundary, not a convenience: the page it
admits is fed to a model, so it must never be a LAN service."""

import ipaddress
import re
import socket
import urllib.error
from urllib.parse import urlparse

from core.errors import _TurnError, openai_error

# Cached across calls: the import costs ~170 ms and a chat-only session never
# reads a URL, so it stays lazy. Module-level state belongs in the module that
# uses it -- leaving this behind in locally.py is exactly how the first run of
# this refactor broke every URL read with a NameError that no static check
# could see, because `global` makes an absent name look defined.
_http_fetcher_mod = None


def _http_fetcher():
    """Return curl_cffi's session class, or say how to install it.

    Optional and lazy, for the same reason flask-sock is: a missing dependency
    should cost the feature it backs, not the server.

    **Scrapling is the escalation tier, not the default one.** Its plain
    `Fetcher` is exactly this -- one curl_cffi request -- so using it for the
    common case buys nothing and costs a lot: `scrapling.fetchers` cannot be
    imported without Playwright, measured at **223 MB** of browser automation
    against **1.1 MB** for curl_cffi and markdownify. Paying that on every
    install, for every user, to fetch pages that answer a plain request, is the
    trade this module refused and still refuses.

    What Scrapling genuinely adds is anti-bot bypass, and that is a real
    capability -- it is simply not needed until a site actually refuses us.
    So it is wired in at the point where the cheap path FAILS (see
    `_stealth_fetch`): curl_cffi first, and a site that blocks it escalates to
    a real browser. Installed, it works; not installed, the error names the
    command. Nobody pays 223 MB for a feature they do not use, and nobody is
    told "no" when they need it.

    curl_cffi is what carries the useful half — a real Chrome TLS/JA3
    fingerprint, so pages that refuse python-requests still answer.
    """
    global _http_fetcher_mod
    if _http_fetcher_mod is None:
        try:
            from curl_cffi import requests as creq
        except ImportError:
            raise _TurnError(openai_error(
                "Reading a URL needs curl_cffi and markdownify, which are not "
                "installed. Run `pip install curl_cffi markdownify`.",
                "server_error", 503))
        _http_fetcher_mod = creq
    return _http_fetcher_mod



_stealth_mod = None
_STEALTH_UNAVAILABLE = object()

# Statuses that mean "a bot check refused us", as opposed to "this page is
# genuinely not there". Escalating on 404 would launch a browser to confirm a
# typo; escalating on these is the entire point of having a browser.
BLOCKED_STATUSES = (401, 403, 405, 406, 409, 418, 429, 503)


def stealth_available():
    """True when Scrapling can be imported. Never raises, never blocks."""
    return _stealth_fetcher() is not None


def _stealth_fetcher():
    """Scrapling's StealthyFetcher, or None when it is not installed.

    Imported lazily and cached -- including the FAILURE, because the import
    walks Playwright and must not be retried on every blocked page.
    """
    global _stealth_mod
    if _stealth_mod is _STEALTH_UNAVAILABLE:
        return None
    if _stealth_mod is None:
        try:
            from scrapling.fetchers import StealthyFetcher
        except Exception:
            # Exception, not ImportError: a half-installed Scrapling raises
            # from inside Playwright, and a browser-automation stack failing to
            # load must not take the ordinary reader down with it.
            _stealth_mod = _STEALTH_UNAVAILABLE
            return None
        _stealth_mod = StealthyFetcher
    return _stealth_mod


def _stealth_fetch(url, timeout_s):
    """Fetch through a real browser. Returns HTML, or None if unavailable.

    NOT VERIFIED END TO END -- Scrapling is not installed on this machine, so
    this path has never run against a live site. Scrapling's fetch API has also
    moved between releases, which is why the result is read through several
    attribute names rather than one. Treat a first success here as the test.
    """
    fetcher = _stealth_fetcher()
    if fetcher is None:
        return None
    page = fetcher.fetch(url, headless=True, network_idle=True,
                         timeout=int(timeout_s * 1000))
    for attr in ("html_content", "content", "body", "text"):
        html = getattr(page, attr, None)
        if isinstance(html, bytes):
            html = html.decode("utf-8", "replace")
        if isinstance(html, str) and html.strip():
            return html
    return None


# Stripped before conversion. Boilerplate is the whole reason this endpoint
# exists: a search result rendered as raw Markdown is nav, cookie banner,
# footer and share buttons, and on the NPU's 8192-token window that clutter
# is the difference between a page fitting and not.
_BOILERPLATE_TAGS = ("script", "style", "noscript", "nav", "header", "footer",
                     "aside", "form", "svg", "iframe", "template", "button")


_BOILERPLATE_ROLES = ("navigation", "banner", "contentinfo", "search",
                      "complementary", "dialog")


def _html_to_markdown(html):
    """Readable Markdown from a web page, with the furniture removed.

    Prefers the page's own <main>/<article> when it has one, because a site
    that marks its content has already answered the question better than any
    heuristic could. Otherwise it strips the known furniture from <body> and
    converts what is left. This is not a readability-grade article extractor
    and does not claim to be — it removes tag soup and chrome, which is where
    nearly all of the token cost is.
    """
    try:
        from bs4 import BeautifulSoup
        import markdownify
    except ImportError:
        raise _TurnError(openai_error(
            "Markdown conversion needs markdownify. Run `pip install markdownify`.",
            "server_error", 503))

    soup = BeautifulSoup(html, "html.parser")
    for el in soup(list(_BOILERPLATE_TAGS)):
        el.decompose()
    for el in soup.find_all(attrs={"role": True}):
        if str(el.get("role", "")).lower() in _BOILERPLATE_ROLES:
            el.decompose()
    for el in soup.find_all(attrs={"aria-hidden": "true"}):
        el.decompose()

    root = soup.find("main") or soup.find("article") or soup.body or soup
    md = markdownify.markdownify(str(root), heading_style="ATX")
    # Collapse the runs of blank lines that dropping elements leaves behind;
    # they are pure token cost and read as a broken document.
    return re.sub(r"\n{3,}", "\n\n", md or "").strip()


def _guard_public_url(url):
    """Reject anything that is not a public http(s) page, and name the reason.

    A fetched page is untrusted text heading straight into a model's prompt, so
    this endpoint must not double as a way to read the machine it runs on — the
    server binds 0.0.0.0 and the Util tab is reachable from a phone. Two checks,
    both load-bearing:

    * **Scheme.** http/https only. `file:///C:/Users/...` is the one-line
      version of this attack and `smb://` is the networked one.
    * **The resolved address, not the string.** Blocking the literal
      "127.0.0.1" is no guard at all: a DNS name pointing at it — or at
      192.168.x.x, or at 169.254.169.254 — walks past a string check. Every
      address `getaddrinfo` returns is checked, since a host with one public
      and one private record must not be half-allowed.

    What this does not claim: the name is resolved here and resolved again by
    curl inside the fetcher, so a hostile resolver could answer differently the
    second time. Pinning the connection to an already-checked IP is what closes
    that, and the fetcher does not expose it. The property that *is* held is the
    one that matters for a prompt: content from a private address is never
    returned, because the same check runs over the redirect chain after the
    fetch as well as over the URL before it.
    """
    try:
        parts = urllib.parse.urlsplit(url)
    except Exception:
        parts = None
    if not parts or parts.scheme not in ("http", "https") or not parts.hostname:
        raise _TurnError(openai_error(
            "Only http:// and https:// URLs can be read."))

    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(parts.hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise _TurnError(openai_error(f"Could not resolve '{parts.hostname}': {e}"))
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        # ::ffff:127.0.0.1 is loopback wearing an IPv6 hat; unwrap before asking.
        ip = getattr(ip, "ipv4_mapped", None) or ip
        private = (ip.is_loopback or ip.is_private or ip.is_link_local
                   or ip.is_reserved or ip.is_multicast or ip.is_unspecified
                   or not ip.is_global)
        if private:
            raise _TurnError(openai_error(
                f"Refusing to read '{parts.hostname}': it resolves to {ip}, "
                f"which is this machine or this LAN. The reader fetches public "
                f"web pages only."))
    return parts
