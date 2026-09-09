"""Who may call this server: source address first, then an optional key.

`--host` defaults to 0.0.0.0 and that is not an oversight -- a container
reaches the host over a virtual adapter, and a 127.0.0.1 socket is invisible
from inside Podman or WSL no matter how the firewall is set
(docs/ODYSSEUS.md). The bind has to stay wide, so the filtering has to happen
here.

**It cannot happen in the firewall**, which is where this project used to put
it. A scoped `New-NetFirewallRule -RemoteAddress 172.17.96.0/20` is correct on
the machine it was typed on and replicates nowhere: WSL picks that subnet per
machine and changes it, the rule needs elevation, Windows-only leaves a Linux
install with nothing, and -- measured on the development box, 2026-09-09 --
two `python.exe` Public/Any/Any rules that Windows had created from an
ordinary permission prompt silently outranked it, on a Wi-Fi that Windows
classed Public. The careful rule was already not in force and nothing said so.
A user on another Intel box clicks that same prompt. Enforcement that does not
ship with the code does not ship.

Two independent layers, because they answer different threats. The source
filter (`--allow-from`, default `auto`) is the posture that has to be right
with nobody configuring anything: loopback plus the private subnets of virtual
adapters, which is exactly the container path and nothing else. The key
(`--api-key`, off) is for when the port is deliberately exposed past the box
-- a real LAN, Tailscale, a second machine -- where no source is trustworthy
on its address alone.
"""

import hmac
import ipaddress
import socket

from flask import request

from core import config
from core.errors import openai_error

# Interface names that mean "a virtual switch to a VM or a container", not
# "the network the user is actually on". Windows puts every Hyper-V, WSL and
# Docker Desktop switch behind `vEthernet (...)`; the rest are Linux.
_VIRTUAL_HINTS = ("vethernet", "wsl", "hyper-v", "hyperv",
                  "docker", "podman", "containers", "cni",
                  "virbr", "veth", "vboxnet", "br-")

# A preflight never carries credentials -- asking whether `Authorization` may
# be sent is what it is FOR -- so requiring one here would fail every browser
# client before the real request happened. The source filter still applies.
_KEY_EXEMPT_PATHS = frozenset({"/"})

_cached_networks = None
_cache_token = None


def _looks_virtual(name):
    low = name.lower()
    return any(hint in low for hint in _VIRTUAL_HINTS)


def _network_of(address, netmask):
    try:
        return ipaddress.ip_network(f"{address.split('%')[0]}/{netmask}",
                                    strict=False)
    except (ValueError, TypeError):
        return None


def virtual_networks():
    """The private subnets of virtual adapters -- the container path, found live.

    Live rather than hardcoded because the WSL subnet is not a constant: it is
    assigned per machine and changes, which is the single reason the firewall
    rule in the docs could never be copied to a second box.
    """
    try:
        import psutil
    except ImportError:
        # Never fail closed on a missing optional dependency: that would take
        # the container path down on a machine where it had been working.
        return []
    found = []
    families = {socket.AF_INET, getattr(socket, "AF_INET6", socket.AF_INET)}
    for name, addrs in psutil.net_if_addrs().items():
        if not _looks_virtual(name):
            continue
        for addr in addrs:
            if addr.family not in families:
                continue
            net = _network_of(addr.address, addr.netmask)
            # Private and non-link-local as a second condition, so a
            # misread interface name can still never open the real LAN.
            if net and net.is_private and not net.is_link_local:
                found.append(net)
    return found


def allowed_networks():
    """Resolve --allow-from. None means 'no source filter' (--allow-from any)."""
    global _cached_networks, _cache_token
    spec = (config.ALLOW_FROM or "auto").strip()
    if _cache_token == spec:
        return _cached_networks
    nets = []
    for token in [t.strip() for t in spec.split(",") if t.strip()]:
        if token.lower() == "any":
            _cached_networks, _cache_token = None, spec
            return None
        if token.lower() == "auto":
            nets.append(ipaddress.ip_network("127.0.0.0/8"))
            nets.append(ipaddress.ip_network("::1/128"))
            nets.extend(virtual_networks())
            continue
        try:
            nets.append(ipaddress.ip_network(token, strict=False))
        except ValueError:
            print(f"WARNING: --allow-from {token!r} is not an address or CIDR; "
                  f"ignoring it", flush=True)
    _cached_networks, _cache_token = nets, spec
    return nets


def reset_cache():
    global _cached_networks, _cache_token
    _cached_networks, _cache_token = None, None


def source_allowed(remote_addr, networks):
    if networks is None:          # --allow-from any
        return True
    if not remote_addr:
        # No peer address is not a pass. Werkzeug always supplies one; a
        # future front-end that does not should be made to say so explicitly.
        return False
    try:
        ip = ipaddress.ip_address(remote_addr.split("%")[0])
    except ValueError:
        return False
    # A v4 address arriving over a dual-stack socket looks like ::ffff:10.0.0.1
    # and would match no v4 rule, so unwrap it before testing.
    if getattr(ip, "ipv4_mapped", None):
        ip = ip.ipv4_mapped
    return any(ip in net for net in networks)


def presented_key(headers):
    auth = headers.get("Authorization", "") or ""
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    return (headers.get("X-Api-Key") or "").strip()


def key_ok(headers):
    expected = config.API_KEY
    if not expected:
        return True
    return hmac.compare_digest(presented_key(headers), expected)


def attach_guard(app):
    """Refuse a caller before any route sees it.

    Registered ahead of every blueprint, so it runs first: `before_request`
    handlers fire in registration order, and a rejected caller must not reach
    the request log, a model lock, or an upload buffer.
    """
    @app.before_request
    def _guard():
        if not source_allowed(request.remote_addr, allowed_networks()):
            return openai_error(
                f"Refused: {request.remote_addr} is outside --allow-from "
                f"({config.ALLOW_FROM}). locally binds 0.0.0.0 so containers "
                f"can reach it, and answers only loopback and virtual/container "
                f"subnets by default. Add a subnet with --allow-from, or set "
                f"--allow-from any (with --api-key) to serve the network.",
                "permission_error", 403)
        if request.method == "OPTIONS" or request.path in _KEY_EXEMPT_PATHS:
            return None
        if not key_ok(request.headers):
            return openai_error(
                "Missing or invalid API key. Send it as 'Authorization: "
                "Bearer <key>' or 'X-Api-Key: <key>'.",
                "authentication_error", 401)
        return None

    return app


def describe(host):
    """One startup line, and a warning when the port is genuinely open."""
    nets = allowed_networks()
    key = " + API key" if config.API_KEY else ""
    if nets is None:
        if not config.API_KEY and host not in ("127.0.0.1", "localhost", "::1"):
            return (f"WARNING: --allow-from any on {host} serves anyone who can "
                    f"reach this port, unauthenticated. Add --api-key.")
        return f"Access: any source{key}"
    return f"Access: {', '.join(str(n) for n in nets) or 'nothing'}{key}"
