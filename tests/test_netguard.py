"""The 0.0.0.0 bind is required (a container cannot see a loopback socket) and
is therefore not, by itself, a decision to serve the Wi-Fi. These pin both
halves: the container path keeps working, and the LAN address that was
answering 200 on the development box gets a 403.
"""

import ipaddress
import unittest
from unittest import mock

from flask import Flask, jsonify

from core import config, netguard
from core.netguard import (allowed_networks, attach_guard, key_ok,
                           presented_key, source_allowed, virtual_networks)

# The real figures from the machine this was written on: WSL's vSwitch, and
# the school Wi-Fi that Windows classed Public.
WSL_NET = ipaddress.ip_network("172.17.96.0/20")
WIFI = "10.153.88.41"


def _app():
    app = Flask("guard-test")
    attach_guard(app)

    @app.route("/v1/models")
    def models():
        return jsonify({"data": []})

    @app.route("/")
    def root():
        return "Ollama is running"

    return app


class SourceFilterTests(unittest.TestCase):
    def setUp(self):
        self.nets = [ipaddress.ip_network("127.0.0.0/8"),
                     ipaddress.ip_network("::1/128"), WSL_NET]

    def test_loopback_is_allowed(self):
        for ip in ("127.0.0.1", "::1"):
            self.assertTrue(source_allowed(ip, self.nets))

    def test_container_gateway_and_peer_are_allowed(self):
        # 172.17.96.1 is the host as a container sees it; the container itself
        # dials out from another address in the same /20.
        for ip in ("172.17.96.1", "172.17.100.25"):
            self.assertTrue(source_allowed(ip, self.nets))

    def test_the_wifi_is_refused(self):
        self.assertFalse(source_allowed(WIFI, self.nets))

    def test_other_rfc1918_is_refused(self):
        # A private address is not the same thing as a container address --
        # the Wi-Fi above is RFC1918 too, which is why "allow private" fails.
        for ip in ("192.168.1.10", "10.0.0.5", "172.16.0.1"):
            self.assertFalse(source_allowed(ip, self.nets))

    def test_ipv4_mapped_v6_is_unwrapped(self):
        # A v4 peer on a dual-stack socket arrives as ::ffff:a.b.c.d and would
        # match no v4 rule unless unwrapped -- which fails OPEN for the Wi-Fi
        # only if we forget, so both directions are pinned.
        self.assertTrue(source_allowed("::ffff:127.0.0.1", self.nets))
        self.assertFalse(source_allowed(f"::ffff:{WIFI}", self.nets))

    def test_missing_or_junk_address_is_refused(self):
        for ip in (None, "", "not-an-ip"):
            self.assertFalse(source_allowed(ip, self.nets))

    def test_any_disables_the_filter(self):
        self.assertTrue(source_allowed(WIFI, None))


class ResolutionTests(unittest.TestCase):
    def setUp(self):
        self.spec = config.ALLOW_FROM
        netguard.reset_cache()

    def tearDown(self):
        config.ALLOW_FROM = self.spec
        netguard.reset_cache()

    def test_auto_includes_loopback_and_discovered_subnets(self):
        config.ALLOW_FROM = "auto"
        with mock.patch.object(netguard, "virtual_networks", return_value=[WSL_NET]):
            nets = allowed_networks()
        self.assertIn(ipaddress.ip_network("127.0.0.0/8"), nets)
        self.assertIn(WSL_NET, nets)

    def test_any_resolves_to_no_filter(self):
        config.ALLOW_FROM = "any"
        self.assertIsNone(allowed_networks())

    def test_explicit_cidrs_are_added(self):
        config.ALLOW_FROM = "auto,192.168.50.0/24"
        with mock.patch.object(netguard, "virtual_networks", return_value=[]):
            nets = allowed_networks()
        self.assertIn(ipaddress.ip_network("192.168.50.0/24"), nets)

    def test_junk_cidr_is_ignored_not_fatal(self):
        config.ALLOW_FROM = "not-a-subnet"
        self.assertEqual(allowed_networks(), [])

    def test_discovery_skips_physical_adapters(self):
        # The name heuristic is the first gate; `is_private` is the second.
        # A Wi-Fi adapter passes neither, and this is the test that would fail
        # if a hint were ever loosened to something like "net".
        fake = {"Wi-Fi": [mock.Mock(family=2, address=WIFI, netmask="255.255.255.0")],
                "vEthernet (WSL (Hyper-V firewall))":
                    [mock.Mock(family=2, address="172.17.96.1", netmask="255.255.240.0")]}
        with mock.patch.dict("sys.modules", {"psutil": mock.Mock(net_if_addrs=lambda: fake)}):
            found = virtual_networks()
        self.assertEqual(found, [WSL_NET])

    def test_missing_psutil_does_not_close_the_container_path(self):
        # Failing closed here would break a working install on an upgrade.
        with mock.patch.dict("sys.modules", {"psutil": None}):
            self.assertEqual(virtual_networks(), [])


class KeyTests(unittest.TestCase):
    def setUp(self):
        self.key = config.API_KEY

    def tearDown(self):
        config.API_KEY = self.key

    def test_no_key_configured_accepts_everyone(self):
        config.API_KEY = None
        self.assertTrue(key_ok({}))

    def test_bearer_and_x_api_key_both_work(self):
        config.API_KEY = "s3cret"
        self.assertTrue(key_ok({"Authorization": "Bearer s3cret"}))
        self.assertTrue(key_ok({"X-Api-Key": "s3cret"}))

    def test_wrong_or_absent_key_is_refused(self):
        config.API_KEY = "s3cret"
        for headers in ({}, {"Authorization": "Bearer nope"}, {"X-Api-Key": ""}):
            self.assertFalse(key_ok(headers))

    def test_bearer_scheme_is_case_insensitive(self):
        config.API_KEY = "s3cret"
        self.assertEqual(presented_key({"Authorization": "bearer s3cret"}), "s3cret")


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.spec, self.key = config.ALLOW_FROM, config.API_KEY
        config.ALLOW_FROM = "127.0.0.0/8,172.17.96.0/20"
        config.API_KEY = None
        netguard.reset_cache()
        self.client = _app().test_client()

    def tearDown(self):
        config.ALLOW_FROM, config.API_KEY = self.spec, self.key
        netguard.reset_cache()

    def test_container_reaches_the_api(self):
        r = self.client.get("/v1/models", environ_overrides={"REMOTE_ADDR": "172.17.100.7"})
        self.assertEqual(r.status_code, 200)

    def test_wifi_gets_403_naming_the_flag(self):
        r = self.client.get("/v1/models", environ_overrides={"REMOTE_ADDR": WIFI})
        self.assertEqual(r.status_code, 403)
        self.assertIn("--allow-from", r.get_json()["error"]["message"])

    def test_key_is_required_when_set(self):
        config.API_KEY = "s3cret"
        r = self.client.get("/v1/models", environ_overrides={"REMOTE_ADDR": "127.0.0.1"})
        self.assertEqual(r.status_code, 401)
        r = self.client.get("/v1/models", headers={"Authorization": "Bearer s3cret"},
                            environ_overrides={"REMOTE_ADDR": "127.0.0.1"})
        self.assertEqual(r.status_code, 200)

    def test_source_filter_outranks_a_correct_key(self):
        # Holding the key is not permission to arrive from the Wi-Fi while
        # --allow-from still excludes it.
        config.API_KEY = "s3cret"
        r = self.client.get("/v1/models", headers={"Authorization": "Bearer s3cret"},
                            environ_overrides={"REMOTE_ADDR": WIFI})
        self.assertEqual(r.status_code, 403)

    def test_preflight_needs_no_key(self):
        # A preflight cannot carry credentials -- asking whether Authorization
        # may be sent is what it is for -- so 401ing it breaks every browser
        # client before the real request is ever made.
        config.API_KEY = "s3cret"
        r = self.client.options("/v1/models", environ_overrides={"REMOTE_ADDR": "127.0.0.1"})
        self.assertNotEqual(r.status_code, 401)

    def test_liveness_root_needs_no_key(self):
        config.API_KEY = "s3cret"
        r = self.client.get("/", environ_overrides={"REMOTE_ADDR": "127.0.0.1"})
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
