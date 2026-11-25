"""The offline guard must block the network and still permit loopback."""
import os
import socket
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import offline_guard  # noqa: E402


class OfflineGuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        offline_guard.install()

    def test_installs_idempotently(self):
        offline_guard.install()
        offline_guard.install()
        self.assertTrue(offline_guard.is_installed())

    def test_loopback_literal_is_local(self):
        for host in ("127.0.0.1", "127.1.2.3", "::1", "localhost", "[::1]"):
            self.assertTrue(offline_guard._is_local(host), host)

    def test_public_addresses_are_not_local(self):
        for host in ("8.8.8.8", "1.1.1.1", "api.openai.com", "huggingface.co",
                     "203.0.113.9", "example.com"):
            self.assertFalse(offline_guard._is_local(host), host)

    def test_connect_to_public_ip_is_blocked(self):
        sock = socket.socket()
        sock.settimeout(2)
        try:
            with self.assertRaises(offline_guard.OfflineGuardBlocked):
                sock.connect(("1.1.1.1", 53))
        finally:
            sock.close()

    def test_dns_resolution_of_public_host_is_blocked(self):
        # Blocking DNS is the stronger claim: the process cannot even resolve
        # a public hostname, let alone reach one.
        with self.assertRaises(offline_guard.OfflineGuardBlocked):
            socket.getaddrinfo("huggingface.co", 443)

    def test_create_connection_to_public_host_is_blocked(self):
        with self.assertRaises(offline_guard.OfflineGuardBlocked):
            socket.create_connection(("api.openai.com", 443), timeout=2)

    def test_loopback_server_still_works(self):
        # ExecuServe lives on loopback, so this must never be blocked.
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            client = socket.create_connection(("127.0.0.1", port), timeout=2)
            client.close()
        finally:
            server.close()

    def test_urllib_cannot_reach_a_public_host(self):
        import urllib.error
        import urllib.request

        with self.assertRaises((urllib.error.URLError, offline_guard.OfflineGuardBlocked)):
            urllib.request.urlopen("http://example.com", timeout=3)

    def test_proof_reports_blocked_with_loopback_allowed(self):
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            proof = offline_guard.proof(local_port=port)
        finally:
            server.close()

        self.assertTrue(proof["guard_active"])
        self.assertTrue(proof["blocked"])

        remote = [p for p in proof["probes"] if not p["target"].startswith("127.0.0.1")]
        self.assertTrue(remote)
        for probe in remote:
            self.assertEqual(probe["result"], "blocked", probe)

        loopback = [p for p in proof["probes"] if p["target"].startswith("127.0.0.1")]
        self.assertEqual(loopback[0]["result"], "allowed")

    def test_blocked_attempts_are_recorded(self):
        before = len(offline_guard.BLOCKED_ATTEMPTS)
        try:
            socket.create_connection(("8.8.4.4", 53), timeout=1)
        except offline_guard.OfflineGuardBlocked:
            pass
        self.assertGreater(len(offline_guard.BLOCKED_ATTEMPTS), before)


if __name__ == "__main__":
    unittest.main()
