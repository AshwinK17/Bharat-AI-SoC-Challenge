"""Process-wide offline policy: loopback only, no DNS.

Xaverix claims to work with no internet. This module makes that claim
enforceable rather than decorative, by gating the *outbound* socket
operations so the process physically cannot reach a remote host.

Design notes
------------
* Only outbound calls are gated: ``connect``, ``connect_ex``, ``sendto``,
  ``create_connection`` and ``getaddrinfo``. ``bind``/``listen``/``accept``/
  ``recv``/``send`` are untouched, so ``http.server`` keeps working.
* ``getaddrinfo`` is gated too, which is the stronger claim: the process
  cannot even *resolve* a public hostname, let alone connect to one.
* ``_is_local`` deliberately performs **no** name resolution. A host is local
  only if it is a loopback literal or a known localhost name. Anything else is
  refused, so a hostname can never be leaked to a resolver.
* Loopback stays open because that is where ExecuServe hosts Llama.

Scope: this guard constrains the Xaverix Python process only. It makes no
claim about other applications on the device. Aeroplane mode remains the
end-to-end verification.
"""
from __future__ import annotations

import ipaddress
import socket
import time

LOOPBACK_NAMES = {
    "localhost",
    "localhost.localdomain",
    "ip6-localhost",
    "ip6-loopback",
}
# Wildcard binds; harmless as connect targets and never remote.
WILDCARD_HOSTS = {"", "0.0.0.0", "::", "*"}

POLICY_ALLOW = ["127.0.0.0/8", "::1", "localhost"]

BLOCKED_ATTEMPTS = []
_MAX_RECORDED = 200

_installed = False
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_sendto = socket.socket.sendto
_real_getaddrinfo = socket.getaddrinfo
_real_create_connection = socket.create_connection


class OfflineGuardBlocked(OSError):
    """Raised when the process attempts to leave the loopback interface."""


def _allow_lan():
    # Read lazily so tests can flip the config without reinstalling the guard.
    from app import config

    return config.ALLOW_LAN


def _is_local(host):
    if host is None:
        return True
    text = str(host).strip().strip("[]").lower()
    if text in WILDCARD_HOSTS or text in LOOPBACK_NAMES:
        return True
    # Strip a zone id such as fe80::1%eth0 before parsing.
    if "%" in text:
        text = text.split("%", 1)[0]
    try:
        addr = ipaddress.ip_address(text)
    except ValueError:
        # Not an IP literal, and not a localhost name: refuse without resolving.
        return False
    if addr.is_loopback or addr.is_unspecified:
        return True
    if _allow_lan() and (addr.is_private or addr.is_link_local):
        return True
    return False


def _host_port(address):
    if isinstance(address, (tuple, list)) and address:
        port = address[1] if len(address) > 1 else None
        return address[0], port
    return address, None


def _record(host, port, api):
    BLOCKED_ATTEMPTS.append(
        {"ts": time.time(), "host": str(host), "port": port, "api": api}
    )
    del BLOCKED_ATTEMPTS[:-_MAX_RECORDED]


def _check(address, api):
    host, port = _host_port(address)
    if _is_local(host):
        return
    _record(host, port, api)
    raise OfflineGuardBlocked(
        "OfflineGuardBlocked: outbound access to {}:{} via {} was denied. "
        "Xaverix is offline-only; permitted targets are {}.".format(
            host, port, api, ", ".join(POLICY_ALLOW)
        )
    )


def _guarded_connect(self, address):
    _check(address, "socket.connect")
    return _real_connect(self, address)


def _guarded_connect_ex(self, address):
    _check(address, "socket.connect_ex")
    return _real_connect_ex(self, address)


def _guarded_sendto(self, *args, **kwargs):
    if args:
        _check(args[-1], "socket.sendto")
    return _real_sendto(self, *args, **kwargs)


def _guarded_getaddrinfo(host, port, *args, **kwargs):
    if not _is_local(host):
        _check((host, port), "socket.getaddrinfo")
    return _real_getaddrinfo(host, port, *args, **kwargs)


def _guarded_create_connection(address, *args, **kwargs):
    _check(address, "socket.create_connection")
    return _real_create_connection(address, *args, **kwargs)


def install():
    """Apply the policy. Idempotent; call as early as possible."""
    global _installed
    if _installed:
        return
    socket.socket.connect = _guarded_connect
    socket.socket.connect_ex = _guarded_connect_ex
    socket.socket.sendto = _guarded_sendto
    socket.getaddrinfo = _guarded_getaddrinfo
    socket.create_connection = _guarded_create_connection
    _installed = True


def uninstall():
    """Restore the real socket API. For debugging and tests only."""
    global _installed
    socket.socket.connect = _real_connect
    socket.socket.connect_ex = _real_connect_ex
    socket.socket.sendto = _real_sendto
    socket.getaddrinfo = _real_getaddrinfo
    socket.create_connection = _real_create_connection
    _installed = False


def is_installed():
    return _installed


# --------------------------------------------------------------- live proof
_PROBES = (
    ("api.openai.com", 443, "socket.create_connection"),
    ("8.8.8.8", 53, "socket.create_connection"),
    ("huggingface.co", None, "socket.getaddrinfo"),
)


def _probe_one(host, port, api):
    target = host if port is None else "{}:{}".format(host, port)
    try:
        if api == "socket.getaddrinfo":
            socket.getaddrinfo(host, port or 80)
        else:
            sock = socket.create_connection((host, port), timeout=1.0)
            sock.close()
    except OfflineGuardBlocked as exc:
        return {"target": target, "api": api, "result": "blocked", "error": str(exc)}
    except OSError as exc:
        # Reached the network stack but failed for another reason. Report it
        # plainly instead of claiming a block we did not perform.
        return {
            "target": target,
            "api": api,
            "result": "not_blocked_by_guard",
            "error": "{}: {}".format(type(exc).__name__, exc),
        }
    return {"target": target, "api": api, "result": "allowed"}


def proof(local_port=None):
    """Demonstrate the policy with live probes, including a loopback control."""
    probes = [_probe_one(*p) for p in _PROBES]

    if local_port:
        loopback = _probe_one("127.0.0.1", local_port, "socket.create_connection")
        loopback["note"] = "loopback control: ExecuServe / local services stay reachable"
        probes.append(loopback)

    remote = [p for p in probes if not p["target"].startswith("127.0.0.1")]
    return {
        "guard_active": _installed,
        "policy": {"allow": POLICY_ALLOW, "allow_lan": _allow_lan()},
        "probes": probes,
        "blocked": bool(remote) and all(p["result"] == "blocked" for p in remote),
        "blocked_attempts_since_start": len(BLOCKED_ATTEMPTS),
        "scope_note": (
            "This guard constrains the Xaverix Python process only. ExecuServe "
            "is a separate app, configured so that only this phone can connect. "
            "The physical proof of offline operation is aeroplane mode."
        ),
        "physical_proof": "Run the full demo with aeroplane mode enabled.",
    }
