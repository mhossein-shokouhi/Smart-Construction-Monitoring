#!/usr/bin/env python3
"""
Heartbeat: report this Pi's LAN IP to a Google Apps Script web app (→ Google Sheet).

Runs on each Raspberry Pi with a unique PI_ID (0–3). Uses only the Python stdlib.

Environment variables (or /etc/pi-ip-announce.env):
  PI_ID                 Required. 0, 1, 2, or 3
  REGISTRY_URL          Required. Web app URL ending in /exec
  REGISTRY_SECRET       Required. Shared secret matching Apps Script
  NETWORK_INTERFACE     Optional. Default wlan0 (falls back wlan1, eth0)
  ANNOUNCE_INTERVAL_SEC Optional. Default 60
"""

from __future__ import annotations

import fcntl
import json
import os
import socket
import struct
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_INTERFACES = ("wlan0", "wlan1", "eth0")
DEFAULT_INTERVAL = 60


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name, default)
    return val.strip() if isinstance(val, str) and val.strip() else default


def get_interface_ip(iface: str) -> Optional[str]:
    """Return IPv4 address assigned to iface, or None."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        packed = struct.pack("256s", iface.encode("utf-8")[:15])
        addr = fcntl.ioctl(sock.fileno(), 0x8915, packed)[20:24]  # SIOCGIFADDR
        ip = socket.inet_ntoa(addr)
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    finally:
        sock.close()
    return None


def resolve_ip(preferred: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Try preferred interface, then common fallbacks."""
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    for iface in DEFAULT_INTERFACES:
        if iface not in candidates:
            candidates.append(iface)

    for iface in candidates:
        ip = get_interface_ip(iface)
        if ip:
            return ip, iface
    return None, None


def post_heartbeat(
    *,
    registry_url: str,
    secret: str,
    pi_id: str,
    ip: str,
    interface: str,
    hostname: str,
    timeout: float = 15.0,
) -> dict:
    payload = {
        "secret": secret,
        "pi_id": pi_id,
        "ip": ip,
        "interface": interface,
        "hostname": hostname,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        registry_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def load_env_file(path: str) -> None:
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def main() -> int:
    load_env_file("/etc/pi-ip-announce.env")

    pi_id = _env("PI_ID")
    registry_url = _env("REGISTRY_URL")
    secret = _env("REGISTRY_SECRET")
    preferred_iface = _env("NETWORK_INTERFACE", "wlan0")
    interval = int(_env("ANNOUNCE_INTERVAL_SEC", str(DEFAULT_INTERVAL)) or DEFAULT_INTERVAL)
    hostname = _env("HOSTNAME") or socket.gethostname()

    missing = [n for n, v in [
        ("PI_ID", pi_id),
        ("REGISTRY_URL", registry_url),
        ("REGISTRY_SECRET", secret),
    ] if not v]
    if missing:
        print(f"Missing required env: {', '.join(missing)}", file=sys.stderr)
        return 1

    print(f"Pi IP announce started (pi_id={pi_id}, interval={interval}s)")
    last_reported: Optional[str] = None

    while True:
        ip, iface = resolve_ip(preferred_iface)
        if not ip:
            print("No IP on network interfaces yet; retrying...", file=sys.stderr)
            time.sleep(min(interval, 15))
            continue

        if ip != last_reported:
            print(f"Detected {ip} on {iface}")

        try:
            result = post_heartbeat(
                registry_url=registry_url,
                secret=secret,
                pi_id=str(pi_id),
                ip=ip,
                interface=iface or "",
                hostname=hostname,
            )
            if not result.get("ok"):
                print(f"Registry rejected update: {result}", file=sys.stderr)
            elif ip != last_reported:
                print(f"Announced pi_id={pi_id} → {ip}")
            last_reported = ip
        except urllib.error.HTTPError as exc:
            print(f"HTTP error {exc.code}: {exc.read().decode()}", file=sys.stderr)
        except urllib.error.URLError as exc:
            print(f"Network error: {exc.reason}", file=sys.stderr)
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON from registry: {exc}", file=sys.stderr)
        except OSError as exc:
            print(f"Request failed: {exc}", file=sys.stderr)

        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
