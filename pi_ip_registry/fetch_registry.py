#!/usr/bin/env python3
"""
Laptop helper: print current Pi IPs from the Google Sheet registry (GET).

Environment variables:
  REGISTRY_URL      Web app URL (same as Pis, ending in /exec)
  REGISTRY_SECRET   Shared secret

Example:
  REGISTRY_URL='https://script.google.com/.../exec' \\
  REGISTRY_SECRET='your-secret' \\
  python3 fetch_registry.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request


def main() -> int:
    url = os.environ.get("REGISTRY_URL", "").strip()
    secret = os.environ.get("REGISTRY_SECRET", "").strip()
    if not url or not secret:
        print("Set REGISTRY_URL and REGISTRY_SECRET", file=sys.stderr)
        return 1

    query = urllib.parse.urlencode({"secret": secret})
    req_url = f"{url}?{query}"
    with urllib.request.urlopen(req_url, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    if not data.get("ok"):
        print(f"Error: {data}", file=sys.stderr)
        return 1

    pis = data.get("pis") or []
    if not pis:
        print("No Pis registered yet.")
        return 0

    pis.sort(key=lambda r: str(r.get("pi_id", "")))
    print(f"{'ID':<4} {'IP':<16} {'IFACE':<8} {'LAST SEEN (UTC)':<26} {'HOSTNAME'}")
    print("-" * 72)
    for row in pis:
        print(
            f"{str(row.get('pi_id', '')):<4} "
            f"{str(row.get('ip', '')):<16} "
            f"{str(row.get('interface', '')):<8} "
            f"{str(row.get('last_seen_utc', '')):<26} "
            f"{row.get('hostname', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
