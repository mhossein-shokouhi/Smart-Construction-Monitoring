#!/usr/bin/env python3
"""
Ping a host multiple times and report mean latency and jitter (std dev of RTT).
"""

import re
import statistics
import subprocess
import sys


def run_pings(host: str, count: int = 100) -> list[float]:
    """Run ping to host and return list of RTT values in milliseconds."""
    # -c count, -n no DNS reverse lookup (faster), works on macOS/Linux
    cmd = ["ping", "-c", str(count), "-n", host]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=count + 10,
        )
    except subprocess.TimeoutExpired:
        raise SystemExit("Ping timed out.")
    except FileNotFoundError:
        raise SystemExit("ping command not found.")

    if result.returncode != 0 and not result.stdout:
        print(result.stderr or "Ping failed.", file=sys.stderr)
        raise SystemExit(1)

    # Parse "time=12.345 ms" or "time=12.345ms" from each line
    rtt_pattern = re.compile(r"time[=:]?\s*([\d.]+)\s*ms", re.IGNORECASE)
    rtts = []
    for line in result.stdout.splitlines():
        match = rtt_pattern.search(line)
        if match:
            rtts.append(float(match.group(1)))

    return rtts


def main() -> None:
    host = "8.8.8.8"  # default
    count = 100

    if len(sys.argv) >= 2:
        host = sys.argv[1]
    if len(sys.argv) >= 3:
        try:
            count = int(sys.argv[2])
        except ValueError:
            print("Usage: ping_metrics.py [host] [count]", file=sys.stderr)
            raise SystemExit(2)

    if count < 2:
        count = 2  # need at least 2 for std dev

    print(f"Pinging {host} ({count} packets)...")
    rtts = run_pings(host, count)

    if not rtts:
        print("No valid RTT replies received.", file=sys.stderr)
        raise SystemExit(1)

    lost = count - len(rtts)
    loss_pct = 100.0 * lost / count if count else 0

    mean_rtt = statistics.mean(rtts)
    jitter = statistics.stdev(rtts) if len(rtts) >= 2 else 0.0

    print(f"\n--- {host} ping metrics ---")
    print(f"  Packets: {len(rtts)} received, {lost} lost ({loss_pct:.1f}% loss)")
    print(f"  Mean latency: {mean_rtt:.2f} ms")
    print(f"  Jitter (std dev): {jitter:.2f} ms")
    if rtts:
        print(f"  Min / Max: {min(rtts):.2f} / {max(rtts):.2f} ms")


if __name__ == "__main__":
    main()
