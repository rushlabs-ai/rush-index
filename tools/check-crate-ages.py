#!/usr/bin/env python3
"""Supply-chain check: every crate in Cargo.lock must be at least N days old (default 7).

    python3 tools/check-crate-ages.py [--days 7]

Reads Cargo.lock, asks the crates.io API when each locked version was published, and
fails if any is newer than the cutoff or has been yanked. Standard library only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import tomllib
import urllib.request
from pathlib import Path

UA = {"User-Agent": "rush-index supply-chain check (admin@rushlabs.ai)"}


def published(name: str, version: str) -> tuple[str, bool]:
    req = urllib.request.Request(f"https://crates.io/api/v1/crates/{name}/{version}", headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        v = json.load(r)["version"]
    return v["created_at"], v["yanked"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--lock", default=str(Path(__file__).resolve().parents[1] / "Cargo.lock"))
    args = ap.parse_args()
    lock = tomllib.loads(Path(args.lock).read_text())
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=args.days)
    bad = []
    pkgs = [p for p in lock["package"] if p.get("source", "").startswith("registry+")]
    for i, p in enumerate(pkgs, 1):
        created, yanked = published(p["name"], p["version"])
        when = dt.datetime.fromisoformat(created.replace("Z", "+00:00"))
        if yanked or when > cutoff:
            bad.append(f"{p['name']} {p['version']}: {'YANKED' if yanked else 'published ' + created[:10]}")
        time.sleep(0.25)  # crates.io asks for at most one request per second per crawler; stay gentle
    print(f"{len(pkgs)} crates checked, cutoff {cutoff.date()}")
    for b in bad:
        print("  TOO NEW OR YANKED:", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
