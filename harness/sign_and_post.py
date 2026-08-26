"""Sign a fixture with the real webhook secret and POST it at the ingress.

BUILD.md's day-1 acceptance test. It exists because a fixture that only
works in a unit test proves nothing about the integration: this path
exercises the actual HMAC verification over the actual raw bytes, which
is where pitfall #2 (re-serialising before hashing) would surface.

    python -m harness.sign_and_post harness/fixtures/scenario_A.json
    python -m harness.sign_and_post --scenario A --repeat 2   # dedupe check
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

from core.config import settings
from harness import scenarios as sc
from services.ingress.signing import sign


async def post_one(
    client: httpx.AsyncClient, body: dict, event_id: str, secret: str, bad_sig: bool = False
) -> int:
    # Sign the exact bytes we are about to send. Serialising twice is the
    # bug this whole module is here to rule out.
    raw = json.dumps(body).encode()
    signature = "deadbeef" if bad_sig else sign(raw, secret)
    r = await client.post(
        "/webhook/razorpay",
        content=raw,
        headers={
            "X-Razorpay-Signature": signature,
            "X-Razorpay-Event-Id": event_id,
            "Content-Type": "application/json",
        },
    )
    return r.status_code


async def main_async(args) -> int:
    cfg = settings()
    secret = args.secret or cfg.rzp_webhook_secret

    if args.file:
        payload = json.loads(Path(args.file).read_text(encoding="utf-8"))
        deliveries = [
            (d["event_id"], d["body"]) for d in payload["timeline"]
        ]
        label = payload.get("scenario", args.file)
    else:
        s = sc.BY_KEY[args.scenario.upper()]
        deliveries = [(d.event_id, d.body) for d in s.deliveries]
        label = s.key

    failures = 0
    async with httpx.AsyncClient(base_url=args.url, timeout=10.0) as client:
        if args.bad_signature:
            code = await post_one(client, deliveries[0][1], "evt_bad", secret, bad_sig=True)
            ok = code == 400
            print(f"  bad signature            -> {code} {'OK' if ok else 'EXPECTED 400'}")
            return 0 if ok else 1

        print(f"scenario {label} -> {args.url}  (x{args.repeat})")
        for attempt in range(args.repeat):
            for event_id, body in deliveries:
                code = await post_one(client, body, event_id, secret)
                note = "" if attempt == 0 else "  (duplicate — expect one stored row)"
                print(f"  {event_id:24} -> {code}{note}")
                if code != 200:
                    failures += 1

        health = await client.get("/health")
        if health.status_code == 200:
            h = health.json()
            print(
                f"\n  received={h['stats']['received']} "
                f"duplicates={h['stats']['duplicates']} "
                f"rejected={h['stats']['rejected']} "
                f"max_skew={h['stats']['max_skew']}s"
            )
            if h.get("degraded"):
                print(f"  degraded: {h['degraded']}")

    return 1 if failures else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Sign fixtures and POST at the ingress.")
    ap.add_argument("file", nargs="?", help="a fixture JSON file")
    ap.add_argument("--scenario", default="A", help="A-F, when no file is given")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--secret", help="defaults to RZP_WEBHOOK_SECRET")
    ap.add_argument("--repeat", type=int, default=1, help="resend to exercise dedupe")
    ap.add_argument("--bad-signature", action="store_true", help="expect a 400")
    sys.exit(asyncio.run(main_async(ap.parse_args())))


if __name__ == "__main__":
    main()
