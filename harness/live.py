"""The live test-mode loop. Real order, real payment, real signed webhook.

Everything else in `harness/` replays fixtures. This module does not: it
creates an actual order through the Razorpay API, hands you a checkout
page to pay on, and then waits for Razorpay to deliver a genuinely
HMAC-signed webhook to your ingress over the public internet.

That distinction matters for the submission. A replay proves the fold is
correct; only this proves the integration is real - that the signature
verification works against bytes we did not construct, that the event
shapes match production, and that the whole chain from Razorpay to a
verdict actually runs.

    python -m harness.live doctor          # what is configured, what is missing
    python -m harness.live order           # create a real order, print a pay link
    python -m harness.live watch           # follow observations and verdicts
    python -m harness.live capture <pay_id>  # move real (test-mode) money

Test mode only. `require_test_mode` refuses live keys (E15).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import webbrowser
from typing import Any

import httpx

from core.config import settings
from core.fold import fold
from core.infra import Infra, use_psycopg_compatible_loop
from core.verdicts import Action

CHECKOUT_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Nishchay - test payment</title>
<script src="https://checkout.razorpay.com/v1/checkout.js"></script>
<style>
 body{{font-family:ui-monospace,Menlo,monospace;background:#09090b;color:#fafafa;
      display:flex;min-height:100vh;align-items:center;justify-content:center;margin:0}}
 .card{{border:1px solid #27272a;border-radius:12px;padding:32px;max-width:560px}}
 h1{{margin:0 0 4px;font-size:20px}} p{{color:#a1a1aa;font-size:14px;line-height:1.6}}
 code{{background:#18181b;padding:2px 6px;border-radius:4px;font-size:13px}}
 button{{background:#10b981;color:#000;border:0;padding:12px 24px;border-radius:8px;
        font-weight:700;font-size:15px;cursor:pointer;margin-top:16px}}
 .hint{{margin-top:20px;padding:12px;background:#18181b;border-radius:8px;font-size:13px}}
</style></head><body>
<div class="card">
  <h1>Test payment</h1>
  <p>Order <code>{order_id}</code> &middot; <b>Rs {rupees}</b></p>
  <p>Paying this fires real <code>payment.*</code> webhooks at your ingress.
     To exercise the interesting path, <b>fail it first</b> and then retry -
     that is scenario A, and the baseline double-charges on it.</p>
  <button onclick="pay()">Open Razorpay checkout</button>
  <div class="hint">
    <b>Test cards.</b> Success <code>4111 1111 1111 1111</code> &middot;
    Failure <code>5104 0600 0000 0008</code><br>
    Any future expiry, any CVV. UPI test id: <code>success@razorpay</code>
    or <code>failure@razorpay</code>.
  </div>
</div>
<script>
function pay() {{
  new Razorpay({{
    key: "{key_id}",
    order_id: "{order_id}",
    amount: {amount},
    currency: "INR",
    name: "Acme Store",
    description: "Nishchay live test",
    handler: function (r) {{
      document.querySelector('.card').innerHTML =
        '<h1>Submitted</h1><p>payment_id <code>' + r.razorpay_payment_id +
        '</code></p><p>Watch your worker log - the webhook is on its way.</p>';
    }},
    modal: {{ ondismiss: function () {{ console.log('dismissed'); }} }},
  }}).open();
}}
</script></body></html>"""


def _auth() -> tuple[str, str]:
    cfg = settings()
    if not cfg.rzp_key_secret:
        raise SystemExit(
            "RZP_KEY_SECRET is empty.\n"
            "  Razorpay Dashboard -> Account & Settings -> API Keys -> Generate Test Key\n"
            "  Put both halves in .env as RZP_KEY_ID and RZP_KEY_SECRET."
        )
    if cfg.require_test_mode and not cfg.is_test_mode:
        raise SystemExit(f"refusing: {cfg.rzp_key_id} is not a test key (E15)")
    return cfg.rzp_key_id, cfg.rzp_key_secret


async def create_order(amount_paise: int, receipt: str | None = None) -> dict[str, Any]:
    """POST /v1/orders. A real order, against a real account."""
    cfg = settings()
    payload = {
        "amount": amount_paise,           # I1: int paise, never a float
        "currency": "INR",
        "receipt": receipt or f"nishchay_{int(time.time())}",
        "notes": {"source": "nishchay live test"},
    }
    async with httpx.AsyncClient(base_url=cfg.rzp_api_base, auth=_auth(), timeout=20) as c:
        r = await c.post("/v1/orders", json=payload)
        if r.status_code >= 400:
            raise SystemExit(f"Razorpay rejected the order ({r.status_code}): {r.text[:300]}")
        return r.json()


async def fetch_attempts(order_id: str) -> list[dict]:
    """GET /v1/orders/:id/payments - the sibling set I3 turns on."""
    cfg = settings()
    async with httpx.AsyncClient(base_url=cfg.rzp_api_base, auth=_auth(), timeout=20) as c:
        r = await c.get(f"/v1/orders/{order_id}/payments")
        r.raise_for_status()
        return r.json().get("items", [])


async def capture(payment_id: str, amount_paise: int) -> dict:
    """Move real test-mode money. The one command here that is not read-only."""
    cfg = settings()
    async with httpx.AsyncClient(base_url=cfg.rzp_api_base, auth=_auth(), timeout=20) as c:
        r = await c.post(
            f"/v1/payments/{payment_id}/capture",
            json={"amount": amount_paise, "currency": "INR"},
        )
        if r.status_code >= 400:
            raise SystemExit(f"capture failed ({r.status_code}): {r.text[:300]}")
        return r.json()


#: What the fold actually consumes. Registering more is harmless; missing
#: one means the resolver never sees the event that would settle a case.
WEBHOOK_EVENTS = [
    "payment.authorized", "payment.failed", "payment.captured",
    "order.paid", "refund.created", "refund.processed", "refund.failed",
]


async def register_webhook(public_url: str) -> dict:
    """Point Razorpay at our tunnel, over the API.

    The dashboard can do this by hand, but a quick tunnel gets a new
    hostname every restart, so by hand means re-pasting a URL every time.

    Two shapes to get right: `events` is a map of name -> 1, not a list
    (a list is echoed back as "Invalid event name/names: 1, 2, 3"), and
    `secret` here is the WEBHOOK secret, which is not the API secret.
    """
    cfg = settings()
    hook = public_url.rstrip("/") + "/webhook/razorpay"
    async with httpx.AsyncClient(base_url=cfg.rzp_api_base, auth=_auth(), timeout=25) as c:
        existing = (await c.get("/v1/webhooks")).json().get("items", [])
        for w in existing:
            if w.get("url") == hook:
                return w                       # already pointed at us
        r = await c.post("/v1/webhooks", json={
            "url": hook,
            "secret": cfg.rzp_webhook_secret,
            "events": {e: 1 for e in WEBHOOK_EVENTS},
        })
        if r.status_code >= 400:
            raise SystemExit(f"webhook registration failed ({r.status_code}): {r.text[:300]}")
        return r.json()


async def list_webhooks() -> list[dict]:
    cfg = settings()
    async with httpx.AsyncClient(base_url=cfg.rzp_api_base, auth=_auth(), timeout=25) as c:
        r = await c.get("/v1/webhooks")
        r.raise_for_status()
        return r.json().get("items", [])


async def delete_webhook(hook_id: str) -> None:
    cfg = settings()
    async with httpx.AsyncClient(base_url=cfg.rzp_api_base, auth=_auth(), timeout=25) as c:
        r = await c.delete(f"/v1/webhooks/{hook_id}")
        if r.status_code >= 400:
            raise SystemExit(f"delete failed ({r.status_code}): {r.text[:200]}")


async def register_whatsapp_templates(waba_id: str) -> list[dict]:
    """Register our DLT templates with Meta as UTILITY templates.

    WhatsApp requires its own approval on top of DLT - the same rule from
    a second regulator. Registering them is what lets a recovery message
    reach a customer who has *not* messaged us in the last 24 hours,
    which is every real customer.

    The {{n}} placeholders are positional and must follow
    `Template.variables`, or the slots fill with the wrong values.
    """
    from core.intents import TEMPLATE_REGISTRY

    cfg = settings()
    out = []
    async with httpx.AsyncClient(
        base_url="https://graph.facebook.com/v21.0",
        headers={"Authorization": f"Bearer {cfg.whatsapp_access_token}"},
        timeout=40,
    ) as c:
        for t in TEMPLATE_REGISTRY.values():
            if not t.whatsapp_name:
                continue
            body = t.body
            example = []
            for i, var in enumerate(t.variables, start=1):
                body = body.replace("{" + var + "}", "{{" + str(i) + "}}")
                example.append(_EXAMPLES.get(var, "sample"))
            r = await c.post(f"/{waba_id}/message_templates", json={
                "name": t.whatsapp_name,
                "language": "en_US",
                "category": "UTILITY",
                "components": [{
                    "type": "BODY", "text": body,
                    "example": {"body_text": [example]},
                }],
            })
            d = r.json()
            out.append({
                "template": t.template_id, "whatsapp_name": t.whatsapp_name,
                "http": r.status_code,
                "status": d.get("status") or (d.get("error") or {}).get("message", "")[:120],
            })
    return out


_EXAMPLES = {
    "amount": "2340", "merchant": "Acme Store", "method": "netbanking",
    "link": "https://rzp.io/rzp/abc123", "window": "2 hours",
}


async def register_whatsapp_hook(public_url: str, waba_id: str) -> dict:
    """Subscribe Meta's status webhook to our tunnel.

    Without this, `EXECUTED` means "the API accepted it" - which live
    testing proved can be false. Meta returned a message id for a message
    that was never delivered, and a human checking their phone was the
    only thing that caught it. Delivery receipts turn that into a fact.
    """
    cfg = settings()
    callback = public_url.rstrip("/") + "/webhook/whatsapp"
    async with httpx.AsyncClient(
        base_url="https://graph.facebook.com/v21.0",
        headers={"Authorization": f"Bearer {cfg.whatsapp_access_token}"},
        timeout=40,
    ) as c:
        # The app-level subscription carries the callback; the WABA-level
        # one says which account's events to send. Both are required, and
        # missing either produces silence rather than an error.
        app_id = cfg.whatsapp_app_id
        if app_id:
            r = await c.post(f"/{app_id}/subscriptions", data={
                "object": "whatsapp_business_account",
                "callback_url": callback,
                "verify_token": cfg.whatsapp_verify_token,
                "fields": "messages",
            })
            if r.status_code >= 400:
                return {"ok": False, "step": "app subscription",
                        "detail": r.text[:300], "callback": callback}
        r = await c.post(f"/{waba_id}/subscribed_apps")
        return {
            "ok": r.status_code < 400,
            "step": "waba subscription",
            "detail": r.text[:200],
            "callback": callback,
        }


# ------------------------------- doctor -------------------------------

async def doctor(url: str) -> int:
    """Say exactly what is configured and what is not.

    Written because the failure modes here are all silent: a wrong webhook
    secret looks like no traffic, and a missing tunnel looks like Razorpay
    ignoring you.
    """
    cfg = settings()
    ok = True

    def line(label: str, good: bool, detail: str) -> None:
        nonlocal ok
        ok = ok and good
        print(f"  [{'OK ' if good else 'XX'}] {label:<22} {detail}")

    print("razorpay")
    line("key id", cfg.rzp_key_id != "rzp_test_placeholder", cfg.rzp_key_id)
    line("key secret", bool(cfg.rzp_key_secret), "set" if cfg.rzp_key_secret else "EMPTY - see README")
    line("test mode", cfg.is_test_mode, "test" if cfg.is_test_mode else "LIVE KEY - refusing")
    line("webhook secret", bool(cfg.rzp_webhook_secret),
         "set (must match the Dashboard value, NOT the API secret)")

    if cfg.rzp_key_secret and cfg.is_test_mode:
        try:
            async with httpx.AsyncClient(base_url=cfg.rzp_api_base, auth=_auth(), timeout=15) as c:
                r = await c.get("/v1/payments?count=1")
            line("api reachable", r.status_code == 200, f"HTTP {r.status_code}")
        except Exception as e:                       # noqa: BLE001
            line("api reachable", False, f"{type(e).__name__}: {e}")

    print("\ningress")
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            h = await c.get(f"{url}/health")
        d = h.json()
        line("local ingress", h.status_code == 200, f"{url} - degraded={d.get('degraded') or 'none'}")
        line("events received", True, json.dumps(d.get("stats", {})))
    except Exception as e:                           # noqa: BLE001
        line("local ingress", False, f"not running: {e}")

    print("\nwebhooks")
    try:
        hooks = await list_webhooks()
        line("registered", bool(hooks), f"{len(hooks)} configured")
        for w in hooks:
            state = "active" if w.get("active") else "INACTIVE"
            print(f"       {w['id']}  {state}  {w['url']}")
        if not hooks:
            print("       Razorpay must reach you over the public internet; localhost")
            print("       will not do. Start a tunnel, then point Razorpay at it:")
            print("         tools\\cloudflared.exe tunnel --url http://localhost:8000")
            print("         python -m harness.live hook https://<name>.trycloudflare.com")
    except Exception as e:                       # noqa: BLE001
        line("registered", False, f"could not list: {e}")

    print("\n" + ("ready for a live run" if ok else "not ready - fix the XX rows above"))
    return 0 if ok else 1


# -------------------------------- watch -------------------------------

async def watch(order_id: str | None, interval: float = 3.0, limit: int = 200) -> None:
    """Follow the event store and re-fold as observations land.

    This is the honest view: it reads what the ingress actually stored, so
    nothing appears here that Razorpay did not really send.
    """
    use_psycopg_compatible_loop()
    infra = Infra()
    await infra.start()
    print(f"[watch] store={type(infra.store).__name__} "
          f"degraded={infra.degraded or 'none'}\n")

    seen: set[str] = set()
    last_verdict = None
    try:
        for _ in range(limit):
            oids = [order_id] if order_id else await _known_orders(infra)
            for oid in oids:
                obs = await infra.store.load(oid)
                for o in sorted(obs, key=lambda x: x.received_at):
                    if o.event_id in seen:
                        continue
                    seen.add(o.event_id)
                    print(f"  <- {o.event_type:<20} {o.payment_id or '':<20} "
                          f"event_time={o.event_time} skew={o.skew}s")
                if obs:
                    v = fold(obs, int(time.time()), order_id=oid)
                    key = (oid, v.verdict, round(v.confidence, 2))
                    if key != last_verdict:
                        last_verdict = key
                        print(f"     VERDICT {v.verdict.value} @ {v.confidence:.2f} "
                              f"-> {v.proposed_action.value}   rules={list(v.rules_fired)}")
                        if v.proposed_action == Action.CAPTURE:
                            pid = next((p.payment_id for p in _authorized(obs)), None)
                            if pid:
                                print(f"     $ python -m harness.live capture {pid}")
            await asyncio.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        await infra.stop()


def _authorized(obs):
    from core.fold import build_state
    from core.verdicts import PaymentStatus

    st = build_state(obs, obs[0].order_id if obs else "")
    return [p for p in st.payments.values() if p.status == PaymentStatus.AUTHORIZED]


async def _known_orders(infra) -> list[str]:
    if hasattr(infra.store, "all_orders"):
        return await infra.store.all_orders()
    async with infra.pool.acquire() as c:
        rows = await c.fetch(
            "SELECT DISTINCT order_id FROM observations "
            "ORDER BY order_id DESC LIMIT 20"
        )
    return [r["order_id"] for r in rows]


# --------------------------------- cli --------------------------------

async def main_async(args) -> int:
    if args.cmd == "doctor":
        return await doctor(args.url)

    if args.cmd == "order":
        cfg = settings()
        order = await create_order(args.amount)
        page = CHECKOUT_HTML.format(
            key_id=cfg.rzp_key_id, order_id=order["id"],
            amount=order["amount"], rupees=order["amount"] / 100,
        )
        path = "checkout.html"
        with open(path, "w", encoding="utf-8") as f:
            f.write(page)
        print(f"order   {order['id']}")
        print(f"amount  Rs {order['amount'] / 100:,.2f} ({order['amount']} paise)")
        print(f"status  {order['status']}")
        print(f"\npay it: {path}  (opening in your browser)")
        print(f"watch : python -m harness.live watch --order {order['id']}")
        if not args.no_open:
            webbrowser.open(f"file://{__import__('os').path.abspath(path)}")
        return 0

    if args.cmd == "hook":
        if args.delete:
            await delete_webhook(args.delete)
            print(f"deleted {args.delete}")
            return 0
        if args.url_public:
            w = await register_webhook(args.url_public)
            print(f"webhook {w['id']}")
            print(f"  url    {w['url']}")
            print(f"  active {w.get('active')}")
            print(f"  events {sorted(k for k, v in (w.get('events') or {}).items() if v)}")
            return 0
        hooks = await list_webhooks()
        print(f"{len(hooks)} webhook(s) registered")
        for w in hooks:
            print(f"  {w['id']:<18} {'active' if w.get('active') else 'INACTIVE':<9} {w['url']}")
        return 0

    if args.cmd == "hook-whatsapp":
        out = await register_whatsapp_hook(args.url_public, args.waba)
        print(f"  callback : {out['callback']}")
        print(f"  step     : {out['step']}")
        print(f"  result   : {'OK' if out['ok'] else 'FAILED'}  {out['detail']}")
        if out["ok"]:
            print("\n  Delivery receipts will now arrive at /api/delivery.")
        return 0 if out["ok"] else 1

    if args.cmd == "templates":
        rows = await register_whatsapp_templates(args.waba)
        for r in rows:
            print(f"  {r['template']:<20} -> {r['whatsapp_name']:<20} "
                  f"HTTP {r['http']}  {r['status']}")
        return 0

    if args.cmd == "attempts":
        items = await fetch_attempts(args.order)
        print(f"{len(items)} attempt(s) on {args.order}")
        for i in items:
            print(f"  {i['id']:<22} {i['status']:<12} {i.get('method',''):<12} "
                  f"{i['amount']} paise  err={i.get('error_reason')}")
        return 0

    if args.cmd == "capture":
        items = await fetch_attempts(args.order) if args.order else []
        amount = args.amount or next(
            (i["amount"] for i in items if i["id"] == args.payment_id), None
        )
        if amount is None:
            print("need --amount (paise), or --order so it can be looked up")
            return 1
        print(f"capturing {args.payment_id} for {amount} paise ...")
        out = await capture(args.payment_id, amount)
        print(f"  status={out.get('status')} captured={out.get('captured')}")
        return 0

    if args.cmd == "watch":
        await watch(args.order)
        return 0

    return 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Live Razorpay test-mode loop.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="what is configured, what is missing")
    d.add_argument("--url", default="http://localhost:8000")

    o = sub.add_parser("order", help="create a real order and open checkout")
    o.add_argument("--amount", type=int, default=234000, help="paise (default 234000 = Rs 2340)")
    o.add_argument("--no-open", action="store_true")

    h = sub.add_parser("hook", help="register / list / delete the webhook")
    h.add_argument("url_public", nargs="?",
                   help="public tunnel base URL; omit to just list")
    h.add_argument("--delete", metavar="HOOK_ID")

    hw = sub.add_parser("hook-whatsapp", help="subscribe Meta delivery receipts")
    hw.add_argument("url_public", help="public tunnel base URL")
    hw.add_argument("waba", help="WhatsApp Business Account ID")

    t = sub.add_parser("templates", help="register the DLT templates with Meta")
    t.add_argument("waba", help="WhatsApp Business Account ID")

    a = sub.add_parser("attempts", help="every sibling attempt on an order")
    a.add_argument("order")

    c = sub.add_parser("capture", help="capture an authorised payment (moves money)")
    c.add_argument("payment_id")
    c.add_argument("--order")
    c.add_argument("--amount", type=int)

    w = sub.add_parser("watch", help="follow observations and verdicts")
    w.add_argument("--order")

    sys.exit(asyncio.run(main_async(ap.parse_args())))


if __name__ == "__main__":
    main()
