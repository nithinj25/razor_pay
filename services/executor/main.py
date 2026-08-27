"""The executor. One module, one place money moves.

Everything upstream produces opinions. This is the only code that can
change the world, which is why it is deliberately dull: no branching on
model output, no retries that could double-fire, no cleverness. It takes
an already-gated decision and performs exactly the action named.

Idempotency is keyed on (payment_id, action, evidence_version). Keying on
the action alone would block a legitimate second attempt after genuinely
new evidence; keying on nothing would double-charge on a retry.

Channel honesty: Razorpay payment links notify natively over SMS and
email. WhatsApp requires an approved BSP on the WhatsApp Business API,
and voice requires telephony - neither is in scope. Rather than silently
sending an SMS while the console claims WhatsApp, those dispatch as
STUBBED with the exact payload they would have sent, and the outcome
records it. A stub that announces itself is defensible in review; one
that quietly downgrades is not.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from core.config import Settings, settings
from core.intents import TEMPLATE_REGISTRY, Channel, RecoveryIntent
from core.verdicts import Action
from services.gate.rules import GateDecision

#: Channels Razorpay payment links notify natively.
LIVE_CHANNELS = frozenset({Channel.SMS, Channel.EMAIL})


@dataclass
class Outcome:
    order_id: str
    payment_id: str
    action: Action
    status: str                      # EXECUTED | STUBBED | SKIPPED | VETOED | FAILED
    idem_key: str
    detail: str = ""
    request: dict[str, Any] = field(default_factory=dict)
    response: dict[str, Any] = field(default_factory=dict)
    at: int = field(default_factory=lambda: int(time.time()))

    def to_row(self) -> dict[str, Any]:
        return {
            "ts": self.at,
            "order_id": self.order_id,
            "payment_id": self.payment_id,
            "action": self.action.value,
            "status": self.status,
            "idem_key": self.idem_key,
            "detail": self.detail,
            "request": json.dumps(self.request),
        }


class RazorpayClient:
    """Thin async wrapper. Test mode only (E15)."""

    def __init__(self, cfg: Settings | None = None, client: httpx.AsyncClient | None = None):
        self.cfg = cfg or settings()
        self._client = client

    async def _post(self, path: str, payload: dict, idem_key: str) -> dict:
        c = self._client or httpx.AsyncClient(
            base_url=self.cfg.rzp_api_base,
            auth=(self.cfg.rzp_key_id, self.cfg.rzp_key_secret),
            timeout=10.0,
        )
        try:
            r = await c.post(path, json=payload, headers={"X-Payment-Idempotency": idem_key})
            r.raise_for_status()
            return r.json()
        finally:
            if self._client is None:
                await c.aclose()

    async def capture(self, payment_id: str, amount: int, idem_key: str) -> dict:
        return await self._post(
            f"/v1/payments/{payment_id}/capture",
            {"amount": amount, "currency": "INR"},
            idem_key,
        )

    async def payment_link(self, payload: dict, idem_key: str) -> dict:
        return await self._post("/v1/payment_links", payload, idem_key)

    async def refund(self, payment_id: str, amount: int, idem_key: str) -> dict:
        return await self._post(
            f"/v1/payments/{payment_id}/refund", {"amount": amount}, idem_key
        )


class Executor:
    """Executes gated decisions. `dry_run` keeps the demo hermetic."""

    def __init__(
        self,
        client: RazorpayClient | None = None,
        dry_run: bool = True,
        cfg: Settings | None = None,
        whatsapp: Any = None,
    ):
        from services.executor.whatsapp import WhatsAppSender

        self.cfg = cfg or settings()
        self.client = client or RazorpayClient(self.cfg)
        self.whatsapp = whatsapp if whatsapp is not None else WhatsAppSender(self.cfg)
        self.dry_run = dry_run
        self.executed: dict[str, Outcome] = {}      # idem_key -> outcome
        self.outcomes: list[Outcome] = []
        self.exception_queue: list[dict] = []

    @property
    def seen_keys(self) -> set[str]:
        """Fed back into the gate so a repeat is vetoed, not re-executed."""
        return set(self.executed)

    def _record(self, o: Outcome) -> Outcome:
        self.outcomes.append(o)
        if o.status in ("EXECUTED", "STUBBED"):
            self.executed[o.idem_key] = o
        return o

    async def execute(
        self,
        decision: GateDecision,
        intent: RecoveryIntent,
        order_id: str,
        payment_id: str,
        amount: int,
        merchant: str = "Acme Store",
        contact: str = "",
        email: str = "",
    ) -> Outcome:
        base = dict(order_id=order_id, payment_id=payment_id, idem_key=decision.idem_key)

        if not decision.allowed:
            return self._record(
                Outcome(**base, action=Action.NOOP, status="VETOED", detail=decision.reason)
            )

        # I5 - the last line of defence. The gate should already have
        # caught this; belt and braces, because the cost is a real charge.
        if decision.idem_key in self.executed:
            return self._record(
                Outcome(**base, action=intent.action, status="SKIPPED",
                        detail="idempotency key already executed")
            )

        action = intent.action

        try:
            if action == Action.NOOP:
                return self._record(
                    Outcome(**base, action=action, status="EXECUTED",
                            detail=intent.reasoning or "no action required")
                )

            if action == Action.CAPTURE:
                payload = {"amount": amount, "currency": "INR"}
                if self.dry_run:
                    return self._record(
                        Outcome(**base, action=action, status="STUBBED",
                                detail=f"dry run: capture {amount} paise", request=payload)
                    )
                resp = await self.client.capture(payment_id, amount, decision.idem_key)
                return self._record(
                    Outcome(**base, action=action, status="EXECUTED",
                            detail=f"captured {amount} paise", request=payload, response=resp)
                )

            if action == Action.SEND_RECOVERY_LINK:
                return self._record(
                    await self._send_link(
                        base, intent, decision, amount, merchant, contact, email
                    )
                )

            if action == Action.REFUND:
                payload = {"amount": amount}
                if self.dry_run:
                    return self._record(
                        Outcome(**base, action=action, status="STUBBED",
                                detail=f"dry run: refund {amount} paise", request=payload)
                    )
                resp = await self.client.refund(payment_id, amount, decision.idem_key)
                return self._record(
                    Outcome(**base, action=action, status="EXECUTED",
                            detail="refund initiated", request=payload, response=resp)
                )

            if action == Action.NOTIFY_MERCHANT:
                return self._record(
                    Outcome(**base, action=action, status="EXECUTED",
                            detail="dashboard event raised", request={"note": intent.reasoning})
                )

            if action in (Action.ESCALATE, Action.HOLD):
                # Scenario D ends here, and so does the video. The row
                # carries the identifiers a human can actually act on.
                self.exception_queue.append(
                    {
                        "order_id": order_id,
                        "payment_id": payment_id,
                        "amount": amount,
                        "reason": intent.reasoning,
                        "at": int(time.time()),
                    }
                )
                return self._record(
                    Outcome(**base, action=action, status="EXECUTED",
                            detail="queued for human review")
                )

            if action == Action.VOICE_CALL:
                # Designed, gated, stubbed. The reasoning is real; the
                # telephony is not (GUARDRAILS section 6).
                return self._record(
                    Outcome(**base, action=action, status="STUBBED",
                            detail="voice intent logged; telephony not implemented",
                            request={"template": intent.template_id,
                                     "variables": list(intent.variables)})
                )

            return self._record(
                Outcome(**base, action=action, status="FAILED",
                        detail=f"unknown action {action}")
            )

        except httpx.HTTPError as e:
            # Never raise into the caller: a failed execution is an
            # outcome, and an outcome is data (I9).
            return self._record(
                Outcome(**base, action=action, status="FAILED", detail=f"{type(e).__name__}: {e}")
            )

    async def _link_url(self, payload: dict, decision: GateDecision) -> str:
        """Create the actual Razorpay payment link and return its short URL.

        This is why the model cannot be trusted to fill the {link} slot:
        the URL does not exist until now. In dry run there is no link to
        create, so the placeholder says so rather than inventing one that
        would 404 on a reviewer's phone.
        """
        if self.dry_run or not self.cfg.rzp_key_secret:
            return "[dry-run: no live link created]"
        try:
            resp = await self.client.payment_link(payload, decision.idem_key)
            return resp.get("short_url", "")
        except httpx.HTTPError:
            return ""

    async def _send_link(
        self, base: dict, intent: RecoveryIntent, decision: GateDecision,
        amount: int, merchant: str, contact: str = "", email: str = "",
    ) -> Outcome:
        template = TEMPLATE_REGISTRY.get(intent.template_id or "")
        payload = {
            "amount": amount,
            "currency": "INR",
            "accept_partial": False,
            "description": decision.rendered or (template.body if template else "Payment"),
            "notify": {
                "sms": intent.channel == Channel.SMS,
                "email": intent.channel == Channel.EMAIL,
            },
            "reminder_enable": False,
            # Razorpay notifies from this, and it is where the WhatsApp
            # recipient comes from. Omitting it produced an empty `to`
            # and a "parameter to is required" from Meta.
            "customer": {k: v for k, v in
                         (("contact", contact), ("email", email)) if v},
            "notes": {
                "nishchay_idem": decision.idem_key[:16],
                "template_id": intent.template_id or "",
                "dlt_id": template.dlt_id if template else "",
                "source_order": base["order_id"],
            },
        }
        if intent.method_hint:
            payload["options"] = {"checkout": {"method": {intent.method_hint: "1"}}}

        # WhatsApp goes out over Meta's Cloud API when it is configured.
        # The gate has already cleared the template for this channel, so
        # this is delivery, not a second decision.
        if intent.channel == Channel.WHATSAPP:
            if not self.whatsapp.configured:
                return Outcome(
                    **base, action=Action.SEND_RECOVERY_LINK, status="STUBBED",
                    detail="WhatsApp not configured; link payload built, not dispatched",
                    request=payload,
                )
            # RCV_DOWNTIME_WAIT deliberately has no link variable - it
            # tells the customer they were NOT charged and to wait. Bolting
            # a pay-now link onto that contradicts the message, so only
            # templates that declare a link slot get one.
            wants_link = "link" in (template.variables if template else ())
            link = await self._link_url(payload, decision) if wants_link else ""
            body = _with_link(decision.rendered, link) if wants_link else decision.rendered
            to = self.whatsapp.recipient_for(contact)
            res = await self.whatsapp.send_text(to, body)
            return Outcome(
                **base, action=Action.SEND_RECOVERY_LINK,
                status="EXECUTED" if res.ok else "FAILED",
                detail=(f"WhatsApp -> {to}: {res.detail}"
                        + (" (demo recipient, not the customer)"
                           if self.whatsapp.uses_demo_recipient else "")),
                request={**payload, "whatsapp": res.request},
                response=res.response or {},
            )

        if intent.channel not in LIVE_CHANNELS:
            return Outcome(
                **base, action=Action.SEND_RECOVERY_LINK, status="STUBBED",
                detail=f"{intent.channel.value if intent.channel else 'unknown'} delivery "
                       f"requires an approved BSP; link payload built, not dispatched",
                request=payload,
            )

        if self.dry_run:
            return Outcome(
                **base, action=Action.SEND_RECOVERY_LINK, status="STUBBED",
                detail=f"dry run: link over {intent.channel.value}", request=payload,
            )

        resp = await self.client.payment_link(payload, decision.idem_key)
        return Outcome(
            **base, action=Action.SEND_RECOVERY_LINK, status="EXECUTED",
            detail=f"link sent over {intent.channel.value}: {resp.get('short_url', '')}",
            request=payload, response=resp,
        )




def _with_link(rendered: str, link: str) -> str:
    """Replace whatever stands in the {link} slot with the real URL.

    Only called for templates that declare a `link` variable.

    The payment link does not exist until the executor creates it, so the
    model cannot know it - and asked for a link variable, a model will
    confidently invent one ("https://pay.acme/alt"). Matching only the
    literal "{link}" left the invented URL in place and appended the real
    one, so the customer received two links, the first of which 404s.

    So: whatever occupies the final token, replace it. A URL, a
    placeholder, or the bare word "link" all mean the same thing here.
    """
    if not link:
        return rendered
    if link in rendered:
        return rendered

    parts = rendered.rstrip().rsplit(" ", 1)
    if len(parts) == 2:
        head, last = parts
        looks_like_a_slot = (
            last.startswith(("http://", "https://", "{"))
            or last.rstrip(".").lower() in ("link", "url")
        )
        if looks_like_a_slot:
            return f"{head} {link}"
    return f"{rendered.rstrip()} {link}"
