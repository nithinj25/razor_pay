"""WhatsApp delivery via Meta's Cloud API.

The strategist already selects a DLT-registered template and the gate has
already validated that the template is approved for the WhatsApp channel,
that the variables fit 5x30, and that DND, consent and opt-out permit it.
Nothing here re-decides any of that. This module only moves the bytes.

Two WhatsApp rules shape the code:

* **The 24-hour session window.** Freeform text is only permitted inside
  24 hours of the *customer* messaging the business. Outside it, Meta
  accepts pre-approved templates only. Our recovery message carries a
  payment link, so it needs the window open - and the failure when it is
  not is a specific error code, which we surface as an instruction rather
  than a stack trace.
* **Test numbers can only message an allow-list.** Meta's free test
  number reaches at most five numbers you have verified. `demo_to`
  overrides the customer's real contact for exactly that reason, and the
  outcome records that it did.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from core.config import Settings, settings

#: Meta's code for "the 24-hour customer service window has closed".
#: Worth naming: it is the single most common reason a correctly built
#: integration silently stops delivering.
WINDOW_CLOSED = 131047

#: Recipient is not on the test number's verified allow-list.
NOT_ALLOWED = 131030


@dataclass
class SendResult:
    ok: bool
    detail: str
    message_id: str = ""
    request: dict[str, Any] | None = None
    response: dict[str, Any] | None = None


class WhatsAppSender:
    """Meta Cloud API client. Configured or not - it says which."""

    def __init__(self, cfg: Settings | None = None, client: Any = None) -> None:
        self.cfg = cfg or settings()
        self._client = client

    @property
    def configured(self) -> bool:
        return bool(self.cfg.whatsapp_access_token and self.cfg.whatsapp_phone_number_id)

    @property
    def uses_demo_recipient(self) -> bool:
        """True when sends are redirected away from the real customer.

        The executor asks the sender rather than reading config itself:
        the sender is what performs the substitution, so it is the only
        thing that can honestly say whether one happened.
        """
        return bool(self.cfg.demo_whatsapp_to)

    def recipient_for(self, contact: str | None) -> str:
        """Who actually receives this.

        In production it is the customer's number from the payment entity.
        On a Meta test number only allow-listed numbers can be reached, so
        a configured demo recipient wins - and the caller records that the
        substitution happened rather than quietly pretending otherwise.
        """
        if self.cfg.demo_whatsapp_to:
            return _digits(self.cfg.demo_whatsapp_to)
        return _digits(contact or "")

    async def send_text(self, to: str, body: str) -> SendResult:
        """Freeform message. Requires the 24h window to be open."""
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": _digits(to),
            "type": "text",
            # Link previews off: the message is a payment instruction, and
            # a preview card of the checkout page adds nothing but noise.
            "text": {"preview_url": False, "body": body},
        }
        return await self._post(payload)

    async def send_template(
        self, to: str, name: str = "hello_world", language: str = "en_US"
    ) -> SendResult:
        """Pre-approved template. Works outside the 24h window."""
        payload = {
            "messaging_product": "whatsapp",
            "to": _digits(to),
            "type": "template",
            "template": {"name": name, "language": {"code": language}},
        }
        return await self._post(payload)

    async def _post(self, payload: dict) -> SendResult:
        if not self.configured:
            return SendResult(False, "WhatsApp not configured", request=payload)

        client = self._client
        owned = client is None
        if owned:
            client = httpx.AsyncClient(
                base_url=self.cfg.whatsapp_api_base,
                headers={"Authorization": f"Bearer {self.cfg.whatsapp_access_token}"},
                timeout=30.0,
            )
        try:
            r = await client.post(
                f"/{self.cfg.whatsapp_phone_number_id}/messages", json=payload
            )
            data = r.json() if r.content else {}
            if r.status_code >= 400:
                return SendResult(False, _explain(data), request=payload, response=data)
            mid = (data.get("messages") or [{}])[0].get("id", "")
            return SendResult(True, f"delivered to {payload['to']}", mid, payload, data)
        except Exception as e:                      # noqa: BLE001
            # A send failure is an outcome, not an exception (I9).
            return SendResult(False, f"{type(e).__name__}: {e}", request=payload)
        finally:
            if owned:
                await client.aclose()


def _digits(number: str) -> str:
    """Meta wants country code + number, no '+', spaces or dashes."""
    return "".join(ch for ch in str(number) if ch.isdigit())


def _explain(data: dict) -> str:
    """Turn Meta's error into something actionable.

    The two failures that actually happen in a demo both look like a
    generic 400 otherwise, and both have a one-line fix.
    """
    err = data.get("error") or {}
    code = err.get("code")
    sub = (err.get("error_data") or {}).get("details", "")
    if code == WINDOW_CLOSED:
        return ("24h session window closed - reply to the WhatsApp thread from "
                "the recipient's phone to reopen it, then retry")
    if code == NOT_ALLOWED:
        return ("recipient not on the test number's allow-list - add it under "
                "'To' -> Manage phone number list in the Meta dashboard")
    return f"{err.get('message', 'send failed')}{f' ({sub})' if sub else ''}"
