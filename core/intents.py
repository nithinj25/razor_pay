"""The intent schema and the DLT template registry.

The model does not write messages. It *selects a registered template and
fills its slots*. That is forced by TRAI/TCCCPR - DLT templates must be
pre-registered and are capped at 5 variables of 30 characters - and it
has a second effect worth stating plainly in review: prompt injection
cannot produce an arbitrary outbound message, because there is no code
path from model output to free text. The compliance constraint and the
security constraint are the same constraint.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, conlist, constr

from core.verdicts import Action

#: DLT hard limits. Not ours - the registry's.
MAX_VARIABLES = 5
MAX_VARIABLE_LEN = 30


class Channel(StrEnum):
    SMS = "SMS"
    WHATSAPP = "WHATSAPP"
    EMAIL = "EMAIL"
    VOICE = "VOICE"


class Category(StrEnum):
    """TCCCPR message classes.

    PROMOTIONAL is present so the gate can reject it explicitly rather
    than by omission - a recovery message that carries an incentive is
    promotional, and is blocked outright for DND subscribers.
    """

    SERVICE_IMPLICIT = "SERVICE_IMPLICIT"
    SERVICE_EXPLICIT = "SERVICE_EXPLICIT"
    TRANSACTIONAL = "TRANSACTIONAL"
    PROMOTIONAL = "PROMOTIONAL"


class Template(BaseModel):
    """A DLT-registered template. The registry is the whitelist."""

    model_config = {"frozen": True}

    template_id: str
    dlt_id: str
    body: str
    variables: tuple[str, ...]
    channels: frozenset[Channel]
    category: Category
    #: The Meta-approved template name for this message, if registered.
    #: WhatsApp requires its own registration on top of DLT - the same
    #: constraint from a second regulator - and the parameter order must
    #: match `variables` exactly, or slots fill with the wrong values and
    #: nobody notices until a customer reads it.
    whatsapp_name: str | None = None

    def render(self, values: list[str]) -> str:
        out = self.body
        for name, val in zip(self.variables, values):
            out = out.replace("{" + name + "}", val)
        return out


#: In production this is pulled from the DLT portal. Here it is a literal
#: so the gate has something concrete to refuse.
TEMPLATE_REGISTRY: dict[str, Template] = {
    "RCV_RETRY": Template(
        template_id="RCV_RETRY",
        whatsapp_name="rcv_retry",
        dlt_id="1207162458392017001",
        body="Your payment of Rs {amount} to {merchant} did not go through. "
             "Complete it here: {link}",
        variables=("amount", "merchant", "link"),
        channels=frozenset({Channel.SMS, Channel.WHATSAPP, Channel.EMAIL}),
        category=Category.SERVICE_IMPLICIT,
    ),
    "RCV_UPI_ALT": Template(
        template_id="RCV_UPI_ALT",
        whatsapp_name="rcv_upi_alt",
        dlt_id="1207162458392017002",
        body="Your {method} payment of Rs {amount} to {merchant} could not be "
             "processed. Pay by UPI instead: {link}",
        variables=("method", "amount", "merchant", "link"),
        channels=frozenset({Channel.SMS, Channel.WHATSAPP}),
        category=Category.SERVICE_IMPLICIT,
    ),
    "RCV_DOWNTIME_WAIT": Template(
        template_id="RCV_DOWNTIME_WAIT",
        whatsapp_name="rcv_downtime_wait",
        dlt_id="1207162458392017003",
        body="Your bank is temporarily unavailable. Your payment of Rs {amount} "
             "to {merchant} has not been charged. Please retry after {window}.",
        variables=("amount", "merchant", "window"),
        channels=frozenset({Channel.SMS, Channel.WHATSAPP}),
        category=Category.SERVICE_IMPLICIT,
    ),
}


def template_registered(template_id: str | None, channel: Channel | None) -> bool:
    """A template is only approved *for a channel*. Both must match.

    This is where the two axes recombine: the payment rail the strategist
    chose and the delivery pipe it chose are validated as a pair, by the
    gate, not by the model that picked them.
    """
    if template_id is None or channel is None:
        return False
    t = TEMPLATE_REGISTRY.get(template_id)
    return t is not None and channel in t.channels


class RecoveryIntent(BaseModel):
    """What the strategist emits. It is a proposal, never an instruction.

    Every field here is treated as untrusted by the gate (I8). The schema
    bounds what can be *said*; the gate bounds what can be *done*.
    """

    model_config = {"frozen": True}

    action: Action
    template_id: Literal["RCV_UPI_ALT", "RCV_RETRY", "RCV_DOWNTIME_WAIT"] | None = None
    variables: conlist(constr(max_length=MAX_VARIABLE_LEN), max_length=MAX_VARIABLES) = []
    channel: Channel | None = None
    category: Category = Category.SERVICE_IMPLICIT
    method_hint: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""

    #: Set by the resolver, not the model - the gate keys idempotency on
    #: it so that genuinely new evidence may act again while a repeat of
    #: the same decision cannot.
    evidence_version: str = ""


class Assessment(BaseModel):
    """One turn of the strategist loop. Forced `tool_choice` target.

    The model cannot return prose because this schema is the only shape
    it is allowed to emit.
    """

    model_config = {"frozen": True}

    reasoning: str
    next_probe: Literal["probe_downtime", "probe_history", "compose"]
    confidence: float = Field(ge=0.0, le=1.0)
