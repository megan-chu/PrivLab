"""
types.py - extraction data: pack fields (what the user defined) and the model's reply schemas.

Schemas have no default values on purpose: strict JSON output needs every key present.
"""
from dataclasses import asdict, dataclass, field
from typing import Literal

from pydantic import BaseModel

READINESS = ("draft", "learning", "ready")


@dataclass
class Field:
    key: str                    # e.g. "tumour_size"
    name: str                   # the user's words: "tumour size"
    description: str            # the user's own description
    meaning: str                # the confirmed meaning question
    type: str                   # "number" | "choice"
    unit: str | None = None     # number fields, e.g. "cm"
    range: list[float] | None = None
    choices: list[str] = field(default_factory=list)
    readiness: str = "draft"
    recent: list[str] = field(default_factory=list)          # last outcomes: accepted / corrected
    examples: dict = field(default_factory=lambda: {"good": [], "bad": []})
    history: list[dict] = field(default_factory=list)
    definition: str = ""        # what counts, where it is written, what to leave out
    version: int = 1            # +1 when the meaning changes; scores and examples start again

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Model reply schemas
# --------------------------------------------------------------------------
class FieldDraft(BaseModel):
    """define_field: the model's suggested definition, shown to the user to accept or edit."""
    key: str
    meaning_question: str
    definition: str
    type: Literal["number", "choice"]
    unit: str | None
    range_min: float | None
    range_max: float | None
    choices: list[str]


class CandidateReply(BaseModel):
    value_text: str             # exactly as written in the report, e.g. "2.0" or "adenocarcinoma"
    unit_text: str | None       # as written, e.g. "cm", "mm"
    evidence_text: str          # the exact sentence or phrase it comes from
    reason: str
    confidence: float


class FieldCandidates(BaseModel):
    key: str
    candidates: list[CandidateReply]


class CandidatesReply(BaseModel):
    fields: list[FieldCandidates]
