"""
define.py - define_field: the user names a field in plain words; Muse Spark drafts its meaning,
type and unit; the user accepts (or edits) and it is saved to the pack.

Only the field name and the user's description are sent, never report text.
"""
import re
from datetime import datetime, timedelta, timezone

from common import spark
from extract.pack import get_field, load_pack, meaning_key, put_field, save_pack
from extract.types import Field, FieldDraft

SGT = timezone(timedelta(hours=8))
INSTRUCTIONS = (
    "A cancer registrar wants to extract a field from pathology reports. Draft its definition.\n"
    "key: short snake_case. meaning_question: one plain question to confirm what they mean, e.g. "
    "'Largest tumour dimension on the surgical specimen?'. definition: 2-3 short sentences a registrar "
    "could apply without asking: exactly what counts, where in a pathology report it is usually written, "
    "and what not to count (e.g. history, another specimen, a metastasis, a differential). type: 'number' (a measurement or count) or "
    "'choice' (one of a fixed list). For number: unit (e.g. 'cm') and a plausible range; choices empty. "
    "For choice: the usual options as they appear in reports; unit and range null."
)


def define_field(name: str, description: str | None = None, accept: dict | None = None, *, pack_path=None) -> dict:
    """Call 1 (accept=None): returns a draft to confirm. Call 2 (accept=draft, possibly edited): saves it."""
    pack = load_pack(pack_path)
    if accept is None:
        prompt = f"{INSTRUCTIONS}\n\nField name: {name}\nUser's description: {description or '(none)'}"
        draft = spark.ask(prompt, schema=FieldDraft, step="define")
        return {"next": "confirm", "draft": draft.model_dump(),
                "question": draft.meaning_question, "note": "Accept as is or edit, then call again with accept=draft."}

    key = re.sub(r"[^a-z0-9_]", "_", accept["key"].lower()).strip("_") or "field"
    rng = [accept["range_min"], accept["range_max"]] if accept.get("range_min") is not None else None
    f = Field(key=key, name=name, description=description or "", meaning=accept["meaning_question"],
              definition=accept.get("definition") or "", type=accept["type"], unit=accept.get("unit"), range=rng, choices=list(accept.get("choices") or []),
              history=[{"when": datetime.now(SGT).isoformat(timespec="seconds"), "what": "created"}])
    existing = get_field(pack, key)
    if existing and meaning_key(existing) == meaning_key(f):           # same meaning: keep what it learned
        f.version, f.examples, f.recent, f.readiness = existing.version, existing.examples, existing.recent, existing.readiness
        f.history = existing.history + [{"when": f.history[0]["when"], "what": "redefined, same meaning"}]
    elif existing:            # a new meaning: old examples and scores were confirmed under the old one
        f.version = existing.version + 1
        f.history = existing.history + [{"when": f.history[0]["when"],
                                         "what": f"redefined as v{f.version}; score and examples start again"}]
    put_field(pack, f)
    save_pack(pack, pack_path)
    return {"next": "done", "field": f.to_dict()}
