"""
pack.py - the user's knowledge pack (packs/default.yaml, git-ignored). Starts empty.

Each field is defined once by the user (define_field) and then learns from confirmations.
"""
import yaml

from common.config import ROOT
from extract.types import Field

PACK_PATH = ROOT / "packs" / "default.yaml"


def meaning_key(f: Field) -> tuple:
    """What a field means, without what it learned. A change here is a new definition version."""
    return (f.meaning, f.definition, f.type, f.unit, tuple(f.range or ()), tuple(f.choices))


def load_pack(path=None) -> dict:
    path = path or PACK_PATH
    if not path.exists():
        return {"version": 1, "fields": []}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {"version": data.get("version", 1), "fields": [Field(**f) for f in data.get("fields", [])]}


def save_pack(pack: dict, path=None) -> None:
    path = path or PACK_PATH
    path.parent.mkdir(exist_ok=True)
    data = {"version": pack["version"], "fields": [f.to_dict() for f in pack["fields"]]}
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def get_field(pack: dict, key: str) -> Field | None:
    return next((f for f in pack["fields"] if f.key == key), None)


def put_field(pack: dict, f: Field) -> None:
    """Add a field, or replace it in place (a redefined field keeps its position)."""
    keys = [x.key for x in pack["fields"]]
    if f.key in keys:
        pack["fields"][keys.index(f.key)] = f
    else:
        pack["fields"].append(f)
    pack["version"] += 1


def remove_field(pack: dict, key: str) -> bool:
    before = len(pack["fields"])
    pack["fields"] = [x for x in pack["fields"] if x.key != key]
    if len(pack["fields"]) == before:
        return False
    pack["version"] += 1
    return True
