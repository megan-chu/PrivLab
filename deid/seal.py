"""
seal.py - HMAC seal (PRIVLAB_SEAL_KEY) proving masked or de-identified text came out
of deid/ unchanged. The leak-check exemptions (kept types, allowed terms) are sealed
too, so they cannot be widened after the fact.
"""
import hashlib
import hmac
import json

from common.config import load_config


def make_seal(deid_id: str, text: str, kept_types=(), allowed=()) -> str:
    settings = load_config()
    settings.require("seal_key")
    msg = json.dumps([deid_id, text, sorted(kept_types), sorted(allowed)]).encode("utf-8")
    return "hmac-sha256:" + hmac.new(settings.seal_key.encode(), msg, hashlib.sha256).hexdigest()


def check_seal(obj) -> bool:
    expected = make_seal(obj.deid_id, obj.text, getattr(obj, "kept_types", ()), getattr(obj, "allowed", ()))
    return hmac.compare_digest(obj.seal or "", expected)
