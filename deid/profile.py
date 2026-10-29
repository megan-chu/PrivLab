"""
profile.py - the user's privacy profile (profiles/default.yaml, git-ignored).

First use returns the checklist; the answers become the profile. Corrections are
learned: unmarked false alarms -> allow_list; marked or Muse Spark-found identifiers
the user keeps hidden -> always_remove, which Layer 1 matches next time. The
always_remove list holds real names, so it is stored encrypted.
"""
import yaml

from common.config import ROOT
from deid.types import CHECKLIST, PHI_TYPES, PrivacyProfile
from deid.vault import decrypt_json, encrypt_json

PROFILE_PATH = ROOT / "profiles" / "default.yaml"


def checklist() -> list[dict]:
    return [{"key": k, "label": label, "default": default, "warn_if_unticked": warn}
            for k, (label, _, warn, default) in CHECKLIST.items()]


def new_profile(choices: dict | None = None) -> tuple[PrivacyProfile, list[str]]:
    profile = PrivacyProfile()
    for key, value in (choices or {}).items():
        if key not in CHECKLIST:
            raise ValueError(f"unknown checklist item {key!r}")
        profile.remove[key] = bool(value)
    warnings = [f"Not removing '{CHECKLIST[k][0]}' leaves identifying details in the text."
                for k, on in profile.remove.items() if not on and CHECKLIST[k][2]]
    return profile, warnings


def removed_types(profile: PrivacyProfile) -> set[str]:
    return {t for key, on in profile.remove.items() if on for t in CHECKLIST[key][1]}


def kept_types(profile: PrivacyProfile) -> tuple[str, ...]:
    return tuple(sorted(set(PHI_TYPES) - removed_types(profile)))


def reusable_allow_list(profile: PrivacyProfile) -> list[str]:
    """False alarms to skip in later reports. Single words are not reused: keeping "Barrett"
    once (Barrett oesophagus) must not unmask "Mr Barrett" in another report."""
    return [a for a in profile.allow_list if len(a.split()) >= 2]


def learn(profile: PrivacyProfile, allow=(), remove=()) -> None:
    """allow: false-alarm texts; remove: {"text", "type"} to catch locally next time."""
    allowed = {a.lower() for a in profile.allow_list}
    for text in allow:
        if text.lower() not in allowed:
            profile.allow_list.append(text)
            allowed.add(text.lower())
    known = {(r["text"].lower(), r["type"]) for r in profile.always_remove}
    for r in remove:
        if (r["text"].lower(), r["type"]) not in known:
            profile.always_remove.append({"text": r["text"], "type": r["type"]})
            known.add((r["text"].lower(), r["type"]))
    if allow or remove:
        profile.version += 1


def load_profile(path=None) -> PrivacyProfile | None:
    path = path or PROFILE_PATH
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    token = data.pop("always_remove_encrypted", "")
    return PrivacyProfile(**data, always_remove=decrypt_json(token.encode()) if token else [])


def save_profile(profile: PrivacyProfile, path=None) -> None:
    path = path or PROFILE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": profile.version, "remove": profile.remove, "allow_list": profile.allow_list,
            "always_remove_encrypted": encrypt_json(profile.always_remove).decode()}
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
