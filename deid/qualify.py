"""
qualify.py - a name-check model is trusted only after it passes the qualification set, under the
exact settings in use. Any change (provider, URL, model, effort, prompt version) needs a new
qualification, so a hospital swapping models cannot silently lower privacy.

  is_qualified(provider) -> bool
  choose_provider()      -> (provider, warning): the configured one if qualified, else a qualified
                            backup, else the configured one with a "review manually" warning.

Results live in profiles/model_qualification.json (git-ignored; one machine, one deployment).
eval/qualify_model.py runs the test and records the result.
"""
import hashlib
import json
from datetime import datetime, timedelta, timezone

from common import spark
from common.config import ROOT, load_config

QUAL_FILE = ROOT / "profiles" / "model_qualification.json"
MIN_CATCH = 0.9          # share of qualification identifiers the model must catch (local rules miss them)
MAX_FALSE_ALARMS = 1     # clinical words the model may wrongly call identifiers


def qualification_key(provider: str) -> str:
    from deid.names import PROMPT_VERSION

    s = load_config()
    url = {"meta": s.meta_api_base_url, "baseten": s.baseten_base_url}[provider]
    model = {"meta": s.meta_model, "baseten": s.baseten_model}[provider]
    raw = json.dumps([provider, url, model, spark.DEFAULT_EFFORT[provider], PROMPT_VERSION])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load() -> dict:
    return json.loads(QUAL_FILE.read_text(encoding="utf-8")) if QUAL_FILE.exists() else {}


def is_qualified(provider: str) -> bool:
    record = _load().get(qualification_key(provider))
    return bool(record and record["passed"])


def record_result(provider: str, catch: float, false_alarms: int, details: dict) -> dict:
    s = load_config()
    result = {"provider": provider,
              "model": {"meta": s.meta_model, "baseten": s.baseten_model}[provider],
              "effort": spark.DEFAULT_EFFORT[provider],
              "when": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
              "catch": round(catch, 3), "false_alarms": false_alarms,
              "passed": catch >= MIN_CATCH and false_alarms <= MAX_FALSE_ALARMS, **details}
    data = _load()
    data[qualification_key(provider)] = result
    QUAL_FILE.parent.mkdir(exist_ok=True)
    QUAL_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return result


def choose_provider() -> tuple[str, str | None]:
    configured = spark.STEP_PROVIDER["names"]
    if is_qualified(configured):
        return configured, None
    for backup in ("meta", "baseten"):
        if backup != configured and is_qualified(backup):
            return backup, (f"Name-check model '{configured}' is not qualified for these settings; "
                            f"used the qualified backup '{backup}'.")
    return configured, ("Name-check model is not qualified for these settings (run python -m "
                        "eval.qualify_model); review the de-identified text manually.")
