"""
data.py - read the synthetic dataset in data/.

Offsets in data/truth are character offsets into LF text, so reports are read
with newline="" (no translation) and rejected if they contain CR.
"""
import csv
import json

from common.config import ROOT

DATA = ROOT / "data"
REPORTS = DATA / "reports"
PHI_TRUTH = DATA / "truth" / "phi"
MANIFEST = DATA / "manifest.csv"
SPLITS = ("dev", "test", "extra")


def manifest(split: str | None = None) -> list[dict]:
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    with open(MANIFEST, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if split is None or r["split"] == split]


def load_report(report_id: str) -> str:
    with open(REPORTS / f"{report_id}.txt", encoding="utf-8", newline="") as f:
        text = f.read()
    if "\r" in text:
        raise ValueError(
            f"{report_id}: CR line endings found; offsets will be wrong. "
            "Check .gitattributes (data/** -text) and re-checkout data/."
        )
    return text


def load_phi_truth(report_id: str) -> dict:
    return json.loads((PHI_TRUTH / f"{report_id}.json").read_text(encoding="utf-8"))
