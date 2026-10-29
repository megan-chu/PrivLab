"""
export.py - structured-only registry export from confirmed values (PLAN.md §3C).

One row per confirmed field: case id (random, from the encrypted vault), report id, field,
value, unit, outcome and evidence position. Never free text. Every value is scanned for
identifier patterns; a hit is left out and counted.

Usage:
  python -m extract.export --out runs/registry_export.csv
"""
import argparse
import csv
import json
import sys

from deid.patterns import leftover_identifiers
from deid.vault import load_vault
from extract.learn import read_log
from extract.pack import get_field, load_pack

COLUMNS = ["case_id", "report_id", "field", "value", "unit", "outcome", "evidence_start", "evidence_end"]


def export_rows(*, pack_path=None, log_path=None) -> tuple[list[dict], int]:
    """Latest confirmed value per (report, field). Returns (rows, number withheld by the scan)."""
    pack, latest = load_pack(pack_path), {}
    for e in read_log(log_path):
        latest[(e["deid_id"], e["field"])] = e
    rows, withheld = [], 0
    for (deid_id, key), e in latest.items():
        conf = e["confirmed"] or {}
        value = conf.get("value")
        if value is not None and leftover_identifiers(str(value)):
            withheld += 1
            continue
        field = get_field(pack, key)
        try:
            case_id = load_vault(deid_id)["case_id"]
        except (OSError, KeyError, ValueError):
            case_id = None
        rows.append({"case_id": case_id, "report_id": e["report_id"], "field": key, "value": value,
                     "unit": field.unit if field and field.type == "number" else None,
                     "outcome": "not stated" if not conf else "confirmed",
                     "evidence_start": conf.get("start"), "evidence_end": conf.get("end")})
    return rows, withheld


def write_export(path, *, pack_path=None, log_path=None) -> dict:
    rows, withheld = export_rows(pack_path=pack_path, log_path=log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    else:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
    return {"rows": len(rows), "withheld": withheld, "path": str(path)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Export confirmed field values (no free text).")
    ap.add_argument("--out", required=True, help="a .csv or .json path")
    from pathlib import Path
    print(write_export(Path(ap.parse_args().out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
