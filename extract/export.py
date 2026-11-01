"""
export.py - the extracted-data export from confirmed values (PLAN.md §3C): one row per report, one column per
field, ready for storage and analysis (Excel, pandas, R) without reshaping.

Columns: case_id (random, from the encrypted vault), report_id, reviewed_at, exported_at, then one column per field in pack
order, headed by the field's name (number fields add their unit: "tumour size (cm)"). Text values are exactly as
the report writes them; numbers are numbers. An empty cell means not stated (or not reviewed yet). Never free
text, never evidence positions. Every value is scanned for identifier patterns; a hit is left empty and counted.

Usage:
  python -m extract.export --out runs/extracted_data.csv      # or .json; every report in the score log
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timedelta, timezone

from deid.patterns import leftover_identifiers
from deid.vault import load_vault
from extract.learn import read_log
from extract.pack import load_pack

FIXED = ["case_id", "report_id", "reviewed_at", "exported_at"]
SGT = timezone(timedelta(hours=8))


def header(field) -> str:
    return f"{field.name} ({field.unit})" if field.type == "number" and field.unit else field.name


def headers(fields) -> dict[str, str]:
    """Field key -> column header. Two fields with the same name (and unit) get their key appended, so
    neither overwrites the other's column."""
    plain = [header(f) for f in fields]
    return {f.key: h if plain.count(h) == 1 else f"{h} [{f.key}]" for f, h in zip(fields, plain)}


def export_rows(*, pack_path=None, log_path=None, deid_ids=None) -> tuple[list[str], list[dict], int]:
    """(columns, rows, number of values withheld by the scan): the latest confirmed value per report and field.
    deid_ids: only these reports (the viewer passes the reports in its list); None = every report in the log."""
    fields = [f for f in load_pack(pack_path)["fields"] if not f.multi]     # list fields: export_items
    column = headers(fields)
    columns = FIXED + [column[f.key] for f in fields]
    by_report, withheld = {}, 0
    exported_at = datetime.now(SGT).isoformat(timespec="seconds")
    wanted = None if deid_ids is None else set(deid_ids)
    for e in read_log(log_path):
        if wanted is not None and e["deid_id"] not in wanted:
            continue
        field = next((f for f in fields if f.key == e["field"]), None)
        if field is None:                                   # a removed field, or a list field
            continue
        row = by_report.setdefault(e["deid_id"], {"report_id": e["report_id"]})
        row["reviewed_at"] = e["ts"]
        value = (e["confirmed"] or {}).get("value")         # None: not stated
        if value is not None and leftover_identifiers(str(value)):
            withheld += 1
            value = None
        row[column[field.key]] = value
    rows = []
    for deid_id, row in by_report.items():
        try:
            case_id = load_vault(deid_id)["case_id"]
        except (OSError, KeyError, ValueError):
            case_id = None
        rows.append({c: ({**row, "case_id": case_id, "exported_at": exported_at}).get(c) for c in columns})
    return columns, rows, withheld


def split_item(item: str) -> tuple[str, str]:
    """"TP53: rearrangement exon 11" -> ("TP53", "rearrangement exon 11"); no label -> (item, "")."""
    name, sep, result = item.partition(":")
    return (name.strip(), result.strip()) if sep and name.strip() else (item.strip(), "")


def export_items(*, pack_path=None, log_path=None, deid_ids=None) -> dict[str, tuple[list[str], list[dict], int]]:
    """One long table per list field, key -> (columns, rows, withheld): a row per item, so a report with eight
    biomarkers gives eight rows. Columns: case_id, report_id, <field name> (the item's label, as written),
    result (after the colon). Joins extracted_data.csv on case_id."""
    pack = load_pack(pack_path)
    wanted = None if deid_ids is None else set(deid_ids)
    latest = {}
    for e in read_log(log_path):
        if wanted is None or e["deid_id"] in wanted:
            latest[(e["deid_id"], e["field"])] = e              # the last confirmation per report and field
    out = {}
    for f in [f for f in pack["fields"] if f.multi]:
        columns, rows, withheld = ["case_id", "report_id", f.name, "result"], [], 0
        for (deid_id, key), e in latest.items():
            if key != f.key:
                continue
            try:
                case_id = load_vault(deid_id)["case_id"]
            except (OSError, KeyError, ValueError):
                case_id = None
            for value in (e["confirmed"] or {}).get("value") or []:
                if leftover_identifiers(str(value)):
                    withheld += 1
                    continue
                name, result = split_item(str(value))
                rows.append({"case_id": case_id, "report_id": e["report_id"], f.name: name, "result": result})
        out[f.key] = (columns, rows, withheld)
    return out


def write_export(path, *, pack_path=None, log_path=None) -> dict:
    columns, rows, withheld = export_rows(pack_path=pack_path, log_path=log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    else:
        with open(path, "w", encoding="utf-8-sig", newline="") as f:      # -sig: Excel reads UTF-8 correctly
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            w.writerows(rows)
    return {"rows": len(rows), "withheld": withheld, "path": str(path)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Export confirmed field values: one row per report (no free text).")
    ap.add_argument("--out", required=True, help="a .csv or .json path")
    from pathlib import Path
    print(write_export(Path(ap.parse_args().out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
