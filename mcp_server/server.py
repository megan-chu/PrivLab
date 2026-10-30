"""
server.py - PrivLab as an MCP server (stdio): 10 privlab_* tools, 3 playbook prompts, and the local
viewer started alongside in the same process, so chat and viewer share the same reports.

A thin wrapper over the functions the viewer calls (deid/pipeline.py, extract/). Tools take a file
path or a deid_id, never report text. Responses carry ids, types and masked context only: no raw
identifier and no raw report text reaches the chat (the de-identified text does, once confirmed).
Every response says what to do next ("next", "say"); errors are plain sentences, never exceptions.
stdout is the MCP protocol: nothing else is ever printed there; logs go to runs/mcp.log.

Settings (environment): PRIVLAB_VIEWER_PORT (default 8765; the next free port is used if taken),
PRIVLAB_MCP_VIEWER=off to not start the viewer.

Usage:
  python -m mcp_server.server                                  # from the repo root
  C:/Users/User/GitHub/PrivLab/.venv/Scripts/python.exe C:/Users/User/GitHub/PrivLab/mcp_server/server.py
"""
import contextlib
import functools
import logging
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))      # works from any working folder

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from common.config import ROOT
from common.data import manifest
from common.pdf import pdf_text
from deid import pipeline
from extract import learn
from extract.candidates import find_candidates
from extract.define import define_field
from extract.pack import load_pack, remove_field, save_pack
import viewer.server as viewer

log = logging.getLogger("privlab.mcp")
LIMIT = 25
VIEWER = {"url": None}
HideType = Literal["NAME", "DOCTOR", "ID", "DOB", "DATE", "MRN", "ACCESSION", "ACCOUNT", "PHONE", "EMAIL",
                   "ADDRESS", "HOSPITAL"]

INSTRUCTIONS = """PrivLab de-identifies clinical reports on this laptop, then extracts the fields the user defines.
Order: privlab_deidentify_report -> privlab_confirm_review (repeat while next is "hold" or "flags") ->
privlab_extract_fields -> privlab_confirm_field for each field -> privlab_export_file.
Privacy rules: never paste or quote report text into the chat; pass file paths. Nothing is sent to any
model until the user confirms the review, and then only masked text. Always ask the user before
privlab_confirm_review; never answer a review, hold or second-check question for them. For reviewing
highlights or hiding a missed name, offer privlab_open_viewer: typing a name into the chat sends it to
the assistant. Follow "next" and relay "say" in every response."""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _page(items: list, offset: int, limit: int) -> dict:
    offset, limit = max(0, offset), max(1, min(limit, 100))
    more = offset + limit < len(items)
    return {"items": items[offset:offset + limit], "total_count": len(items), "has_more": more,
            "next_offset": offset + limit if more else None}


def _url(deid_id: str = "") -> str | None:
    return VIEWER["url"] and VIEWER["url"] + (f"#{deid_id}" if deid_id else "")


def _safe(fn):
    """Errors become a plain message the assistant can act on; details go to runs/mcp.log."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            log.error("%s failed: %s", fn.__name__, traceback.format_exc())
            return {"next": "error", "error": f"{type(e).__name__}: something went wrong; see runs/mcp.log",
                    "say": "That step failed on the laptop. Nothing was sent. Try again, or use the viewer."}
    return wrapper


def _read(path: str) -> tuple[str | None, str, list[str]]:
    """(text, report id or error, warnings). .txt or a PDF with selectable text; test reports refused."""
    p = Path(path).expanduser()
    if not p.is_file():
        return None, f"No file at {p}. Give the full path to a .txt or .pdf report.", []
    if p.suffix.lower() not in (".txt", ".pdf"):
        return None, ("PrivLab reads .txt files and PDFs with selectable text. Scanned PDFs, images and other "
                      "formats are not supported."), []
    if any(r["report_id"] == p.stem and r["split"] == "test" for r in manifest()):
        return None, "That name belongs to a held-out test report; it is never opened here.", []
    warnings = []
    if p.suffix.lower() == ".pdf":
        out = pdf_text(p.read_bytes())
        if "error" in out:
            return None, out["error"], []
        text, warnings = out["text"], out["warnings"]
    else:
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return None, "The file is not UTF-8 text. Save it as plain text (UTF-8) and try again.", []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return None, "The report is empty.", []
    if len(text) > viewer.MAX_UPLOAD:
        return None, f"The report is too long ({len(text):,} characters; limit {viewer.MAX_UPLOAD:,}).", []
    return text, p.stem, warnings


def _step(out: dict, offset: int = 0, limit: int = LIMIT) -> dict:
    """A pipeline response, shaped for the chat: paginated items, next, say, viewer link."""
    did, nxt = out.get("deid_id", ""), out.get("next")
    if nxt == "error" or "error" in out:
        return {"deid_id": did, "next": "error", "error": out.get("error"), "say": out.get("error")}
    base = {"deid_id": did, "next": nxt, "warnings": out.get("warnings", []), "viewer_url": _url(did)}
    if nxt == "done":
        reasons = (out.get("review_priority") or {}).get("reasons", [])
        if any("left in the output" in r for r in reasons):
            return {**base, "next": "blocked", "say": "A possible identifier is still in the output, so the text is "
                    "withheld and nothing more is sent for this report. Open the viewer and de-identify it again."}
        s = out["summary"]
        return {**base, "text": out["text"], "summary": s, "second_check": out.get("layer2"),
                "review_priority": out.get("review_priority"),
                "say": f"De-identified on this laptop: {s['total']} identifiers replaced with tags. Next: "
                       "privlab_extract_fields, or privlab_export_file(kind='masked')."}
    items = out.get("items") or out.get("detections") or out.get("hold") or out.get("flags") or []
    page = _page(items, offset, limit)
    say = {"review": f"{page['total_count']} identifiers found on this laptop; nothing has been sent. Ask the user "
                     "which ids to keep visible (false alarms) and whether anything was missed, then call "
                     "privlab_confirm_review. The viewer shows them highlighted.",
           "hold": "These words still look like names. They will all be hidden unless the user says some are not "
                   "names (keep_ids). Ask, then call privlab_confirm_review. Nothing has been sent yet.",
           "flags": "The second check (on masked text only) found more. Ticked items will be hidden; ask the user "
                    "which are false alarms (keep_ids) and which unticked ones to hide (hide_ids)."}.get(nxt, "")
    return {**base, "counts": out.get("counts"), **page, "say": say}


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
mcp = MCPServer("privlab", instructions=INSTRUCTIONS)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=False))
@_safe
def privlab_deidentify_report(path: str, limit: int = LIMIT) -> dict:
    """Use when the user wants a report made safe. Finds identifiers on this laptop; nothing is sent.
    Arguments: path to a .txt file or a PDF with selectable text (never paste report text); limit = page size.
    Returns: deid_id, detections (id, type, masked context), counts. Next: ask the user, then privlab_confirm_review."""
    text, rid, warnings = _read(path)
    if text is None:
        return {"next": "error", "error": rid, "say": rid}
    with viewer._WRITE:                     # first use creates the default privacy profile
        out = pipeline.deidentify_report(text=text, report_id=rid, profile_choices={})
    out["warnings"] = out.get("warnings", []) + warnings
    return _step(out, 0, limit)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
@_safe
def privlab_get_report(deid_id: str, offset: int = 0, limit: int = LIMIT) -> dict:
    """Use when you need where a report is up to, or the next page of its items, or to resume after a pause.
    Arguments: deid_id; offset and limit for paging (see next_offset).
    Returns: next (review, hold, flags, done), the items still to answer, or the de-identified text when done."""
    return _step(pipeline.report_state(deid_id), offset, limit)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=True))
@_safe
def privlab_confirm_review(deid_id: str, keep_ids: list[str] = [], hide_ids: list[str] = [],
                           hide_text: list[str] = [], hide_type: HideType = "NAME") -> dict:
    """Use when the user has answered the current question for a report (review, hold or second check).
    Arguments: keep_ids = ids to leave visible (false alarms); hide_ids = unticked second-check ids to hide;
    hide_text = missed text to hide at the review step, typed by the user (this puts it in the chat:
    prefer privlab_open_viewer for names); hide_type = its type. Ask the user first; never answer for them.
    Returns: the next question, or the de-identified text. The review step then runs the second check on
    masked text only."""
    run = pipeline._RUNS.get(deid_id)
    if run is None:
        return {"next": "error", "error": "Unknown deid_id.", "say": "Unknown report: de-identify it first."}
    step = run["step"]
    with viewer._report_lock(deid_id):
        if step == "review":
            out = pipeline.confirm_deid(deid_id, unmark=keep_ids,
                                        add=[{"text": t, "type": hide_type} for t in hide_text if t.strip()])
        elif step == "hold":
            out = pipeline.confirm_deid(deid_id, hold_answers={"keep": keep_ids})
        elif step == "flags":
            out = pipeline.confirm_deid_flags(deid_id, keep=keep_ids, hide=hide_ids)
        else:
            return {"deid_id": deid_id, "next": "done", "say": "This report is already de-identified. Next: "
                    "privlab_extract_fields or privlab_export_file."}
    return _step(out)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=True))
@_safe
def privlab_define_field(name: str, description: str = "", save: bool = False, meaning: str = "",
                         definition: str = "", type: Literal["number", "choice"] | None = None, unit: str = "",
                         choices: list[str] = [], range_min: float | None = None,
                         range_max: float | None = None) -> dict:
    """Use when the user wants to collect a field (e.g. "tumour size"). Only the name and the user's words are sent.
    Arguments: save=False returns a draft to confirm ("Did you mean ...?"); save=True saves it, using the
    draft's details or the user's changes (meaning, definition, type, unit, choices, range).
    Returns: the draft, or the saved field. Next: privlab_extract_fields."""
    if not save:
        out = define_field(name, description or None, pack_path=viewer.STATE["pack"])
        d = out["draft"]
        return {"next": "confirm", "draft": d,
                "say": f"Did you mean: {d['meaning_question']} {d.get('definition', '')} Ask the user; then call again "
                       "with save=True and these details (changed if the user wants)."}
    if not meaning or type is None:
        msg = "To save, give meaning and type (from the draft: call with save=False first)."
        return {"next": "error", "error": msg, "say": msg}
    accept = {"key": name, "meaning_question": meaning, "definition": definition, "type": type,
              "unit": unit or None, "range_min": range_min, "range_max": range_max, "choices": choices}
    with viewer._WRITE:
        out = define_field(name, description or None, accept=accept, pack_path=viewer.STATE["pack"])
    f = out["field"]
    return {"next": "done", "field": {k: f[k] for k in ("key", "name", "meaning", "definition", "type", "unit",
                                                        "range", "choices", "version")},
            "say": f"Saved '{name}' (definition v{f['version']}). It will be filled on every report."}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
@_safe
def privlab_list_fields(offset: int = 0, limit: int = LIMIT) -> dict:
    """Use when the user asks what PrivLab collects, or how accurate a field is.
    Returns: each field's definition, version, readiness and accuracy (95% bound on wrong verified values)."""
    keep = ("key", "name", "meaning", "definition", "type", "unit", "range", "choices", "version", "readiness",
            "validation")
    fields = [{k: f[k] for k in keep} for f in viewer.api_pack(None)["fields"]]
    return {"next": "done", **_page(fields, offset, limit),
            "say": f"{len(fields)} field(s) defined." if fields else "No fields yet: use privlab_define_field."}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=False))
@_safe
def privlab_remove_field(field: str) -> dict:
    """Use only when the user asks to stop collecting a field. Confirm with the user first.
    Arguments: field = its key (from privlab_list_fields). Returns: done, or an error if there is no such field."""
    with viewer._WRITE:
        pack = load_pack(viewer.STATE["pack"])
        if not remove_field(pack, field):
            return {"next": "error", "error": "No such field.", "say": "That field is not in the pack."}
        save_pack(pack, viewer.STATE["pack"])
    return {"next": "done", "say": f"Removed '{field}'. Its past confirmations stay in the score log."}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=True))
@_safe
def privlab_extract_fields(deid_id: str) -> dict:
    """Use when a report is de-identified and the user wants their fields filled.
    Reads the de-identified text only (one model call for all fields, cached).
    Returns per field: status (green = verified, amber = needs review, missing, blind), value, quote, reason,
    candidates. Next: show each to the user, then privlab_confirm_field."""
    run = pipeline._RUNS.get(deid_id)
    if run is None or run["step"] != "done":
        return {"next": "error", "error": "Not de-identified yet.",
                "say": "Finish de-identifying this report first (privlab_confirm_review)."}
    if any("left in the output" in r for r in (run.get("priority") or {}).get("reasons", [])):
        return {"next": "blocked", "say": "A possible identifier is left in this report, so nothing is extracted."}
    fields = load_pack(viewer.STATE["pack"])["fields"]
    if not fields:
        return {"next": "define", "say": "No fields defined yet: use privlab_define_field."}
    results = []
    for f in fields:
        r = find_candidates(deid_id, f.key, pack_path=viewer.STATE["pack"], blind_checks=True)
        results.append({"field": f.key, "name": f.name, "status": r.get("status"), "value": r.get("value"),
                        "unit": r.get("unit"), "quote": r.get("evidence_text"), "section": r.get("section"),
                        "reason": r.get("reason"), "candidate_id": r.get("candidate_id"),
                        "candidates": [{"id": c["id"], "value": c["value"], "quote": c["evidence_text"],
                                        "failed_check": c["fail"]} for c in r.get("candidates", [])]})
    blind = sum(r["status"] == "blind" for r in results)
    return {"deid_id": deid_id, "next": "confirm", "fields": results, "viewer_url": _url(deid_id),
            "say": "Show each value with its quote. Confirm, correct or mark not stated with privlab_confirm_field."
                   + (f" {blind} blind check(s): ask the user for the value without showing PrivLab's." if blind else "")}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=False))
@_safe
def privlab_confirm_field(deid_id: str, field: str, candidate_id: str = "", value: str = "",
                          evidence_text: str = "", not_stated: bool = False, note: str = "") -> dict:
    """Use when the user confirms or corrects one field's value. Scored first, then PrivLab learns.
    Arguments: candidate_id to accept a candidate; or value + evidence_text (exact words from the de-identified
    text) to correct it; or not_stated=True; note = optional reason PrivLab's pick was wrong.
    Returns: saved, outcome, readiness. A blind check reveals PrivLab's answer only now."""
    with viewer._WRITE:
        out = learn.confirm_field(deid_id, field, candidate_id=candidate_id or None, value=value or None,
                                  evidence_text=evidence_text or None, missing=not_stated, note=note or None,
                                  pack_path=viewer.STATE["pack"], log_path=viewer.STATE["log"])
    if out.get("status") != "saved":
        return {"next": "error", "error": out.get("reason"), "say": out.get("reason")}
    say = f"Saved {field}: {out['value'] if out['value'] is not None else 'not stated'} ({out['outcome']})."
    if out.get("blind"):
        say += f" Blind check: PrivLab had {out['predicted'] if out['predicted'] is not None else 'nothing'}."
    return {"next": "done", **{k: out[k] for k in ("outcome", "value", "readiness")}, "say": say}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=False))
@_safe
def privlab_export_file(kind: Literal["registry", "masked"], out_path: str) -> dict:
    """Use when the user wants a file. registry = structured fields as CSV (no report text);
    masked = every finished report's de-identified text as a .zip. Written on this laptop.
    Returns: the file path and how much it holds (never the contents)."""
    p = Path(out_path).expanduser().resolve()
    if ROOT / "data" in p.parents:
        return {"next": "error", "error": "Not inside data/.", "say": "Choose a folder outside PrivLab's data/ folder."}
    p.parent.mkdir(parents=True, exist_ok=True)
    if kind == "registry":
        body = viewer.export_csv()
        p.write_bytes(body)
        rows = body.decode("utf-8").count("\n") - 1
        return {"next": "done", "path": str(p), "rows": rows, "say": f"Wrote {rows} registry rows to {p}."}
    out = viewer.deid_download(None)
    if out is None:
        return {"next": "error", "error": "Nothing finished.", "say": "No de-identified reports to export yet."}
    p.write_bytes(out[0])
    n = len(viewer._finished())
    return {"next": "done", "path": str(p), "reports": n, "say": f"Wrote {n} masked report(s) to {p}."}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
@_safe
def privlab_open_viewer(deid_id: str = "") -> dict:
    """Use when the user wants to see highlights, review identifiers, hide a missed name or check values.
    The viewer runs on this laptop and shows the report itself; nothing in it reaches the chat.
    Returns: a link to open (to one report if deid_id is given)."""
    if not VIEWER["url"]:
        return {"next": "error", "error": "Viewer off.", "say": "The viewer is not running in this session."}
    return {"next": "open", "viewer_url": _url(deid_id), "say": f"Open {_url(deid_id)} in your browser."}


# --------------------------------------------------------------------------
# Prompts: the playbook
# --------------------------------------------------------------------------
@mcp.prompt(title="De-identify a batch of reports")
def deidentify_batch(folder: str) -> str:
    return (f"De-identify every .txt or .pdf report in {folder} with PrivLab. For each file: "
            "privlab_deidentify_report(path); show me the counts and offer the viewer link; ask me which ids to keep "
            "visible and whether anything was missed; then privlab_confirm_review with my answers; repeat while next "
            "is hold or flags, asking me each time. Never paste report text into the chat. At the end, "
            "privlab_export_file(kind='masked') to a path I choose.")


@mcp.prompt(title="Fill registry fields")
def fill_registry_fields(path: str) -> str:
    return (f"Make {path} safe with PrivLab, then fill my registry fields: privlab_deidentify_report, review with me, "
            "privlab_confirm_review until done; privlab_extract_fields; for each field show the value with its quote "
            "and ask me to confirm, correct or mark not stated (privlab_confirm_field; for a blind check, ask me first "
            "without showing PrivLab's value); then privlab_export_file(kind='registry').")


@mcp.prompt(title="Set up my fields")
def set_up_fields(fields: str) -> str:
    return (f"Help me define these fields in PrivLab: {fields}. For each: privlab_define_field(save=False), show me "
            "the 'Did you mean' question, definition, type and choices; change anything I ask; then save=True. "
            "Finish with privlab_list_fields.")


# --------------------------------------------------------------------------
# Start-up
# --------------------------------------------------------------------------
def start_viewer() -> None:
    """The viewer in a background thread of this process (same reports as the chat)."""
    if os.environ.get("PRIVLAB_MCP_VIEWER", "").lower() == "off":
        return
    first = int(os.environ.get("PRIVLAB_VIEWER_PORT", "8765"))
    for port in range(first, first + 10):
        try:
            server = viewer.make_server(port)
        except OSError:
            continue
        threading.Thread(target=server.serve_forever, daemon=True).start()
        VIEWER["url"] = f"http://127.0.0.1:{port}/"
        log.info("viewer at %s", VIEWER["url"])
        return
    log.warning("no free port for the viewer (%s-%s)", first, first + 9)


def main() -> None:
    (ROOT / "runs").mkdir(exist_ok=True)
    logging.basicConfig(filename=ROOT / "runs" / "mcp.log", level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    with contextlib.redirect_stdout(sys.stderr):        # stdout is the protocol: no stray prints
        pipeline.warm_up()
        start_viewer()
    mcp.run("stdio")


if __name__ == "__main__":
    main()
