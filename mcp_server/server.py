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
import csv
import functools
import io
import logging
import os
import re
import sys
import threading
import traceback
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))      # works from any working folder

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from common import spark
from common.config import ROOT
from common.data import manifest
from common.pdf import pdf_text
from deid import pipeline
from deid.hold import _plain, name_list, vocabulary
from extract import learn
from extract.candidates import find_candidates
from extract.define import define_field
from extract.pack import load_pack, remove_field, save_pack
import viewer.server as viewer

log = logging.getLogger("privlab.mcp")
LIMIT = 25
VIEWER = {"url": None}
HideType = Literal["NAME", "DOCTOR", "ID", "DOB", "DATE", "MRN", "ACCESSION", "ACCOUNT", "PHONE", "EMAIL",
                   "ADDRESS", "HOSPITAL", "BANK"]

INSTRUCTIONS = """PrivLab: private report processing on this computer.

WHAT IT DOES
PrivLab removes personal details (names, ID numbers, phones, addresses, card and bank details) from a report on
this computer, then fills in the fields the user chooses (e.g. tumour size), each value with its quote. It only
de-identifies and structures existing reports: no medical advice, no diagnosis.

HOW TO INTRODUCE IT (when the user asks what PrivLab can do, or starts)
"PrivLab can take a report file, hide the personal details on your computer, and fill in the fields you want,
with every value traced to its quote. I only ever see tags like [PERSON-1], never the real names. Which report?"

THE STEPS (follow "next" in every reply, and relay "say" in your own words)
1. Review identifiers: privlab_deidentify_report(path). Tell the user how many personal details were found and
   by type; nothing has been sent yet. Show the viewer link (see VIEWER). Ask which, if any, to keep visible
   and whether anything was missed.
2. De-identify: privlab_confirm_review with the user's answers. Only now is the masked text sent, once, for a
   safety check. If "next" is "hold" or "flags", ask the user and call privlab_confirm_review again.
3. Choose fields: privlab_list_fields. If the user wants a new one, privlab_define_field (draft first, then
   save=True after the user agrees with the definition).
4. Extract: privlab_extract_fields. Show each value with its quote. Ask the user to confirm, correct or mark it
   "not stated"; then privlab_confirm_field. A blind check: ask for the value without showing PrivLab's.
5. Export: privlab_export_file(kind="data") for the CSV, or kind="masked" for the masked text.

PRIVACY RULES (always)
- Pass file paths; never paste or quote report text into the chat.
- Always ask the user before privlab_confirm_review; never answer a review question for them.
- Never ask the user to type a name into the chat (it would reach the assistant). To hide a missed name, send
  them to the viewer.

VIEWER
PrivLab has a visual page running on this computer. The first time you de-identify a report, show its
viewer_url as a clickable link and say what it is: "the same report on your own screen, with every personal
detail highlighted; you can check and correct it there, and the real text never comes into this chat." Offer
it again for extracted values. privlab_open_viewer gives the link any time."""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
TOKEN = re.compile(r"\[[A-Z]+-\d+\]|>>|<<|[^\W_]+")


def scrub(context: str) -> str:
    """Context for the chat (a cloud model): only tags and ordinary report words. Anything else, such as an
    identifier the local rules missed, becomes "..." and any number "#" (red-team round 2, 30 Oct)."""
    vocab, names = vocabulary(), name_list()

    def one(m):
        w = m.group()
        if w.startswith("[") or w in (">>", "<<"):
            return w
        if any(c.isdigit() for c in w):
            return "#"
        return w if _plain(w) in vocab and _plain(w) not in names else "..."
    return TOKEN.sub(one, context)


def _page(items: list, offset: int, limit: int) -> dict:
    offset, limit = max(0, offset), max(1, min(limit, 100))
    more = offset + limit < len(items)
    return {"items": items[offset:offset + limit], "total_count": len(items), "has_more": more,
            "next_offset": offset + limit if more else None}


def _url(deid_id: str = "") -> str | None:
    return VIEWER["url"] and VIEWER["url"] + (f"#{deid_id}" if deid_id else "")


def _see(deid_id: str, what: str) -> str:
    """A sentence pointing to the viewer, for "say": the user learns there is a visual page and where it is."""
    url = _url(deid_id)
    return f" Tell the user they can {what} in the PrivLab viewer, on this computer only: {url}" if url else ""


def _safe(fn):
    """Errors become a plain message the assistant can act on; details go to runs/mcp.log."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except spark.OfflineError:
            return {"next": "offline", "error": spark.OFFLINE,
                    "say": f"{spark.OFFLINE}. De-identification still works on this laptop; the second check and "
                           "fields need Muse Spark. Try again when online."}
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
    items = [{**i, "context": scrub(i.get("context", ""))} for i in items]
    page = _page(items, offset, limit)
    say = {"review": f"{page['total_count']} identifiers found on this laptop; nothing has been sent. Ask the user "
                     "which ids to keep visible (false alarms) and whether anything was missed, then call "
                     "privlab_confirm_review." + _see(did, "check them highlighted on your own screen"),
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
    with viewer._report_lock(deid_id):
        step = run["step"]                      # read under the lock: the viewer may have moved the report on
        if step in ("hold", "flags") and [t for t in hide_text if t.strip()]:
            msg = ("Missed text can only be added at the review step. Nothing was changed. To hide more, open the "
                   "report in the viewer (privlab_open_viewer) and select the text there.")
            return {"deid_id": deid_id, "next": step, "error": msg, "say": msg}
        if step == "review":
            out = pipeline.confirm_deid(deid_id, unmark=keep_ids, expect="review",
                                        add=[{"text": t, "type": hide_type} for t in hide_text if t.strip()])
        elif step == "hold":
            if hide_ids:
                msg = ("At the hold question every word is hidden unless you keep it: send keep_ids only. "
                       "Nothing was changed.")
                return {"deid_id": deid_id, "next": "hold", "error": msg, "say": msg}
            out = pipeline.confirm_deid(deid_id, hold_answers={"keep": keep_ids}, expect="hold")
        elif step == "flags":
            out = pipeline.confirm_deid_flags(deid_id, keep=keep_ids, hide=hide_ids)
        else:
            return {"deid_id": deid_id, "next": "done", "say": "This report is already de-identified. Next: "
                    "privlab_extract_fields or privlab_export_file."}
    return _step(out)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=True))
@_safe
def privlab_define_field(name: str, description: str = "", save: bool = False, meaning: str = "",
                         definition: str = "", type: Literal["number", "text"] | None = None, unit: str = "",
                         range_min: float | None = None,
                         range_max: float | None = None, multi: bool = False) -> dict:
    """Use when the user wants to collect a field (e.g. "tumour size"). Only the name and the user's words are sent.
    Arguments: save=False returns a draft to confirm ("Did you mean ...?"); save=True saves it, using the
    draft's details or the user's changes (meaning, definition, type, unit, range). Text values are kept as written.
    multi=True makes it a list (e.g. biomarker results): every item the report states, exported one row per item.
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
              "unit": unit or None, "range_min": range_min, "range_max": range_max, "multi": multi}
    with viewer._WRITE:
        out = define_field(name, description or None, accept=accept, pack_path=viewer.STATE["pack"])
    f = out["field"]
    return {"next": "done", "field": {k: f[k] for k in ("key", "name", "meaning", "definition", "type", "unit",
                                                        "range", "multi", "version")},
            "say": f"Saved '{name}' (definition v{f['version']}). It will be filled on every report."}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
@_safe
def privlab_list_fields(offset: int = 0, limit: int = LIMIT) -> dict:
    """Use when the user asks what PrivLab collects, or how accurate a field is.
    Returns: each field's definition, version, readiness and accuracy (95% bound on wrong verified values)."""
    keep = ("key", "name", "meaning", "definition", "type", "unit", "range", "version", "readiness",
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
                   + _see(deid_id, "see every value highlighted in the report")
                   + (f" {blind} blind check(s): ask the user for the value without showing PrivLab's." if blind else "")}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=False))
@_safe
def privlab_confirm_field(deid_id: str, field: str, candidate_id: str = "", value: str = "",
                          evidence_text: str = "", not_stated: bool = False, note: str = "",
                          items: list[str] = []) -> dict:
    """Use when the user confirms or corrects one field's value. Scored first, then PrivLab learns.
    Arguments: candidate_id to accept a candidate; or value + evidence_text (exact words from the de-identified
    text) to correct it; or not_stated=True; note = optional reason PrivLab's pick was wrong.
    For a list field: items = every item as written in the text (the whole list), or not_stated=True.
    Returns: saved, outcome, readiness. A blind check reveals PrivLab's answer only now."""
    with viewer._WRITE:
        out = learn.confirm_field(deid_id, field, candidate_id=candidate_id or None, value=value or None,
                                  evidence_text=evidence_text or None, missing=not_stated, note=note or None,
                                  items=[{"value": x, "evidence_text": x} for x in items] or None,
                                  pack_path=viewer.STATE["pack"], log_path=viewer.STATE["log"])
    if out.get("status") != "saved":
        return {"next": "error", "error": out.get("reason"), "say": out.get("reason")}
    say = f"Saved {field}: {out['value'] if out['value'] is not None else 'not stated'} ({out['outcome']})."
    if out.get("blind"):
        say += f" Blind check: PrivLab had {out['predicted'] if out['predicted'] is not None else 'nothing'}."
    return {"next": "done", **{k: out[k] for k in ("outcome", "value", "readiness")}, "say": say}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=False))
@_safe
def privlab_export_file(kind: Literal["data", "masked"], out_path: str) -> dict:
    """Use when the user wants a file, for the reports of this session. data = a CSV with one row per report and
    one column per field
    (values as written, no report text);
    masked = every finished report's de-identified text as a .zip. Written on this laptop.
    Returns: the file path and how much it holds (never the contents)."""
    p = Path(out_path).expanduser().resolve()
    if ROOT / "data" in p.parents:
        return {"next": "error", "error": "Not inside data/.", "say": "Choose a folder outside PrivLab's data/ folder."}
    p.parent.mkdir(parents=True, exist_ok=True)
    session = list(pipeline._RUNS)
    if kind == "data":
        body = viewer.export_csv(session)
        p.write_bytes(body)
        rows = max(0, len(list(csv.reader(io.StringIO(body.decode("utf-8-sig"))))) - 1)   # a quoted value may hold a line break
        return {"next": "done", "path": str(p), "rows": rows,
                "say": f"Wrote {rows} report row(s), one column per field, to {p}."}
    out = viewer.deid_download(None, session)
    if out is None:
        return {"next": "error", "error": "Nothing finished.", "say": "No de-identified reports to export yet."}
    p.write_bytes(out[0])
    n = len(viewer._finished(None, session))
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


@mcp.prompt(title="Fill my fields")
def fill_fields(path: str) -> str:
    return (f"Make {path} safe with PrivLab, then fill my fields: privlab_deidentify_report, review with me, "
            "privlab_confirm_review until done; privlab_extract_fields; for each field show the value with its quote "
            "and ask me to confirm, correct or mark not stated (privlab_confirm_field; for a blind check, ask me first "
            "without showing PrivLab's value); then privlab_export_file(kind='data').")


@mcp.prompt(title="Set up my fields")
def set_up_fields(fields: str) -> str:
    return (f"Help me define these fields in PrivLab: {fields}. For each: privlab_define_field(save=False), show me "
            "the 'Did you mean' question, definition and type; change anything I ask; then save=True. "
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
