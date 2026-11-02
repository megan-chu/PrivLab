"""
server.py - PrivLab as an MCP server (stdio): 5 privlab_* tools, one per step of the flow (deidentify,
review, fields, extract, export), 4 playbook prompts, and the local viewer started alongside in the same
process, so chat and viewer share the same reports.

A thin wrapper over the functions the viewer calls (deid/pipeline.py, extract/). Tools take a file
path or a deid_id, never report text. Responses carry ids, types and masked context only: no raw
identifier and no raw report text reaches the chat (the de-identified text does, once confirmed).
Every response says what to do next ("next", "say"); errors are plain sentences, never exceptions.
stdout is the MCP protocol: nothing else is ever printed there; logs go to runs/mcp.log.

Settings (environment): PRIVLAB_VIEWER_PORT (default 8765; the next free port is used if taken),
PRIVLAB_MCP_VIEWER=off to not start the viewer, PRIVLAB_MCP_OPEN=off to not open the browser on each new report.

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
import webbrowser
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))      # works from any working folder

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field
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
The user can also work on the same reports in the viewer, so before you act on a report you have not touched for a
while, or when the user says they did something in the browser, call privlab_deidentify(deid_id) to see where it
is up to now, and continue from there.
1. Review identifiers: privlab_deidentify(path). Tell the user how many personal details were found and
   by type; nothing has been sent yet. Show the viewer link (see VIEWER). Ask which, if any, to keep visible
   and whether anything was missed.
2. De-identify: privlab_review with the user's answers. Only now is the masked text sent, once, for a
   safety check. If "next" is "hold" or "flags", ask the user and call privlab_review again.
3. Choose fields: privlab_fields(action="list"). For a new one, ask the user for its name, what it means in
   their words and one example value as a report writes it; action="draft" with all three; show the "Did you
   mean" definition; action="save" once the user agrees. The save previews the field on one report: show the
   value and its quote and ask if that is what they want (if not, draft again with their correction).
4. Extract: privlab_extract(deid_id). Show each value with its quote. Ask the user to confirm, correct or mark
   it "not stated", then send all the answers in one privlab_extract(deid_id, answers) call. A blind check: ask
   for the value without showing PrivLab's.
5. Export: privlab_export(kind="data") for the CSV, or kind="masked" for the masked text.

A FOLDER OF REPORTS (bulk)
1. If the fields are new, teach them first on 1-3 reports (in the browser, or one report at a time here).
2. privlab_deidentify(folder): every report is read and its personal details hidden on this computer. Show the
   summary; ask whether to review any (all are listed in the browser) or to hide them all; privlab_review each.
3. Ask: "Do you want to check the masked text in your browser before I extract?"
4. privlab_extract(deid_id, auto_accept_verified=True) for each report: verified values are saved by themselves;
   show the user only the ones that need them, then send their answers.
5. privlab_export(kind="data") without out_path, ask where to save (suggest the default), then export; offer the
   masked text (kind="masked") too.

PRIVACY RULES (always)
- Pass file paths; never paste or quote report text into the chat.
- Always ask the user before privlab_review; never answer a review question for them.
- Never ask the user to type a name into the chat (it would reach the assistant). To hide a missed name, send
  them to the viewer.

VIEWER
PrivLab has a visual page running on this computer. When you de-identify a report, PrivLab opens it in the
user's browser by itself: tell the user it has opened, give its viewer_url as a link in case it did not, and say
what it is: "the same report on your own screen, with every personal
detail highlighted; you can check and correct it there, and the real text never comes into this chat." Offer
it again for extracted values. privlab_deidentify (no arguments) gives the link any time.
Never open, fetch or read the viewer link yourself: it is for the user's eyes only, because it shows the real
report. Just give it to the user as a link."""


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


def _show(deid_id: str) -> None:
    """Open the user's browser on this report, once per report, from this computer: the user sees the highlights
    without clicking a link, and the assistant never touches the page. PRIVLAB_MCP_OPEN=off turns it off."""
    url = _url(deid_id)
    if url and deid_id and os.environ.get("PRIVLAB_MCP_OPEN", "").lower() != "off":
        threading.Timer(0.3, webbrowser.open, [url]).start()


def _see(deid_id: str, what: str) -> str:
    """A sentence pointing to the viewer, for "say": the user learns there is a visual page and where it is."""
    url = _url(deid_id)
    return (f" Tell the user they can {what} in the PrivLab viewer, on this computer only: {url} (give the user this link; never open it yourself)") if url else ""


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
                       "privlab_extract, or privlab_export(kind='masked')."}
    items = out.get("items") or out.get("detections") or out.get("hold") or out.get("flags") or []
    items = [{**i, "context": scrub(i.get("context", ""))} for i in items]
    page = _page(items, offset, limit)
    say = {"review": f"{page['total_count']} identifiers found on this laptop; nothing has been sent. Ask the user "
                     "which ids to keep visible (false alarms) and whether anything was missed, then call "
                     "privlab_review." + _see(did, "check them highlighted on your own screen"),
           "hold": "These words still look like names. They will all be hidden unless the user says some are not "
                   "names (keep_ids). Ask, then call privlab_review. Nothing has been sent yet.",
           "flags": "The second check (on masked text only) found more. Ticked items will be hidden; ask the user "
                    "which are false alarms (keep_ids) and which unticked ones to hide (hide_ids)."}.get(nxt, "")
    return {**base, "counts": out.get("counts"), **page, "say": say}


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
mcp = MCPServer("privlab", instructions=INSTRUCTIONS)


class Answer(BaseModel):
    """The user's answer for one field."""
    field: str = Field(description="the field's key, from privlab_extract")
    candidate_id: str = Field("", description="accept this candidate as it is")
    value: str = Field("", description="a corrected value, as written in the masked report")
    evidence_text: str = Field("", description="the exact words of the masked report that state the corrected value")
    not_stated: bool = Field(False, description="the report does not state this field")
    items: list[str] = Field([], description="a list field: every item as written (the whole list)")
    note: str = Field("", description="optional: why PrivLab's pick was wrong")


# --- 1. Review identifiers (and where a report is up to) ------------------------------------------------------
@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=False))
@_safe
def privlab_deidentify(path: str = "", deid_id: str = "", offset: int = 0, limit: int = LIMIT) -> dict:
    """Use when the user wants a report (or a folder of reports) made safe, wants to know where a report is up to,
    or wants the viewer.
    Arguments: path = a .txt file or a PDF with selectable text, or a folder of them (never paste report text):
    finds the personal details on this computer, nothing is sent. Or deid_id = a report already started: where it
    is up to, and the next page of its items (offset, limit). Neither: the viewer link and how to start.
    Returns: deid_id, next, items (id, type, masked context), counts, viewer_url; for a folder, one line per report.
    Next: ask the user, then privlab_review."""
    if path and Path(path).expanduser().is_dir():
        return _folder(Path(path).expanduser())
    if path:
        text, rid, warnings = _read(path)
        if text is None:
            return {"next": "error", "error": rid, "say": rid}
        again = _started(Path(path).expanduser().resolve(), text)
        if again:
            _show(again)
            out = _step(pipeline.report_state(again), 0, limit)
            return {**out, "say": "This file was already started in this session; continuing it. " + out.get("say", "")}
        with viewer._WRITE:                     # first use creates the default privacy profile
            out = pipeline.deidentify_report(text=text, report_id=rid, profile_choices={})
        out["warnings"] = out.get("warnings", []) + warnings
        SOURCES[out.get("deid_id", "")] = Path(path).expanduser().resolve()
        STARTED[(SOURCES[out.get("deid_id", "")], hash(text))] = out.get("deid_id", "")
        _show(out.get("deid_id", ""))
        return _step(out, 0, limit)
    if deid_id:
        return _step(pipeline.report_state(deid_id), offset, limit)
    return {"next": "start", "viewer_url": _url(),
            "say": "Give me the path of a report (.txt or PDF) to start." + _see("", "also add and review reports")}


MAX_FOLDER = 100
SOURCES: dict[str, Path] = {}             # deid_id -> the file it came from (for a default output folder)
STARTED: dict[tuple, str] = {}            # (file, text) -> deid_id: the same file again continues, not a copy


def _started(path: Path, text: str) -> str | None:
    """The report already started from this file with this text, if it is still in the session (not removed)."""
    d = STARTED.get((path, hash(text)))
    return d if d in pipeline._RUNS else None


def _names(names: list[str]) -> str:
    """File names for a sentence: a range if they run in sequence (PP0101 to PP0200), else the first few."""
    m = [re.fullmatch(r"(.*?)(\d+)", Path(n).stem) for n in names]
    if (len(names) > 2 and all(m) and len({(x[1], len(x[2])) for x in m}) == 1
            and [int(x[2]) for x in m] == list(range(int(m[0][2]), int(m[0][2]) + len(m)))):
        return f"{Path(names[0]).stem} to {Path(names[-1]).stem}"
    return ", ".join(Path(n).stem for n in names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else "")


def _folder(folder: Path) -> dict:
    """Every .txt and .pdf in a folder (not its subfolders), de-identified on this computer; nothing is sent.
    At most MAX_FOLDER new reports per call; files already started do not count, so calling again continues."""
    files = sorted(f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in (".txt", ".pdf"))
    if not files:
        return {"next": "error", "error": "No .txt or .pdf files.", "say": f"There are no .txt or .pdf reports in {folder}."}
    reports, skipped, left = [], [], []
    started = {p for p, _ in STARTED}
    for f in files:
        if sum(not r.get("already_started") for r in reports) >= MAX_FOLDER and f.resolve() not in started:
            left.append(f.name)
            continue
        text, rid, warnings = _read(str(f))
        if text is None:
            skipped.append({"file": f.name, "reason": rid})
            continue
        again = _started(f.resolve(), text)
        if again:
            reports.append({"deid_id": again, "report": rid, "already_started": True,
                            "found": len(pipeline._RUNS[again]["spans"])})
            continue
        with viewer._WRITE:
            out = pipeline.deidentify_report(text=text, report_id=rid, profile_choices={})
        SOURCES[out["deid_id"]] = f.resolve()
        STARTED[(f.resolve(), hash(text))] = out["deid_id"]
        reports.append({"deid_id": out["deid_id"], "report": rid, "found": sum((out.get("counts") or {}).values()),
                        "counts": out.get("counts"), "possible_names": sum(d["source"] == "hold" for d in out["detections"])})
    if reports:
        _show(reports[0]["deid_id"])               # one tab: the browser lists every report of the batch
    total = sum(r["found"] for r in reports)
    say = (f"{len(reports)} report(s) read on this computer; {total} personal details found and hidden, nothing sent yet."
           + (f" {len(skipped)} file(s) skipped." if skipped else "")
           + (f" PrivLab reads {MAX_FOLDER} new reports at a time, so {len(left)} were not processed yet: {_names(left)}."
              " Tell the user, and offer to run the same folder again to continue with them (reports already"
              " started are kept, not read twice)." if left else "")
           + " Ask the user whether to review any (they are all listed in the browser) or to hide them all; then call"
             " privlab_review for each report." + _see(reports[0]["deid_id"] if reports else "", "check every report"))
    return {"next": "review_batch", "reports": reports, "skipped": skipped, "not_processed": left,
            "viewer_url": _url(), "say": say}


# --- 2. De-identify --------------------------------------------------------------------------------------------
@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=True))
@_safe
def privlab_review(deid_id: str, keep_ids: list[str] = [], hide_ids: list[str] = [],
                   hide_text: list[str] = [], hide_type: HideType = "NAME") -> dict:
    """Use when the user has answered the current question for a report (review, possible names, safety check).
    Arguments: keep_ids = ids to leave visible (not personal); hide_ids = unticked safety-check ids to hide;
    hide_text = missed text to hide, at the first review only (it puts the text in the chat: for names, send the
    user to the viewer instead); hide_type = its type. Ask the user first; never answer for them.
    Returns: the next question, or the de-identified text. After the first answer, the masked text (only) is sent
    once for a safety check."""
    run = pipeline._RUNS.get(deid_id)
    if run is None:
        return {"next": "error", "error": "Unknown deid_id.", "say": "Unknown report: de-identify it first."}
    with viewer._report_lock(deid_id):
        step = run["step"]                      # read under the lock: the viewer may have moved the report on
        if step in ("hold", "flags") and [t for t in hide_text if t.strip()]:
            msg = ("Missed text can only be added at the first review. Nothing was changed. To hide more, open the "
                   "report in the viewer (viewer_url) and select the text there.")
            return {"deid_id": deid_id, "next": step, "error": msg, "say": msg}
        if step == "review":
            out = pipeline.confirm_deid(deid_id, unmark=keep_ids, expect="review",
                                        add=[{"text": t, "type": hide_type} for t in hide_text if t.strip()])
        elif step == "hold":
            if hide_ids:
                msg = ("At this question every word is hidden unless you keep it: send keep_ids only. "
                       "Nothing was changed.")
                return {"deid_id": deid_id, "next": "hold", "error": msg, "say": msg}
            out = pipeline.confirm_deid(deid_id, hold_answers={"keep": keep_ids}, expect="hold")
        elif step == "flags":
            out = pipeline.confirm_deid_flags(deid_id, keep=keep_ids, hide=hide_ids)
        else:
            return {"deid_id": deid_id, "next": "done", "viewer_url": _url(deid_id),
                    "say": "This report is already de-identified. Next: privlab_extract or privlab_export."}
    return _step(out)


# --- 3. Choose fields --------------------------------------------------------------------------------------------
@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=True))
@_safe
def privlab_fields(action: Literal["list", "draft", "save", "remove"] = "list", name: str = "",
                   description: str = "", example: str = "", meaning: str = "", definition: str = "",
                   type: Literal["number", "text"] | None = None, unit: str = "", range_min: float | None = None,
                   range_max: float | None = None, multi: bool = False, offset: int = 0, limit: int = LIMIT) -> dict:
    """Use for the user's fields (what PrivLab fills in, e.g. "tumour size").
    action="list": every field with its definition, version, readiness and accuracy.
    action="draft": name + description (what it means, in the user's words) + example (one value as a report
    writes it, e.g. "2.1 cm"); ask the user for any that is missing. Only these are sent -> a precise definition
    to confirm ("Did you mean ...?"). action="save": save it with the draft's details or the user's changes
    (meaning, definition, type, unit, range; multi=True for a list such as biomarker results), then preview it on
    one de-identified report. action="remove": stop collecting the field called name; only when the user asks,
    after confirming.
    Returns: the fields, the draft, or the saved field and its preview. Next: privlab_extract."""
    if action == "list":
        keep = ("key", "name", "meaning", "definition", "type", "unit", "range", "version", "readiness", "validation")
        fields = [{k: f[k] for k in keep} for f in viewer.api_pack(None)["fields"]]
        return {"next": "done", **_page(fields, offset, limit),
                "say": f"{len(fields)} field(s) defined." if fields else "No fields yet: draft one (action='draft')."}
    if not name:
        msg = "Give the field's name."
        return {"next": "error", "error": msg, "say": msg}
    if action == "remove":
        with viewer._WRITE:
            pack = load_pack(viewer.STATE["pack"])
            if not remove_field(pack, name):
                return {"next": "error", "error": "No such field.", "say": "That field is not in the list."}
            save_pack(pack, viewer.STATE["pack"])
        return {"next": "done", "say": f"Removed '{name}'. Its past confirmations stay in the score log."}
    if action == "draft":
        missing = [w for w, v in (("what it means, in the user's words (description)", description),
                                  ("one example value as a report writes it (example)", example)) if not v.strip()]
        if missing:
            return {"next": "ask", "missing": missing,
                    "say": f"Before drafting '{name}', ask the user for: {'; '.join(missing)}."}
        d = define_field(name, f"{description.strip()}. Example value: {example.strip()}",   # as the viewer sends it
                         pack_path=viewer.STATE["pack"])["draft"]
        return {"next": "confirm", "draft": d,
                "say": f"Did you mean: {d['meaning_question']} {d.get('definition', '')} Ask the user; then call "
                       "again with action='save' and these details (changed if the user wants)."}
    if not meaning or type is None:
        msg = "To save, give meaning and type (from the draft: action='draft' first)."
        return {"next": "error", "error": msg, "say": msg}
    accept = {"key": name, "meaning_question": meaning, "definition": definition, "type": type,
              "unit": unit or None, "range_min": range_min, "range_max": range_max, "multi": multi}
    with viewer._WRITE:
        f = define_field(name, description or None, accept=accept, pack_path=viewer.STATE["pack"])["field"]
    saved = {k: f[k] for k in ("key", "name", "meaning", "definition", "type", "unit", "range", "multi", "version")}
    return {"next": "preview", "field": saved, **_preview(f["key"], name)}


def _preview(key: str, name: str) -> dict:
    """The new field on one de-identified report of this session (one model call, cached for its extraction)."""
    deid_id = next((d for d, r in reversed(pipeline._RUNS.items()) if r["step"] == "done"), None)
    if deid_id is None:
        return {"say": f"Saved '{name}'. De-identify a report to see it filled in."}
    r = find_candidates(deid_id, key, pack_path=viewer.STATE["pack"])
    report = pipeline._RUNS[deid_id]["report_id"]
    if r.get("status") == "error":
        return {"say": f"Saved '{name}'. The preview could not run: {r.get('reason')}."}
    value = r.get("items") and "; ".join(str(i["value"]) for i in r["items"]) or r.get("value")
    if value is None and not r.get("candidates") and r.get("status") == "amber":          # the model call failed
        return {"say": f"Saved '{name}'. The preview could not run ({r.get('reason')}); it will be read at extraction."}
    if value is None:
        return {"preview": {"report": report, "value": None},
                "say": f"Saved '{name}'. On {report} PrivLab found no value. Ask the user if that is right; if not, "
                       "draft again with a clearer description or example."}
    return {"preview": {"report": report, "value": value, "quote": r.get("evidence_text"), "status": r.get("status")},
            "viewer_url": _url(deid_id),
            "say": f"Saved '{name}'. Preview on {report}: {value}, from \"{r.get('evidence_text')}\". Ask the user if "
                   "that is the value they want; if not, draft again with their correction."}


# --- 4. Extract --------------------------------------------------------------------------------------------------
def _confirm(deid_id: str, a: Answer) -> dict:
    with viewer._WRITE:
        out = learn.confirm_field(deid_id, a.field, candidate_id=a.candidate_id or None, value=a.value or None,
                                  evidence_text=a.evidence_text or None, missing=a.not_stated, note=a.note or None,
                                  items=[{"value": x, "evidence_text": x} for x in a.items] or None,
                                  pack_path=viewer.STATE["pack"], log_path=viewer.STATE["log"])
    if out.get("status") != "saved":
        return {"field": a.field, "saved": False, "error": out.get("reason")}
    r = {"field": a.field, "saved": True, **{k: out[k] for k in ("outcome", "value", "readiness")}}
    if out.get("blind"):
        r["blind_check_prediction"] = out["predicted"]       # shown only now, after the user's answer
    return r


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=True))
@_safe
def privlab_extract(deid_id: str, answers: list[Answer] = [], auto_accept_verified: bool = False) -> dict:
    """Use when a report is de-identified and the user wants their fields filled, and again with their answers.
    Without answers: reads the masked text (one model call for all fields, cached) and returns per field: status
    (green = verified, amber = needs review, missing, blind = ask the user first), value, quote, reason, candidates.
    With answers (one per field the user answered: accept a candidate_id, a corrected value + evidence_text,
    not_stated, or items for a list field): saves them all; each is scored first, then PrivLab learns from it.
    auto_accept_verified=True (bulk, after the user has taught the fields on a few reports): values PrivLab verified
    are saved by themselves, never used for teaching and never counted as the user's check; only the others come
    back for the user.
    Returns: the values to show, or what was saved (and, for a blind check, PrivLab's own answer)."""
    run = pipeline._RUNS.get(deid_id)
    if run is None or run["step"] != "done":
        return {"next": "error", "error": "Not de-identified yet.",
                "say": "Finish de-identifying this report first (privlab_review)."}
    if any("left in the output" in r for r in (run.get("priority") or {}).get("reasons", [])):
        return {"next": "blocked", "say": "A possible identifier is left in this report, so nothing is extracted."}
    if answers:
        saved = [_confirm(deid_id, a) for a in answers]
        ok = [r for r in saved if r["saved"]]
        blind = [r for r in ok if "blind_check_prediction" in r]
        say = f"Saved {len(ok)} of {len(saved)} answer(s)." + (
            f" Blind check(s): PrivLab had " + ", ".join(f"{r['field']} = {r['blind_check_prediction']}" for r in blind)
            + "." if blind else "") + (" Next: privlab_export." if len(ok) == len(saved) else " Fix the others and send them again.")
        return {"deid_id": deid_id, "next": "done" if len(ok) == len(saved) else "confirm", "saved": saved,
                "viewer_url": _url(deid_id), "say": say}
    fields = load_pack(viewer.STATE["pack"])["fields"]
    if not fields:
        return {"next": "define", "say": "No fields defined yet: use privlab_fields(action='draft')."}
    if auto_accept_verified:
        return _bulk(deid_id, fields)
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
            "say": "Show each value with its quote. Ask the user to confirm, correct or mark it not stated, then send "
                   "all the answers in one privlab_extract call." + _see(deid_id, "see every value highlighted in the report")
                   + (f" {blind} blind check(s): ask the user for the value without showing PrivLab's." if blind else "")}


def _bulk(deid_id: str, fields) -> dict:
    """Like the viewer's Extract all: a verified value is saved by itself (phase "bulk": never taught, never scored as
    a person's check); a blind check, a missing value or one that needs review comes back for the user."""
    auto, need = [], []
    for f in fields:
        r = find_candidates(deid_id, f.key, pack_path=viewer.STATE["pack"], blind_checks=True)
        if r.get("status") == "green":
            with viewer._WRITE:
                out = learn.confirm_field(deid_id, f.key, candidate_id=r.get("candidate_id") if not f.multi else None,
                                          items=[{"value": it["value"], "evidence_text": it["evidence_text"]}
                                                 for it in r.get("items", [])] if f.multi else None,
                                          phase="bulk", pack_path=viewer.STATE["pack"], log_path=viewer.STATE["log"])
            if out.get("status") == "saved":
                auto.append({"field": f.key, "value": out["value"]})
                continue
        need.append({"field": f.key, "name": f.name, "status": r.get("status"), "value": r.get("value"),
                     "quote": r.get("evidence_text"), "reason": r.get("reason"), "candidate_id": r.get("candidate_id"),
                     "candidates": [{"id": c["id"], "value": c["value"], "quote": c["evidence_text"]}
                                    for c in r.get("candidates", [])]})
    say = (f"{len(auto)} verified value(s) saved by themselves." + (f" {len(need)} need the user: show each with its "
           "quote (for a blind check, ask without showing PrivLab's value), then send the answers with privlab_extract."
           if need else " Nothing needs the user on this report."))
    return {"deid_id": deid_id, "next": "confirm" if need else "done", "auto_saved": auto, "fields": need,
            "viewer_url": _url(deid_id), "say": say}


# --- 5. Export ---------------------------------------------------------------------------------------------------
@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=False))
@_safe
def privlab_export(kind: Literal["data", "masked"], out_path: str = "") -> dict:
    """Use when the user wants a file, for the reports of this session, written on this computer.
    kind="data": a CSV, one row per report and one column per field (values as written, no report text).
    kind="masked": every finished report's de-identified text, as a .zip.
    out_path: where to write it. Leave it empty first: PrivLab suggests a folder next to the reports ("PrivLab
    output", created if missing); ask the user, then call again with their choice or the suggestion.
    Returns: the suggestion, or the file path and how much it holds (never the contents)."""
    if not out_path:
        base = next(reversed(SOURCES.values()), None)             # next to the latest report or folder
        folder = (base.parent if base else Path.home() / "Documents") / "PrivLab output"
        name = f"privlab_extracted_data_{viewer._stamp()}.csv" if kind == "data" else f"privlab_masked_{viewer._stamp()}.zip"
        suggestion = str(folder / name)
        return {"next": "ask", "suggested_path": suggestion,
                "say": f"Ask the user where to save it. Suggest: {suggestion} (the folder is created if missing)."}
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


# --------------------------------------------------------------------------
# Prompts: the playbook
# --------------------------------------------------------------------------
@mcp.prompt(title="Fill my fields")
def fill_fields(path: str) -> str:
    return (f"Make {path} safe with PrivLab, then fill my fields: privlab_deidentify(path); show me the counts and the "
            "viewer link; ask me what to keep visible; privlab_review until done; privlab_extract; show each value "
            "with its quote and ask me to confirm, correct or mark not stated (for a blind check, ask me first without "
            "showing PrivLab's value); send all my answers in one privlab_extract call; then privlab_export(kind='data').")


@mcp.prompt(title="De-identify a batch of reports")
def deidentify_batch(folder: str) -> str:
    return (f"De-identify every .txt or .pdf report in {folder} with PrivLab. For each file: privlab_deidentify(path); "
            "show me the counts and the viewer link; ask me what to keep visible; then privlab_review with my answers, "
            "repeating while next is hold or flags and asking me each time. Never paste report text into the chat. "
            "At the end, privlab_export(kind='masked') to a path I choose.")


@mcp.prompt(title="Set up my fields")
def set_up_fields(fields: str) -> str:
    return (f"Help me define these fields in PrivLab: {fields}. For each: ask me what it means and one example value "
            "if I have not said; privlab_fields(action='draft'), show me the 'Did you mean' question, definition and "
            "type; change anything I ask; then action='save' and show me the preview. "
            "Finish with privlab_fields(action='list').")


@mcp.prompt(title="Process a folder of reports")
def process_folder(folder: str) -> str:
    return (f"Process every report in {folder} with PrivLab, following the bulk steps: de-identify the folder, show "
            "me the summary and ask whether to review any or hide them all; ask if I want to check the masked text in "
            "my browser; extract with auto_accept_verified=True and show me only what needs me; then ask me where to "
            "save (suggest the default) and export the CSV, and offer the masked text.")


@mcp.prompt(title="Open the PrivLab viewer")
def open_viewer() -> str:
    return ("Call privlab_deidentify with no arguments and give me the viewer link as a clickable link. Tell me what it "
            "is: PrivLab's page on this computer, where I can add reports, see personal details highlighted and check "
            "values; the real text never comes into this chat.")


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
    logging.basicConfig(filename=ROOT / "runs" / "mcp.log", level=logging.INFO, force=True,   # the MCP SDK
                        format="%(asctime)s %(name)s %(message)s")       # already put a handler on the root logger
    with contextlib.redirect_stdout(sys.stderr):        # stdout is the protocol: no stray prints
        pipeline.warm_up()
        start_viewer()
    mcp.run("stdio")


if __name__ == "__main__":
    main()
