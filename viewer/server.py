"""
server.py - the local viewer: one HTML page plus a JSON API over the same functions the MCP
server calls (deid/pipeline.py, extract/). Standard library only; listens on 127.0.0.1.

Raw report text is shown only here, on the laptop. Muse Spark is called only by confirm steps,
"find" and the extraction prefetch (after de-identification), exactly as in the MCP flow.
Reports: dropped or pasted text (kept in memory only), or samples from dev and extra (test is
never listed, and an upload named like a test report is refused).

Threaded, so a 2-10 s model call never blocks other clicks; steps on one report are serialised
by a per-report lock, and quick pack / log / profile writes by one shared lock.

--demo keeps the privacy profile, pack and score log in runs/demo/, with a "Start over" button,
so a video can be re-recorded from a clean first use. Without it the real profile, pack and
runs/score_log.jsonl are used (Day 7 teaching).

Exports (only the reports in the page's list, ?ids=...; a missing or blank ids exports nothing):
/api/export.csv (extracted data, one row per report, no text) and /api/deid_export (de-identified
text: ?deid_id=… for one report as .txt, ?ids=… for the listed finished reports as .zip). A report whose
output scan still found an identifier pattern is never exported.

Usage:
  python -m viewer.server            # http://127.0.0.1:8765
  python -m viewer.server --demo
"""
import argparse
import base64
import csv
import io
import json
import logging
import shutil
import socket
import sys
import threading
import traceback
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from common import spark
from common.config import ROOT
from common.data import load_report, manifest
from common.pdf import pdf_text
from common.timing import timed
from deid import pipeline
from deid import profile as prof
from extract import learn
from extract.candidates import find_candidates, prefetch
from extract.define import define_field
from extract.export import export_rows
from extract.pack import load_pack, remove_field, save_pack

PAGE = Path(__file__).with_name("index.html")
DEMO_DIR = ROOT / "runs" / "demo"
MAX_UPLOAD = 200_000                     # characters; pathology reports are a few thousand
log = logging.getLogger("privlab.viewer")
STATE = {"pack": None, "log": None, "demo": False}
_WRITE = threading.Lock()                # pack, score log and profile files
_REPORT_LOCKS: dict[str, threading.Lock] = {}
_GUARD = threading.Lock()


def _report_lock(deid_id: str) -> threading.Lock:
    with _GUARD:
        return _REPORT_LOCKS.setdefault(deid_id, threading.Lock())


# --------------------------------------------------------------------------
# API: one function per action; each returns a JSON-able dict
# --------------------------------------------------------------------------
def api_reports(_):
    return {"reports": [{"id": r["report_id"], "split": r["split"], "cancer": r.get("cancer_type", "")}
                        for r in manifest() if r["split"] in ("dev", "extra")], "demo": STATE["demo"]}


def api_open(b):
    """A report started in the chat (MCP server, same process): where it is up to, with its text (local only)."""
    return pipeline.report_state(str(b.get("deid_id", "")), show_text=True)


def api_profile(_):
    p = prof.load_profile()
    return {"exists": p is not None, "checklist": prof.checklist(), "remove": p.remove if p else None,
            "allow": len(p.allow_list) if p else 0, "learned": len(p.always_remove) if p else 0}


def api_profile_choices(b):
    """Change what is hidden. Keeps what the profile learned (allow and remove lists)."""
    with _WRITE:
        p = prof.load_profile() or prof.new_profile()[0]
        for key, on in (b.get("choices") or {}).items():
            if key in p.remove:
                p.remove[key] = bool(on)
        p.version += 1
        prof.save_profile(p)
    return api_profile(None)


def _start(text: str, name: str, choices) -> dict:
    with _WRITE:                         # first use saves the profile
        out = pipeline.deidentify_report(text=text, report_id=name, profile_choices=choices, show_text=True)
    return {**out, "name": name, "text": text}


def api_deid(b):
    """A sample report from dev or extra."""
    rid = b["report_id"]
    if not any(r["report_id"] == rid and r["split"] in ("dev", "extra") for r in manifest()):
        return {"error": "unknown sample (only dev and extra are available here)"}
    return {**_start(load_report(rid), rid, b.get("choices")), "sample": True}


def api_add(b):
    """A dropped or pasted report, or a text PDF (pdf: base64). Kept in memory only; nothing is sent anywhere."""
    text, name, warnings = b.get("text"), str(b.get("name") or "pasted report").strip()[:60], []
    if b.get("pdf") is not None:
        try:
            out = pdf_text(base64.b64decode(b["pdf"], validate=True))
        except (ValueError, TypeError):
            return {"error": "The PDF upload was damaged. Add it again.", "refused": True}
        if "error" in out:
            return {"error": out["error"], "refused": True}
        text, warnings = out["text"], out["warnings"]
    if not isinstance(text, str) or not text.strip():
        return {"error": "The report is empty."}
    if len(text) > MAX_UPLOAD:
        return {"error": f"The report is too long ({len(text):,} characters; limit {MAX_UPLOAD:,})."}
    name = Path(name).stem or "pasted report"
    if any(r["report_id"] == name and r["split"] == "test" for r in manifest()):
        return {"error": "That name belongs to a held-out test report; rename the file."}
    out = _start(text.replace("\r\n", "\n").replace("\r", "\n"), name, b.get("choices"))
    return {**out, "warnings": out.get("warnings", []) + warnings}


def _locked_step(deid_id: str, fn):
    with _report_lock(deid_id):          # not _WRITE: it would hold every other click during the name check
        return fn()


def api_confirm(b):
    return _locked_step(b["deid_id"], lambda: pipeline.confirm_deid(
        b["deid_id"], unmark=b.get("unmark", []), add=b.get("add", []), show_text=True))


def api_hold(b):
    return _locked_step(b["deid_id"], lambda: pipeline.confirm_deid(
        b["deid_id"], hold_answers={"keep": b.get("keep", [])}, show_text=True))


def api_flags(b):
    return _locked_step(b["deid_id"], lambda: pipeline.confirm_deid_flags(
        b["deid_id"], keep=b.get("keep", []), hide=b.get("hide", [])))


def api_prefetch(b):
    return prefetch(b["deid_id"], pack_path=STATE["pack"])


def api_pack(_):
    log = learn.read_log(STATE["log"])
    return {"fields": [{"key": f.key, "name": f.name, "meaning": f.meaning, "definition": f.definition, "description": f.description,
                        "type": f.type, "unit": f.unit, "range": f.range,
                        "readiness": learn.readiness(v), "recent": f.recent, "version": f.version,
                        "validation": v, "history": f.history,
                        "checks": [{"when": e["ts"], "version": e.get("field_version", 1), "status": e["predicted"]["status"],
                                    "predicted": e["predicted"]["value"], "confirmed": (e["confirmed"] or {}).get("value"),
                                    "outcome": e["outcome"], "blind": e.get("blind", False)}
                                   for e in reversed(log) if e["field"] == f.key][:20]}
                       for f in load_pack(STATE["pack"])["fields"] for v in [learn.validation(f, log)]]}


def api_define(b):
    if b.get("accept") is None:          # the draft: one model call, no file write
        return define_field(b["name"], b.get("description"), pack_path=STATE["pack"])
    with _WRITE:
        return define_field(b["name"], b.get("description"), accept=b["accept"], pack_path=STATE["pack"])


def api_back(b):
    return _locked_step(b["deid_id"], lambda: pipeline.back_to_review(b["deid_id"]))


def api_fix_leak(b):
    return _locked_step(b["deid_id"], lambda: pipeline.resolve_leftover(b["deid_id"], b["leftover_id"], b["action"]))


def api_remove_field(b):
    with _WRITE:
        pack = load_pack(STATE["pack"])
        if not remove_field(pack, b["field"]):
            return {"error": "That field is not in your pack."}
        save_pack(pack, STATE["pack"])
    return {"ok": True}


def api_find(b):
    return find_candidates(b["deid_id"], b["field"], pack_path=STATE["pack"], blind_checks=bool(b.get("blind_ok")))


def api_confirm_field(b):
    with _WRITE:
        return learn.confirm_field(b["deid_id"], b["field"], candidate_id=b.get("candidate_id"),
                                   value=b.get("value"), evidence_text=b.get("evidence_text"),
                                   missing=bool(b.get("missing")), note=b.get("note"),
                                   pack_path=STATE["pack"], log_path=STATE["log"])


def api_reset(_):
    if not STATE["demo"]:
        return {"error": "Start over is only available with --demo"}
    with _WRITE:
        shutil.rmtree(DEMO_DIR, ignore_errors=True)
        pipeline._RUNS.clear()
    return {"ok": True}


def export_csv(deid_ids=None) -> bytes:
    """One row per report, one column per field (extract/export.py); no report text, no positions."""
    columns, rows, _ = export_rows(pack_path=STATE["pack"], log_path=STATE["log"], deid_ids=deid_ids)
    return _csv(columns, rows)


def _csv(columns, rows) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")              # -sig: Excel opens UTF-8 correctly


def _finished(deid_id: str | None = None, deid_ids=None) -> list[tuple[str, str]]:
    """(file name, de-identified text) of finished reports (one, or only deid_ids); a report whose output scan
    still found an identifier pattern is withheld."""
    out, seen = [], set()
    for did, run in list(pipeline._RUNS.items()):
        if (deid_id and did != deid_id) or (deid_ids is not None and did not in deid_ids) \
                or run.get("step") != "done" or not run.get("result"):
            continue
        if any("left in the output" in x for x in (run.get("priority") or {}).get("reasons", [])):
            continue
        name = Path(str(run.get("report_id") or "report")).name or "report"
        name = f"{name}_deidentified.txt" if name not in seen else f"{name}_{did}_deidentified.txt"
        seen.add(Path(str(run.get("report_id") or "report")).name)
        out.append((name, run["result"].text))
    return out


def deid_download(deid_id: str | None, deid_ids=None) -> tuple[bytes, str, str] | None:
    """One report as .txt, or the finished reports (only deid_ids, if given) as .zip. None if there is nothing."""
    files = _finished(deid_id, deid_ids)
    if not files:
        return None
    if deid_id:
        return files[0][1].encode("utf-8"), "text/plain; charset=utf-8", files[0][0]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in files:
            z.writestr(name, text)
    return buf.getvalue(), "application/zip", f"privlab_deidentified_{_stamp()}.zip"


DICTIONARY_TXT = """PrivLab export: data dictionary ({when})

extracted_data.csv
  One row per report, one column per field. Values are exactly as the report writes them; number fields
  carry their unit in the column name, e.g. "tumour size (cm)". An empty cell means not stated (or not
  reviewed yet).

  Column            Type      Meaning
  case_id           text      Random id for the report (the link to the original stays in PrivLab's encrypted vault)
  report_id         text      The report's file name
  reviewed_at       datetime  When its last value was confirmed
  exported_at       datetime  When this file was made
{fields}

masked/
  The full text of each de-identified report, with identifiers replaced by tags such as [PATIENT-1].
  A report where a possible identifier was left in the text is not included.

Contains no original identifiers. Made on this device by PrivLab.
"""


def _field_lines() -> str:
    """One entry per field column: its header, type and definition (version)."""
    from extract.export import headers
    fields = load_pack(STATE["pack"])["fields"]
    column, out = headers(fields), []
    for f in fields:
        kind = "number" if f.type == "number" else "text"     # the unit is already in the column name
        meaning = " ".join((f.definition or f.meaning).split())
        out.append(f"  {column[f.key]:<17} {kind:<9} {meaning} [definition v{f.version}]")
    return "\n".join(out)


def export_all(deid_ids) -> bytes | None:
    """Everything for the listed reports in one .zip: the extracted data, the masked texts and a data dictionary.
    None if there is nothing to export."""
    from datetime import datetime
    columns, rows, _ = export_rows(pack_path=STATE["pack"], log_path=STATE["log"], deid_ids=deid_ids)
    masked = _finished(None, deid_ids)
    if not rows and not masked:
        return None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("data_dictionary.txt", DICTIONARY_TXT.format(when=datetime.now().strftime("%d %b %Y, %H:%M"),
                                                                fields=_field_lines()))
        if rows:
            z.writestr("extracted_data.csv", _csv(columns, rows))
        for name, text in masked:
            z.writestr(f"masked/{name}", text)
    return buf.getvalue()


def _stamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d-%H%M")


def _ids(url) -> list[str]:
    """The reports the page lists (?ids=d_1,d_2). Missing or blank means none: a link opened without the
    page's click handler (middle-click, copied address) must never export reports from other sessions."""
    return [x for x in (parse_qs(url.query).get("ids", [""])[0]).split(",") if x]


GET = {"/api/reports": api_reports, "/api/profile": api_profile, "/api/pack": api_pack, "/api/open": api_open}
POST = {"/api/deid": api_deid, "/api/add": api_add, "/api/confirm": api_confirm, "/api/hold": api_hold,
        "/api/flags": api_flags, "/api/prefetch": api_prefetch, "/api/define": api_define, "/api/find": api_find,
        "/api/confirm_field": api_confirm_field, "/api/fix_leak": api_fix_leak, "/api/back": api_back, "/api/profile_choices": api_profile_choices, "/api/remove_field": api_remove_field, "/api/reset": api_reset}


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if url.path == "/api/export.csv":
            try:
                body = export_csv(_ids(url))
            except Exception:
                log.error("export failed: %s", traceback.format_exc())
                return self._json(500, {"error": "export failed; see runs/viewer.log"})
            return self._send(200, body, "text/csv; charset=utf-8",
                              {"Content-Disposition": f'attachment; filename="privlab_extracted_data_{_stamp()}.csv"'})
        if url.path == "/api/export_all":
            try:
                body = export_all(_ids(url))
            except Exception:
                log.error("export failed: %s", traceback.format_exc())
                return self._json(500, {"error": "export failed; see runs/viewer.log"})
            if body is None:
                return self._json(404, {"error": "Nothing to export yet."})
            return self._send(200, body, "application/zip",
                              {"Content-Disposition": f'attachment; filename="privlab_export_{_stamp()}.zip"'})
        if url.path == "/api/deid_export":
            try:
                one = parse_qs(url.query).get("deid_id", [None])[0]
                out = deid_download(one, None if one else _ids(url))
            except Exception:
                log.error("de-identified export failed: %s", traceback.format_exc())
                return self._json(500, {"error": "export failed; see runs/viewer.log"})
            if out is None:
                return self._json(404, {"error": "No de-identified reports to export yet."})
            body, ctype, fname = out
            return self._send(200, body, ctype, {"Content-Disposition": f'attachment; filename="{fname}"'})
        fn = GET.get(url.path)
        if fn is None:
            return self._json(404, {"error": "not found"})
        self._run(fn, {k: v[0] for k, v in parse_qs(url.query).items()})

    def do_POST(self):
        fn = POST.get(urlparse(self.path).path)
        if fn is None:
            return self._json(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except ValueError:
            return self._json(400, {"error": "bad JSON"})
        self._run(fn, body)

    def _run(self, fn, arg):
        try:
            with timed("viewer_" + fn.__name__):
                out = fn(arg)
        except spark.OfflineError:
            out = {"error": f"{spark.OFFLINE}. De-identification still works on this device; the second check "
                            "and fields need Muse Spark. Try again when online.", "offline": True}
        except Exception as e:      # never crash the page; details go to the log, not the browser
            log.error("%s failed: %s", fn.__name__, traceback.format_exc())
            out = {"error": f"{type(e).__name__}: something went wrong; see runs/viewer.log"}
        self._json(200, out)

    def _json(self, code, data):
        self._send(code, json.dumps(data, default=str).encode(), "application/json")

    def _send(self, code, body, ctype, headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        log.info(fmt, *args)


class ExclusiveServer(ThreadingHTTPServer):
    """One viewer per port. Python's default lets a second server bind the same port on Windows,
    and requests then reach whichever copy (often an old one) answers first."""
    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):                  # Windows
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def make_server(port: int) -> ThreadingHTTPServer:
    return ExclusiveServer(("127.0.0.1", port), Handler)


def main() -> int:
    ap = argparse.ArgumentParser(description="PrivLab local viewer.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--demo", action="store_true", help="separate profile/pack/log in runs/demo/ + Start over")
    args = ap.parse_args()
    (ROOT / "runs").mkdir(exist_ok=True)
    logging.basicConfig(filename=ROOT / "runs" / "viewer.log", level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    if args.demo:
        STATE.update(demo=True, pack=DEMO_DIR / "pack.yaml", log=DEMO_DIR / "score_log.jsonl")
        prof.PROFILE_PATH = DEMO_DIR / "profile.yaml"
    pipeline.warm_up()                 # load once at start-up, then freeze for the garbage collector
    try:
        server = make_server(args.port)
    except OSError:
        print(f"Port {args.port} is already in use: another PrivLab viewer is probably running. Close it "
              f"(Ctrl+C in its window), or start this one with --port {args.port + 1}.", file=sys.stderr)
        return 1
    print(f"PrivLab viewer: http://127.0.0.1:{args.port}{'  (demo mode)' if args.demo else ''}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
