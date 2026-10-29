"""
server.py - the local viewer: one HTML page plus a JSON API over the same functions the MCP
server calls (deid/pipeline.py, extract/). Standard library only; listens on 127.0.0.1.

Raw report text is shown only here, on the laptop. Muse Spark is called only by confirm steps
and "find", exactly as in the MCP flow. Reports: dev and extra only (test is never listed).

--demo keeps the privacy profile, pack and score log in runs/demo/, with a "Start over" button,
so a video can be re-recorded from a clean first use. Without it the real profile, pack and
runs/score_log.jsonl are used (Day 7 teaching).

Usage:
  python -m viewer.server            # http://127.0.0.1:8765
  python -m viewer.server --demo
"""
import argparse
import json
import logging
import shutil
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from common.config import ROOT
from common.data import REPORTS, load_report, manifest
from common.timing import timed
from deid import pipeline
from deid import profile as prof
from deid.hold import name_list, vocabulary
from extract import learn
from extract.candidates import find_candidates
from extract.define import define_field
from extract.pack import load_pack

PAGE = Path(__file__).with_name("index.html")
DEMO_DIR = ROOT / "runs" / "demo"
log = logging.getLogger("privlab.viewer")
STATE = {"pack": None, "log": None, "demo": False}


# --------------------------------------------------------------------------
# API: one function per action; each returns a JSON-able dict
# --------------------------------------------------------------------------
def api_reports(_):
    return {"reports": [{"id": r["report_id"], "split": r["split"], "cancer": r.get("cancer_type", "")}
                        for r in manifest() if r["split"] in ("dev", "extra")], "demo": STATE["demo"]}


def api_report(q):
    rid = q["id"]
    if not any(r["report_id"] == rid and r["split"] in ("dev", "extra") for r in manifest()):
        return {"error": "unknown report (only dev and extra are available here)"}
    return {"id": rid, "text": load_report(rid)}


def api_profile(_):
    p = prof.load_profile()
    return {"exists": p is not None, "checklist": prof.checklist(),
            "allow": len(p.allow_list) if p else 0, "learned": len(p.always_remove) if p else 0}


def api_deid(b):
    return pipeline.deidentify_report(str(REPORTS / f"{b['report_id']}.txt"), report_id=b["report_id"],
                                      profile_choices=b.get("choices"), show_text=True)


def api_confirm(b):
    return pipeline.confirm_deid(b["deid_id"], unmark=b.get("unmark", []), add=b.get("add", []), show_text=True)


def api_hold(b):
    return pipeline.confirm_deid(b["deid_id"], hold_answers={"keep": b.get("keep", [])}, show_text=True)


def api_flags(b):
    return pipeline.confirm_deid_flags(b["deid_id"], keep=b.get("keep", []), hide=b.get("hide", []))


def api_pack(_):
    return {"fields": [{"key": f.key, "name": f.name, "meaning": f.meaning, "type": f.type, "unit": f.unit,
                        "choices": f.choices, "readiness": f.readiness, "recent": f.recent}
                       for f in load_pack(STATE["pack"])["fields"]]}


def api_define(b):
    return define_field(b["name"], b.get("description"), accept=b.get("accept"), pack_path=STATE["pack"])


def api_find(b):
    return find_candidates(b["deid_id"], b["field"], pack_path=STATE["pack"])


def api_confirm_field(b):
    return learn.confirm_field(b["deid_id"], b["field"], candidate_id=b.get("candidate_id"), value=b.get("value"),
                               evidence_text=b.get("evidence_text"), missing=bool(b.get("missing")),
                               pack_path=STATE["pack"], log_path=STATE["log"])


def api_reset(_):
    if not STATE["demo"]:
        return {"error": "Start over is only available with --demo"}
    shutil.rmtree(DEMO_DIR, ignore_errors=True)
    pipeline._RUNS.clear()
    return {"ok": True}


GET = {"/api/reports": api_reports, "/api/report": api_report, "/api/profile": api_profile, "/api/pack": api_pack}
POST = {"/api/deid": api_deid, "/api/confirm": api_confirm, "/api/hold": api_hold, "/api/flags": api_flags,
        "/api/define": api_define, "/api/find": api_find, "/api/confirm_field": api_confirm_field,
        "/api/reset": api_reset}


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
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
        except Exception as e:      # never crash the page; details go to the log, not the browser
            log.error("%s failed: %s", fn.__name__, traceback.format_exc())
            out = {"error": f"{type(e).__name__}: something went wrong; see runs/viewer.log"}
        self._json(200, out)

    def _json(self, code, data):
        self._send(code, json.dumps(data, default=str).encode(), "application/json")

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        log.info(fmt, *args)


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
    vocabulary(), name_list()          # load once at start-up
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"PrivLab viewer: http://127.0.0.1:{args.port}{'  (demo mode)' if args.demo else ''}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
