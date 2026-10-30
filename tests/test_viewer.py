"""Viewer server: uploads, prefetch and the threaded server. Fake Muse Spark only."""
import json
import threading
import time
import urllib.request

import pytest

from deid import pipeline
from deid.profile import new_profile
from extract import learn
from extract.candidates import find_candidates, prefetch
from extract.define import define_field

DRAFT = {"key": "tumour_size", "meaning_question": "Largest tumour dimension?", "type": "number",
         "unit": "cm", "range_min": 0.1, "range_max": 30, "choices": []}
REPORT = "DIAGNOSIS: Invasive ductal carcinoma, 2.1 cm. GROSS: specimen 9 x 6 x 3 cm."


@pytest.fixture
def viewer(fake_spark, local_keys, monkeypatch):
    import deid.profile as prof
    import viewer.server as server
    monkeypatch.setattr(prof, "PROFILE_PATH", local_keys / "profile.yaml")
    monkeypatch.setitem(server.STATE, "pack", local_keys / "pack.yaml")
    monkeypatch.setitem(server.STATE, "log", local_keys / "log.jsonl")
    return server


def test_add_normalises_line_endings_and_needs_a_profile_first(viewer):
    out = viewer.api_add({"name": "mine.txt", "text": "Patient: Mary Tan\r\nDIAGNOSIS: carcinoma\r"})
    assert out["next"] == "profile"                                          # first use: the checklist
    out = viewer.api_add({"name": "mine.txt", "text": "Patient: Mary Tan\r\nDIAGNOSIS: carcinoma\r", "choices": {}})
    assert out["next"] == "review" and "\r" not in out["text"] and out["name"] == "mine"


@pytest.mark.parametrize("body, message", [
    ({"text": "   "}, "empty"),
    ({"text": "x" * 200_001}, "too long"),
    ({"name": "PP0008.txt", "text": "some text"}, "held-out test report"),
])
def test_add_refuses_bad_input(viewer, body, message):
    assert message in viewer.api_add(body)["error"]


def test_prefetch_makes_the_one_call_and_freezes_no_prediction(viewer, fake_spark):
    pack = viewer.STATE["pack"]
    fake_spark["state"]["reply"] = DRAFT
    define_field("tumour size", None, accept=DRAFT, pack_path=pack)
    r = pipeline.deidentify_report(text=REPORT, profile=new_profile()[0])
    pipeline.confirm_deid(r["deid_id"], use_spark=False)
    fake_spark["state"]["reply"] = {"fields": [{"key": "tumour_size", "candidates": [
        {"value_text": "2.1", "unit_text": "cm", "evidence_text": "Invasive ductal carcinoma, 2.1 cm",
         "reason": "t", "confidence": 0.9}]}]}
    sent = len(fake_spark["sent"])
    assert prefetch(r["deid_id"], pack_path=pack)["status"] == "ready"
    assert len(fake_spark["sent"]) == sent + 1
    assert (r["deid_id"], "tumour_size") not in learn._PREDICTIONS             # review timer not started
    res = find_candidates(r["deid_id"], "tumour_size", pack_path=pack)
    assert len(fake_spark["sent"]) == sent + 1 and res["status"] == "green"    # instant, no second call


def test_threaded_server_answers_while_a_model_call_runs(viewer, monkeypatch):
    def slow_find(_):
        time.sleep(1.0)
        return {"status": "green"}
    monkeypatch.setitem(viewer.POST, "/api/find", slow_find)
    srv = viewer.make_server(0)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/find", b"{}", {"Content-Type": "application/json"})
        threading.Thread(target=lambda: urllib.request.urlopen(req).read(), daemon=True).start()
        time.sleep(0.1)
        t0 = time.perf_counter()
        pack = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/pack"))
        assert "fields" in pack and time.perf_counter() - t0 < 0.5           # not stuck behind the slow call
    finally:
        srv.shutdown()
