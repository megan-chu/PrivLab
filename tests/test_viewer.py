"""Viewer server: uploads, prefetch and the threaded server. Fake Muse Spark only."""
import base64
import json
import threading
import time
import urllib.request

import pytest

from deid import pipeline
from deid.profile import new_profile
from extract import learn
from extract.candidates import find_candidates, prefetch, preview
from extract.define import define_field
from test_pdf import REPORT as PDF_REPORT, make_pdf

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


def test_preview_reads_the_read_ahead_only(viewer, fake_spark, monkeypatch):
    import extract.candidates as cand
    monkeypatch.setattr(cand, "BLIND_EVERY", 10**9)                          # this field is not due a blind check
    pack = viewer.STATE["pack"]
    fake_spark["state"]["reply"] = DRAFT
    define_field("tumour size", None, accept=DRAFT, pack_path=pack)
    r = pipeline.deidentify_report(text=REPORT + " ", profile=new_profile()[0])
    pipeline.confirm_deid(r["deid_id"], use_spark=False)
    sent = len(fake_spark["sent"])
    assert preview(r["deid_id"], pack_path=pack)["status"] == "none"         # nothing read yet: no call made
    assert len(fake_spark["sent"]) == sent
    fake_spark["state"]["reply"] = {"fields": [{"key": "tumour_size", "candidates": [
        {"value_text": "2.1", "unit_text": "cm", "evidence_text": "Invasive ductal carcinoma, 2.1 cm",
         "reason": "t", "confidence": 0.9}]}]}
    prefetch(r["deid_id"], pack_path=pack)
    out = preview(r["deid_id"], pack_path=pack)
    assert len(fake_spark["sent"]) == sent + 1 and out["fields"]["tumour_size"][0]["value"] == 2.1
    assert (r["deid_id"], "tumour_size") not in learn._PREDICTIONS             # review timer not started


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


def test_deidentified_text_export_one_all_and_withheld(viewer):
    import io
    import zipfile
    ids = []
    for text in ("Patient: Mary Tan    MRN: H1088762\nDIAGNOSIS: carcinoma, 2.1 cm.\n",
                 "Patient: John Lim    MRN: H1088763\nDIAGNOSIS: adenoma.\n"):
        r = pipeline.deidentify_report(text=text, report_id="mine", profile=new_profile()[0])
        pipeline.confirm_deid(r["deid_id"], use_spark=False)
        ids.append(r["deid_id"])
    body, ctype, name = viewer.deid_download(ids[0])
    assert ctype.startswith("text/plain") and name == "mine_deidentified.txt"
    assert b"[PATIENT-1]" in body and b"Mary Tan" not in body
    pipeline._RUNS[ids[1]]["priority"] = {"level": "high", "reasons": ["1 identifier pattern(s) left in the output"]}
    body, ctype, _ = viewer.deid_download(None)
    names = zipfile.ZipFile(io.BytesIO(body)).namelist()
    assert ctype == "application/zip" and len([n for n in names if n.startswith("mine")]) >= 1
    assert viewer.deid_download(ids[1]) is None                                 # withheld: identifier left


def test_second_viewer_on_the_same_port_is_refused(viewer):
    srv = viewer.make_server(0)
    try:
        with pytest.raises(OSError):
            viewer.make_server(srv.server_address[1])
    finally:
        srv.server_close()


def test_viewer_adds_a_text_pdf_and_refuses_a_scan(viewer):
    pdf = base64.b64encode(make_pdf(PDF_REPORT, [])).decode()
    out = viewer.api_add({"name": "mine.pdf", "pdf": pdf, "choices": {}})
    assert out["next"] == "review" and "Mary Tan" in out["text"] and out["name"] == "mine"
    assert any("Page 2" in w for w in out["warnings"])
    assert any(d["text"] == "S1234567D" for d in out["detections"])            # de-identified as usual
    scan = viewer.api_add({"name": "scan.pdf", "pdf": base64.b64encode(make_pdf([])).decode(), "choices": {}})
    assert scan["refused"] and "not supported" in scan["error"]
    assert viewer.api_add({"name": "x.pdf", "pdf": "%%%", "choices": {}})["refused"]


def test_export_everything_in_one_zip(viewer, fake_spark):
    import io, zipfile
    out = viewer.api_add({"name": "a.txt", "text": "Patient: Mary Tan\nDIAGNOSIS: carcinoma.", "choices": {}})
    pipeline.confirm_deid(out["deid_id"], use_spark=False)
    with zipfile.ZipFile(io.BytesIO(viewer.export_all([out["deid_id"]]))) as z:
        names = z.namelist()
        assert "data_dictionary.txt" in names and "masked/a_deidentified.txt" in names
        assert "Mary Tan" not in z.read("masked/a_deidentified.txt").decode("utf-8")
    assert viewer.export_all(["d_not_listed"]) is None                        # only the listed reports
    define_field("Biomarker", None, accept={"key": "biomarker", "meaning_question": "Biomarkers?", "type": "text",
                 "unit": None, "range_min": None, "range_max": None, "multi": True}, pack_path=viewer.STATE["pack"])
    fake_spark["state"]["reply"] = {"fields": [{"key": "biomarker", "candidates": [
        {"value_text": "DIAGNOSIS: carcinoma", "unit_text": None, "evidence_text": "DIAGNOSIS: carcinoma", "reason": "t", "confidence": 0.9}]}]}
    res = viewer.api_find({"deid_id": out["deid_id"], "field": "biomarker"})
    viewer.api_confirm_field({"deid_id": out["deid_id"], "field": "biomarker", "items": [{"value": i["value"], "evidence_text": i["evidence_text"]} for i in res["items"]]})
    with zipfile.ZipFile(io.BytesIO(viewer.export_all([out["deid_id"]]))) as z:
        assert "lists/biomarker.csv" in z.namelist() and "lists/biomarker.csv" in z.read("data_dictionary.txt").decode("utf-8")
        assert z.read("lists/biomarker.csv").decode("utf-8-sig").splitlines()[0] == "case_id,report_id,Biomarker,result"


def test_export_links_without_ids_export_nothing(viewer):
    from urllib.parse import urlparse
    out = viewer.api_add({"name": "a.txt", "text": "Patient: Mary Tan\nDIAGNOSIS: carcinoma.", "choices": {}})
    pipeline.confirm_deid(out["deid_id"], use_spark=False)
    assert viewer._ids(urlparse("/api/export_all")) == [] and viewer._ids(urlparse("/api/export_all?ids=")) == []
    assert viewer.export_all([]) is None and viewer.deid_download(None, []) is None     # a copied link, no list
    assert viewer.export_all([out["deid_id"]]) is not None
    assert viewer.export_csv([]).decode("utf-8-sig").count("\n") == 1                   # header only


def test_download_summary_names_what_is_left_out_and_why(viewer, fake_spark):
    pack = viewer.STATE["pack"]
    fake_spark["state"]["reply"] = DRAFT
    define_field("tumour size", None, accept=DRAFT, pack_path=pack)
    done = viewer.api_add({"name": "a.txt", "text": REPORT, "choices": {}})            # de-identified and a field confirmed
    pipeline.confirm_deid(done["deid_id"], use_spark=False)
    fake_spark["state"]["reply"] = {"fields": [{"key": "tumour_size", "candidates": [
        {"value_text": "2.1", "unit_text": "cm", "evidence_text": "Invasive ductal carcinoma, 2.1 cm", "reason": "t", "confidence": 0.9}]}]}
    res = viewer.api_find({"deid_id": done["deid_id"], "field": "tumour_size"})
    viewer.api_confirm_field({"deid_id": done["deid_id"], "field": "tumour_size", "candidate_id": res["candidate_id"]})
    masked_only = viewer.api_add({"name": "b.txt", "text": "Patient: John Lim\nDIAGNOSIS: adenoma.", "choices": {}})
    pipeline.confirm_deid(masked_only["deid_id"], use_spark=False)                 # de-identified, nothing confirmed
    in_review = viewer.api_add({"name": "c.txt", "text": "Patient: Mary Tan\nDIAGNOSIS: carcinoma.", "choices": {}})
    ids = [done["deid_id"], masked_only["deid_id"], in_review["deid_id"]]

    csv = viewer.export_summary("csv", ids)
    assert csv["included"] == [done["deid_id"]]
    assert {(x["deid_id"], x["why"]) for x in csv["excluded"]} == {
        (masked_only["deid_id"], "no field confirmed yet"), (in_review["deid_id"], "not de-identified yet")}
    zipped = viewer.export_summary("deid", ids)
    assert zipped["included"] == ids[:2] and zipped["excluded"] == [{"deid_id": in_review["deid_id"], "why": "not de-identified yet"}]
    pipeline._RUNS[masked_only["deid_id"]]["priority"] = {"level": "high", "reasons": ["1 identifier pattern(s) left in the output"]}
    one = viewer.export_summary("one", deid_id=masked_only["deid_id"])
    assert one["included"] == [] and one["excluded"][0]["why"] == "possible identifier left in the text"
    assert viewer.export_summary("all", ids)["included"] == [done["deid_id"]]
    assert "Mary Tan" not in json.dumps(csv)                                           # ids and reasons only


def test_learned_view_lists_and_forgets_with_the_source_report(viewer, fake_spark):
    """Hide a name on report 1: the profile remembers it came from report 1, report 2 says so, and Remove undoes it."""
    text = "Patient: Mary Tan\nThe nurse asked that zola mensah receive a copy.\n"
    out = viewer.api_add({"name": "report_1.txt", "text": text, "choices": {}})
    held = pipeline.confirm_deid(out["deid_id"], add=[{"text": "zola mensah", "type": "NAME"}], use_spark=False)
    if held["next"] == "hold":
        held = pipeline.confirm_deid(out["deid_id"], hold_answers={"keep": []}, use_spark=False)
    assert {"text": "zola mensah", "type": "NAME"} in held["learned"]["hide"]
    learned = viewer.api_learned(None)
    assert {"text": "zola mensah", "type": "NAME", "from": "report_1"} in learned["hide"]
    again = viewer.api_add({"name": "report_2.txt", "text": "Seen while zola mensah watched.\n"})
    found = next(d for d in again["detections"] if d["text"].lower() == "zola mensah")
    assert found["learned"] and found["learned_from"] == "report_1"                     # caught on the laptop
    assert viewer.api_forget({"kind": "hide", "text": "zola mensah", "type": "NAME"})["ok"]
    assert viewer.api_learned(None)["hide"] == []
    assert "error" in viewer.api_forget({"kind": "hide", "text": "zola mensah", "type": "NAME"})


def test_export_only_the_ticked_fields(viewer, fake_spark):
    pack = viewer.STATE["pack"]
    fake_spark["state"]["reply"] = DRAFT
    define_field("tumour size", None, accept=DRAFT, pack_path=pack)
    define_field("margin", None, accept={"key": "margin", "meaning_question": "Margin?", "type": "text", "unit": None,
                 "range_min": None, "range_max": None}, pack_path=pack)
    assert viewer.export_csv([], ["margin"]).decode("utf-8-sig").splitlines()[0] == "case_id,report_id,reviewed_at,exported_at,margin"
    assert "tumour size" in viewer.export_csv([]).decode("utf-8-sig").splitlines()[0]    # no ?fields: every field


def test_privacy_choices_can_change_after_first_use(viewer):
    out = viewer.api_add({"name": "a.txt", "text": "Seen at Riverside General Hospital. Patient: Mary Tan", "choices": {}})
    assert out["next"] == "review" and any(d["type"] == "HOSPITAL" for d in out["detections"])   # defaults hide it
    prof = viewer.api_profile_choices({"choices": {"hospital": False}})
    assert prof["remove"]["hospital"] is False and prof["exists"]
    again = viewer.api_add({"name": "a.txt", "text": "Seen at Riverside General Hospital. Patient: Mary Tan"})
    assert not any(d["type"] == "HOSPITAL" for d in again["detections"])                      # now kept


@pytest.mark.parametrize("err", [ConnectionAbortedError, ConnectionResetError, BrokenPipeError])
def test_browser_hangup_during_reply_is_silent(err, capsys):
    import viewer.server as server
    srv = object.__new__(server.ExclusiveServer)          # no port needed: handle_error only
    try:
        raise err("[WinError 10053] aborted")
    except err:
        srv.handle_error(None, ("127.0.0.1", 0))          # must not raise or print
    assert capsys.readouterr().err == ""


def test_other_server_errors_still_print(capsys):
    import viewer.server as server
    srv = object.__new__(server.ExclusiveServer)
    try:
        raise ValueError("real bug")
    except ValueError:
        srv.handle_error(None, ("127.0.0.1", 0))
    assert "Traceback" in capsys.readouterr().err


def test_only_this_page_on_this_computer_may_call_the_api():
    from viewer.server import _local
    assert _local("127.0.0.1:8765", None) and _local("localhost:8765", "http://localhost:8765")
    assert not _local("127.0.0.1:8765", "https://evil.example")          # another site's page
    assert not _local("rebind.example:8765", None)                      # DNS rebinding


def test_download_names_survive_any_characters():
    from viewer.server import _attachment
    h = _attachment('病理报告 – "Tan".txt')
    h.encode("latin-1")                                                  # http.server sends latin-1 headers
    assert "filename*=UTF-8''" in h and '"' not in h.split('filename="')[1].split('"')[0]


def test_the_three_demo_samples_load_and_hide_their_identifiers(viewer):
    for rid in ("demo_breast", "demo_lung", "demo_colon"):
        out = viewer.api_deid({"report_id": rid, "choices": {}})
        assert out["next"] == "review" and out["sample"], rid
        found = {d["text"] for d in out["detections"]}
        assert {"demo_breast": "Goh Mei Ling", "demo_lung": "Rahman, Daniel Arif", "demo_colon": "Chua Li Hua"}[rid] in found
