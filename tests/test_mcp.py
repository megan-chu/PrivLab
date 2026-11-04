"""MCP server: the tools, the chat flow without raw identifiers, refusals, paging, clean stdio. Fake Muse Spark."""
import asyncio
import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REPORT = "Patient: Mary Tan  NRIC: S1234567D\nDIAGNOSIS: Invasive ductal carcinoma, 2.1 cm.\n"
DRAFT = {"key": "tumour_size", "meaning_question": "Largest tumour dimension?",
         "definition": "The largest invasive tumour dimension on the resection specimen.", "type": "number",
         "unit": "cm", "range_min": 0.1, "range_max": 30, "choices": []}


@pytest.fixture
def mcp(fake_spark, local_keys, monkeypatch):
    import deid.profile as prof
    import extract.candidates as candidates
    import viewer.server as viewer
    from mcp_server import server
    monkeypatch.setattr(prof, "PROFILE_PATH", local_keys / "profile.yaml")
    monkeypatch.setitem(viewer.STATE, "pack", local_keys / "pack.yaml")
    monkeypatch.setitem(viewer.STATE, "log", local_keys / "log.jsonl")
    monkeypatch.setattr(candidates, "_is_blind", lambda *a: False)
    monkeypatch.setattr(server, "spark_state", fake_spark["state"], raising=False)
    monkeypatch.setitem(server.LAST_OUT, "folder", None)                 # no folder remembered from another test
    return server


def report_file(tmp: Path, text: str = REPORT, name: str = "mine.txt") -> str:
    p = tmp / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_six_tools_and_five_prompts(mcp):
    tools = asyncio.run(mcp.mcp.list_tools())
    assert [t.name for t in tools] == ["privlab_deidentify", "privlab_review", "privlab_fields", "privlab_extract",
                                       "privlab_wait", "privlab_export"]
    assert next(t for t in tools if t.name == "privlab_fields").annotations.destructive_hint   # it can remove a field
    assert len(asyncio.run(mcp.mcp.list_prompts())) == 5


def test_chat_flow_never_shows_a_raw_identifier(mcp, local_keys):
    out = mcp.privlab_deidentify(report_file(local_keys))
    assert out["next"] == "review" and out["total_count"] == 2 and "say" in out
    seen = json.dumps(out)
    out = mcp.privlab_review(out["deid_id"])
    assert out["next"] == "done" and "[PATIENT-1]" in out["text"]
    seen += json.dumps(out)
    did = out["deid_id"]

    mcp.spark_state["reply"] = DRAFT
    assert mcp.privlab_fields("draft", name="tumour size")["next"] == "ask"        # its meaning and an example first
    draft = mcp.privlab_fields("draft", name="tumour size", description="largest tumour size", example="2.1 cm")
    assert draft["next"] == "confirm" and "Did you mean" in draft["say"]
    mcp.spark_state["reply"] = {"fields": [{"key": "tumour_size", "candidates": [
        {"value_text": "2.1", "unit_text": "cm", "evidence_text": "Invasive ductal carcinoma, 2.1 cm",
         "reason": "diagnosis line", "confidence": 0.9}]}]}
    saved = mcp.privlab_fields("save", name="tumour size", **{k: draft["draft"][k] for k in (
        "definition", "type", "unit", "range_min", "range_max")}, meaning=draft["draft"]["meaning_question"])
    assert saved["field"]["version"] == 1 and saved["next"] == "preview"
    assert saved["preview"]["value"] == 2.1 and "Invasive ductal carcinoma" in saved["say"]   # shown on a report
    seen += json.dumps(saved)
    got = mcp.privlab_extract(did)
    assert next(r for r in mcp.viewer.api_session(None)["reports"] if r["deid_id"] == did)["read"] == 1   # the page sees it
    field = got["fields"][0]
    assert field["status"] == "green" and field["value"] == 2.1
    done = mcp.privlab_extract(did, [mcp.Answer(field="tumour_size", candidate_id=field["candidate_id"])])
    assert done["next"] == "done" and done["saved"][0]["outcome"] == "correct"      # all answers in one call
    seen += json.dumps(got) + json.dumps(done) + json.dumps(mcp.privlab_fields("list"))
    assert "Mary Tan" not in seen and "S1234567D" not in seen                   # nothing raw reached the chat

    csv_out = mcp.privlab_export("data", str(local_keys / "out" / "data.csv"))
    assert csv_out["rows"] == 1 and "carcinoma" not in Path(csv_out["path"]).read_text(encoding="utf-8")
    zip_out = mcp.privlab_export("masked", str(local_keys / "out" / "masked.zip"))
    with zipfile.ZipFile(zip_out["path"]) as z:
        assert "Mary Tan" not in z.read(z.namelist()[0]).decode("utf-8")


def test_refusals_are_plain_messages(mcp, local_keys):
    assert "not supported" in mcp.privlab_deidentify(report_file(local_keys, name="scan.png"))["say"]
    assert "No file" in mcp.privlab_deidentify(str(local_keys / "missing.txt"))["say"]
    assert "held-out test" in mcp.privlab_deidentify(report_file(local_keys, name="PP0008.txt"))["say"]
    assert mcp.privlab_extract("d_unknown")["next"] == "error"
    assert "data/" in mcp.privlab_export("data", str(ROOT / "data" / "x.csv"))["say"]
    assert mcp.privlab_fields("remove", name="nope")["next"] == "error"


def test_paging_resume_and_the_viewer_link(mcp, local_keys):
    out = mcp.privlab_deidentify(report_file(local_keys), limit=1)
    assert (len(out["items"]), out["has_more"], out["next_offset"]) == (1, True, 1)
    rest = mcp.privlab_deidentify(deid_id=out["deid_id"], offset=1, limit=1)
    assert rest["next"] == "review" and len(rest["items"]) == 1 and not rest["has_more"]
    assert rest["items"][0]["id"] != out["items"][0]["id"]
    assert mcp.privlab_deidentify()["next"] == "start"                       # no arguments: how to start, and the link


def test_same_file_again_continues_and_removed_reports_leave_the_session(mcp, local_keys):
    import viewer.server as viewer
    first = mcp.privlab_deidentify(report_file(local_keys))
    again = mcp.privlab_deidentify(report_file(local_keys))
    assert again["deid_id"] == first["deid_id"] and "already started" in again["say"]
    folder = mcp.privlab_deidentify(str(local_keys))
    assert [r["deid_id"] for r in folder["reports"] if r["report"] == "mine"] == [first["deid_id"]]
    row = next(r for r in viewer.api_session(None)["reports"] if r["deid_id"] == first["deid_id"])
    assert (row["step"], row["saved"]) == ("review", 0)                  # what the page compares with its own
    mcp.privlab_review(first["deid_id"])
    row = next(r for r in viewer.api_session(None)["reports"] if r["deid_id"] == first["deid_id"])
    assert row["step"] == "done"                                         # done in the chat: the page catches up
    waited = asyncio.run(mcp.privlab_wait([first["deid_id"]], seconds=1))    # no fields: done once de-identified
    assert waited["next"] == "done" and waited["finished"][0]["deid_id"] == first["deid_id"]
    viewer.api_remove({"deid_ids": [first["deid_id"]]})
    assert first["deid_id"] not in [r["deid_id"] for r in viewer.api_session(None)["reports"]]
    assert mcp.privlab_deidentify(report_file(local_keys))["deid_id"] != first["deid_id"]   # removed: starts afresh


def test_a_big_folder_names_what_was_not_processed_and_continues(mcp, local_keys, monkeypatch):
    monkeypatch.setattr(mcp, "MAX_FOLDER", 2)
    for i in range(1, 6):
        report_file(local_keys, REPORT, f"PP{i:04d}.txt")
    first = mcp.privlab_deidentify(str(local_keys))
    assert len(first["reports"]) == 2 and first["not_processed"] == ["PP0003.txt", "PP0004.txt", "PP0005.txt"]
    assert "PP0003 to PP0005" in first["say"]
    second = mcp.privlab_deidentify(str(local_keys))                     # again: the next ones, the first kept
    assert sum(not r.get("already_started") for r in second["reports"]) == 2 and second["not_processed"] == ["PP0005.txt"]
    assert mcp._names(["b.txt", "a7.txt", "x.pdf", "y.txt"]) == "b, a7, x and 1 more"


def test_a_missing_key_is_named_and_the_second_opinion_only_named_when_it_ran(mcp, local_keys, monkeypatch):
    done = mcp.privlab_review(mcp.privlab_deidentify(report_file(local_keys))["deid_id"])
    monkeypatch.setenv("META_API_KEY", "")
    d = mcp.privlab_fields("draft", name="tumour size", description="largest tumour size", example="2.1 cm")
    assert "META_API_KEY" in d["say"] and "Nothing was sent" in d["say"]
    mcp.privlab_fields("save", name="tumour size", meaning=DRAFT["meaning_question"], type="number", unit="cm",
                       definition=DRAFT["definition"], range_min=0.1, range_max=30)
    assert "META_API_KEY" in mcp.privlab_extract(done["deid_id"])["fields"][0]["reason"]
    monkeypatch.setenv("META_API_KEY", "test-key")
    monkeypatch.setenv("PRIVLAB_SECOND_OPINION", "on")
    monkeypatch.setenv("BASETEN_DECISION_API_KEY", "")                       # no decision model: never named
    mcp.spark_state["reply"] = {"fields": [{"key": "tumour_size", "candidates": [
        {"value_text": "2.1", "unit_text": "cm", "evidence_text": "Invasive ductal carcinoma, 2.1 cm",
         "reason": "diagnosis line", "confidence": 0.9}]}]}
    got = mcp.privlab_extract(mcp.privlab_review(mcp.privlab_deidentify(
        report_file(local_keys, REPORT + "Two.", name="two.txt"))["deid_id"])["deid_id"])
    assert got["fields"][0]["value"] is not None and "elephant" not in got["say"]


def test_a_report_from_the_chat_shows_its_original_file_and_only_that_file(mcp, local_keys):
    import viewer.server as viewer
    path = report_file(local_keys)
    did = mcp.privlab_deidentify(path)["deid_id"]
    assert viewer.api_open({"deid_id": did})["has_original"] is True          # the page shows the Original button
    body, ctype = viewer.original_file(did)
    assert body == Path(path).read_bytes() and ctype.startswith("text/plain")  # the file itself, as it is on disk
    assert viewer.original_file("d_unknown") is None                           # never any other path
    added = viewer.api_add({"name": "pasted.txt", "text": REPORT})                # added on the page: kept for a reload
    assert viewer.api_open({"deid_id": added["deid_id"]})["from_chat"] is False
    assert viewer.original_file(added["deid_id"])[0] == REPORT.encode("utf-8")


def test_replies_say_what_was_sent_where_values_come_from_and_allow_undo(mcp, local_keys):
    first = mcp.privlab_deidentify(report_file(local_keys))
    kept = mcp.privlab_review(first["deid_id"], keep_ids=[first["items"][0]["id"]])       # kept one visible
    assert kept["next"] == "done" and "start_over=True" in kept["say"] and "only after they chose the fields" in kept["say"]
    again = mcp.privlab_review(first["deid_id"], start_over=True)                        # changed their mind
    assert again["next"] == "review" and again["deid_id"] != first["deid_id"]
    assert first["deid_id"] not in mcp.pipeline._RUNS                                    # never exported twice
    done = mcp.privlab_review(again["deid_id"])
    mcp.spark_state["reply"] = DRAFT
    d = mcp.privlab_fields("draft", name="tumour size", description="largest tumour size", example="2.1 cm")
    assert "Only the field's name" in d["say"]
    mcp.privlab_fields("save", name="tumour size", meaning=d["draft"]["meaning_question"], type="number", unit="cm",
                       definition=d["draft"]["definition"], range_min=0.1, range_max=30)
    mcp.spark_state["reply"] = {"fields": [{"key": "tumour_size", "candidates": [
        {"value_text": "2.1", "unit_text": "cm", "evidence_text": "Invasive ductal carcinoma, 2.1 cm",
         "reason": "diagnosis line", "confidence": 0.9}]}]}
    got = mcp.privlab_extract(done["deid_id"])
    assert "Only the masked text was sent" in got["say"] and "Muse Spark 1.3" in got["fields"][0]["source"]
    second = mcp.privlab_deidentify(report_file(local_keys, REPORT + "More.", name="two.txt"))  # not finished
    out = mcp.privlab_export("data", str(local_keys / "out" / "d.csv"))
    assert "still being reviewed" in out["say"] and "two" in out["say"]                  # named, not silently left out
    assert str(local_keys / "out") in mcp.privlab_export("masked")["suggested_path"]    # the folder used last


def test_stdio_server_starts_cleanly_from_another_folder(tmp_path):
    """stdout is the protocol: any stray print would break listing the tools."""
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "mcp_server" / "server.py")],
                                   env={"PRIVLAB_MCP_VIEWER": "off"}, cwd=str(tmp_path))

    async def names():
        async with Client(params) as client:
            return [t.name for t in (await client.list_tools()).tools]
    assert len(asyncio.run(names())) == 6


def test_chat_context_shows_only_tags_and_report_words(mcp, local_keys):
    text = "Patient: Mary Tan\nSigned out by Oyelaran: agree\nIBAN: GB82 WEST 1234 5698 7654 32\nSeen by 陈小明 and Dr Lee.\n"
    out = mcp.privlab_deidentify(report_file(local_keys, text))
    seen = json.dumps(out, ensure_ascii=False)
    assert all(x not in seen for x in ("Oyelaran", "7654", "陈小明", "Mary", "Lee"))  # red-team round 2, M01-M06
    assert "[PATIENT-1]" in seen


def test_hide_text_after_the_review_step_is_refused_not_dropped(mcp, local_keys):
    out = mcp.privlab_deidentify(report_file(local_keys, "Report handed to Lum Wai at the counter.\n"))
    pipeline_run = __import__("deid.pipeline", fromlist=["_RUNS"])._RUNS[out["deid_id"]]
    pipeline_run["step"] = "hold"                                        # as if the hold question were open
    pipeline_run["holds"] = []
    res = mcp.privlab_review(out["deid_id"], hide_text=["Dr Wong"])
    assert res["next"] == "hold" and "first review" in res["error"] and pipeline_run["step"] == "hold"


def test_a_new_report_opens_in_the_browser_once(mcp, local_keys, monkeypatch):
    opened = []
    monkeypatch.setitem(mcp.VIEWER, "url", "http://127.0.0.1:8765/")
    monkeypatch.setattr(mcp.threading, "Timer", lambda delay, fn, args: type("T", (), {"start": lambda s: opened.append(args[0])})())
    out = mcp.privlab_deidentify(report_file(local_keys))
    assert opened == [f"http://127.0.0.1:8765/#{out['deid_id']}"]
    mcp.privlab_deidentify(deid_id=out["deid_id"])                          # asking where it is up to: no new tab
    assert len(opened) == 1


def test_status_of_a_finished_report_never_carries_the_original(mcp, local_keys):
    out = mcp.privlab_deidentify(report_file(local_keys))
    mcp.privlab_review(out["deid_id"])
    status = mcp.privlab_deidentify(deid_id=out["deid_id"])                  # the assistant re-checking
    assert status["next"] == "done" and "original" not in status and "Mary Tan" not in json.dumps(status)


def test_a_folder_in_bulk_verified_values_alone_and_a_suggested_save_folder(mcp, local_keys, monkeypatch):
    folder = local_keys / "batch"
    folder.mkdir()
    report_file(folder, REPORT, "a.txt")
    report_file(folder, "Patient: John Lim  NRIC: S1234567D\nDIAGNOSIS: carcinoma, 3.0 cm.\n", "b.txt")
    (folder / "photo.png").write_bytes(b"x")                                 # not a report: left out
    batch = mcp.privlab_deidentify(str(folder))
    assert batch["next"] == "review_batch" and [r["report"] for r in batch["reports"]] == ["a", "b"]
    assert "Mary Tan" not in json.dumps(batch) and "John Lim" not in json.dumps(batch)
    for r in batch["reports"]:
        assert mcp.privlab_review(r["deid_id"])["next"] == "done"

    mcp.spark_state["reply"] = DRAFT
    d = mcp.privlab_fields("draft", name="tumour size", description="largest tumour size", example="2.1 cm")["draft"]
    mcp.privlab_fields("save", name="tumour size", meaning=d["meaning_question"], type="number", unit="cm",
                       definition=d["definition"], range_min=0.1, range_max=30)
    mcp.spark_state["reply"] = {"fields": [{"key": "tumour_size", "candidates": [
        {"value_text": "2.1", "unit_text": "cm", "evidence_text": "Invasive ductal carcinoma, 2.1 cm",
         "reason": "diagnosis line", "confidence": 0.9}]}]}
    opened = []
    monkeypatch.setattr(mcp, "_show", opened.append)
    monkeypatch.setitem(mcp._REVIEW_OPENED, "at", 0.0)
    first = mcp.privlab_extract(batch["reports"][0]["deid_id"], auto_accept_verified=True)
    mcp.privlab_extract(batch["reports"][1]["deid_id"], auto_accept_verified=True)
    assert opened == [batch["reports"][0]["deid_id"]] and "opened" in first["say"]   # one tab for the batch
    assert first["auto_saved"] == [] and first["fields"][0]["status"] == "green"   # still learning: the user's OK
    assert first["next"] == "confirm" and not Path(mcp.viewer.STATE["log"]).exists()
    assert mcp.privlab_fields("skip", name="tumour size")["next"] == "done"         # unticked, as in the browser
    assert mcp.viewer.api_skip(None)["skip"] == ["tumour_size"]
    assert mcp.privlab_extract(batch["reports"][1]["deid_id"])["next"] == "define"   # nothing chosen to extract
    assert not mcp.privlab_fields("list")["items"][0]["chosen"]
    mcp.privlab_fields("use", name="tumour size")

    ask = mcp.privlab_export("data")
    assert ask["next"] == "ask" and "PrivLab output" in ask["suggested_path"] and str(folder) in ask["suggested_path"]
    out = mcp.privlab_export("data", ask["suggested_path"])
    assert out["next"] == "done" and Path(out["path"]).parent.name == "PrivLab output"
