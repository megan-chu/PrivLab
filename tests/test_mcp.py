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
    return server


def report_file(tmp: Path, text: str = REPORT, name: str = "mine.txt") -> str:
    p = tmp / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_ten_tools_and_three_prompts(mcp):
    tools = asyncio.run(mcp.mcp.list_tools())
    assert sorted(t.name for t in tools) == sorted([
        "privlab_deidentify_report", "privlab_get_report", "privlab_confirm_review", "privlab_define_field",
        "privlab_list_fields", "privlab_remove_field", "privlab_extract_fields", "privlab_confirm_field",
        "privlab_export_file", "privlab_open_viewer"])
    assert next(t for t in tools if t.name == "privlab_remove_field").annotations.destructive_hint
    assert len(asyncio.run(mcp.mcp.list_prompts())) == 3


def test_chat_flow_never_shows_a_raw_identifier(mcp, local_keys):
    out = mcp.privlab_deidentify_report(report_file(local_keys))
    assert out["next"] == "review" and out["total_count"] == 2 and "say" in out
    seen = json.dumps(out)
    out = mcp.privlab_confirm_review(out["deid_id"])
    assert out["next"] == "done" and "[PATIENT-1]" in out["text"]
    seen += json.dumps(out)
    did = out["deid_id"]

    mcp.spark_state["reply"] = DRAFT
    draft = mcp.privlab_define_field("tumour size")
    assert draft["next"] == "confirm" and "Did you mean" in draft["say"]
    saved = mcp.privlab_define_field("tumour size", save=True, **{k: draft["draft"][k] for k in (
        "definition", "type", "unit", "choices", "range_min", "range_max")}, meaning=draft["draft"]["meaning_question"])
    assert saved["field"]["version"] == 1

    mcp.spark_state["reply"] = {"fields": [{"key": "tumour_size", "candidates": [
        {"value_text": "2.1", "unit_text": "cm", "evidence_text": "Invasive ductal carcinoma, 2.1 cm",
         "reason": "diagnosis line", "confidence": 0.9}]}]}
    got = mcp.privlab_extract_fields(did)
    field = got["fields"][0]
    assert field["status"] == "green" and field["value"] == 2.1
    done = mcp.privlab_confirm_field(did, "tumour_size", candidate_id=field["candidate_id"])
    assert done["outcome"] == "correct"
    seen += json.dumps(got) + json.dumps(done) + json.dumps(mcp.privlab_list_fields())
    assert "Mary Tan" not in seen and "S1234567D" not in seen                   # nothing raw reached the chat

    csv_out = mcp.privlab_export_file("registry", str(local_keys / "out" / "registry.csv"))
    assert csv_out["rows"] == 1 and "carcinoma" not in Path(csv_out["path"]).read_text(encoding="utf-8")
    zip_out = mcp.privlab_export_file("masked", str(local_keys / "out" / "masked.zip"))
    with zipfile.ZipFile(zip_out["path"]) as z:
        assert "Mary Tan" not in z.read(z.namelist()[0]).decode("utf-8")


def test_refusals_are_plain_messages(mcp, local_keys):
    assert "not supported" in mcp.privlab_deidentify_report(report_file(local_keys, name="scan.png"))["say"]
    assert "No file" in mcp.privlab_deidentify_report(str(local_keys / "missing.txt"))["say"]
    assert "held-out test" in mcp.privlab_deidentify_report(report_file(local_keys, name="PP0008.txt"))["say"]
    assert mcp.privlab_extract_fields("d_unknown")["next"] == "error"
    assert "data/" in mcp.privlab_export_file("registry", str(ROOT / "data" / "x.csv"))["say"]
    assert mcp.privlab_remove_field("nope")["next"] == "error"


def test_paging_and_resume(mcp, local_keys):
    out = mcp.privlab_deidentify_report(report_file(local_keys), limit=1)
    assert (len(out["items"]), out["has_more"], out["next_offset"]) == (1, True, 1)
    rest = mcp.privlab_get_report(out["deid_id"], offset=1, limit=1)
    assert rest["next"] == "review" and len(rest["items"]) == 1 and not rest["has_more"]
    assert rest["items"][0]["id"] != out["items"][0]["id"]


def test_stdio_server_starts_cleanly_from_another_folder(tmp_path):
    """stdout is the protocol: any stray print would break listing the tools."""
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "mcp_server" / "server.py")],
                                   env={"PRIVLAB_MCP_VIEWER": "off"}, cwd=str(tmp_path))

    async def names():
        async with Client(params) as client:
            return [t.name for t in (await client.list_tools()).tools]
    assert len(asyncio.run(names())) == 10
