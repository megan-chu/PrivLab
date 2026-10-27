"""Raw text must not reach Muse Spark. None of these tests call the API."""
import ast

import pytest

from common.config import ROOT
from common.spark import PhiLeakError, call, check_no_leak
from deid.types import MaskedText


def test_plain_string_refused():
    with pytest.raises(TypeError):
        call("Patient: Mary Tan", instructions="x", schema=None, step="test")


def test_identifier_left_in_masked_text_refused():
    leaky = MaskedText(deid_id="d_test", text="Patient: [PATIENT-A]  NRIC: S1234567D", tags=())
    with pytest.raises(PhiLeakError, match="ID"):
        call(leaky, instructions="x", schema=None, step="test")


def test_tags_pass_leak_check():
    check_no_leak("Patient: [PATIENT-A]  NRIC: [ID-1]  DOB: [DOB-1]  Phone: [PHONE-1]")


def test_only_spark_module_imports_openai():
    importers = []
    for path in ROOT.rglob("*.py"):
        if ".venv" in path.parts or "tests" in path.parts:   # tests may import openai error types
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if any(n.split(".")[0] == "openai" for n in names):
                importers.append(path.relative_to(ROOT).as_posix())
    assert sorted(set(importers)) == ["common/spark.py"]


def test_only_smoke_test_uses_raw_call():
    users = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*.py")
             if ".venv" not in p.parts and "_call_raw" in p.read_text(encoding="utf-8")]
    assert sorted(users) == ["common/spark.py", "eval/smoke_spark.py", "tests/test_spark_guard.py"]


def test_request_path_with_fake_client(monkeypatch, tmp_path):
    """Fallback to json_object, schema validation, cache and usage log, without the real API."""
    import openai
    from types import SimpleNamespace

    from pydantic import BaseModel

    import common.spark as spark

    class Names(BaseModel):
        names: list[str]

    calls = []

    class FakeCompletions:
        def create(self, **kw):
            calls.append(kw["response_format"]["type"])
            if kw["response_format"]["type"] == "json_schema":
                raise openai.BadRequestError("no json_schema", response=SimpleNamespace(
                    request=None, status_code=400, headers={}), body=None)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='```json\n{"names": ["Mary Tan"]}\n```'))],
                usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10))

    monkeypatch.setenv("META_API_KEY", "test-key")
    monkeypatch.setattr(spark, "_client", SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())))
    monkeypatch.setattr(spark, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(spark, "USAGE_LOG", tmp_path / "usage.log")
    monkeypatch.setattr(spark, "USAGE_TOTALS", tmp_path / "usage.json")

    first = spark._call_raw("hand-written text", Names, step="t")
    second = spark._call_raw("hand-written text", Names, step="t")   # served from cache
    assert first.names == second.names == ["Mary Tan"]
    assert calls == ["json_schema", "json_object"]                  # one live call, after fallback
    log = (tmp_path / "usage.log").read_text(encoding="utf-8")
    assert "json_object\tlive\tin=50\tout=10" in log and "\tcached\t" in log
    assert "hand-written" not in log
