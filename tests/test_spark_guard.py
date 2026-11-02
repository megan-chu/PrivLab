"""Raw text must not reach Muse Spark. None of these tests call the API."""
import ast

import pytest

from common.config import ROOT
from common.spark import PhiLeakError, call, check_no_leak
from deid.types import MaskedText


def test_plain_string_refused():
    with pytest.raises(TypeError):
        call("Patient: Mary Tan", instructions="x", schema=None, step="test")


def test_identifier_left_in_masked_text_refused(local_keys):
    from deid.seal import make_seal
    text = "Patient: [PATIENT-1]  NRIC: S1234567D"
    leaky = MaskedText(deid_id="d_test", text=text, tags=(), seal=make_seal("d_test", text))
    with pytest.raises(PhiLeakError, match="ID"):
        call(leaky, instructions="x", schema=None, step="test")


def test_unsealed_or_tampered_text_refused(local_keys):
    from deid.seal import make_seal
    forged = MaskedText(deid_id="d_test", text="Patient: Mary Tan", tags=())
    with pytest.raises(PhiLeakError, match="seal"):
        call(forged, instructions="x", schema=None, step="test")
    sealed = MaskedText(deid_id="d_test", text="[PATIENT-1]", tags=(), seal=make_seal("d_test", "[PATIENT-1]"))
    widened = MaskedText(deid_id="d_test", text=sealed.text, tags=(), seal=sealed.seal, kept_types=("NAME",))
    with pytest.raises(PhiLeakError, match="seal"):
        call(widened, instructions="x", schema=None, step="test")


def test_tags_pass_leak_check():
    check_no_leak("Patient: [PATIENT-1]  NRIC: [ID-1]  DOB: [DOB-1]  Phone: [PHONE-1]")


def test_only_spark_module_imports_openai():
    importers = []
    for path in ROOT.rglob("*.py"):
        if {".venv", "runs", "tests"} & set(path.parts):   # tests may import openai error types; runs/ is scratch
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if any(n.split(".")[0] == "openai" for n in names):
                importers.append(path.relative_to(ROOT).as_posix())
    assert sorted(set(importers)) == ["common/spark.py"]


def test_only_smoke_test_uses_raw_call():
    users = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*.py")
             if not {".venv", "runs"} & set(p.parts) and "_call_raw" in p.read_text(encoding="utf-8")]
    assert sorted(users) == ["common/spark.py", "eval/smoke_spark.py", "tests/test_spark_guard.py"]


def test_request_path_with_fake_client(monkeypatch, tmp_path, local_keys):
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
    monkeypatch.setattr(spark, "_clients", {"meta": SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))})
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
    for f in (tmp_path / "cache").rglob("*.enc"):
        f.write_bytes(b"locked with another vault key")                # e.g. after the key was changed
    third = spark._call_raw("hand-written text", Names, step="t")
    assert third.names == ["Mary Tan"] and len(calls) == 4             # asked again, not failed


def test_proxy_skips_the_probe_and_a_blank_url_is_a_config_error(monkeypatch):
    """Behind an HTTP proxy a direct connect fails although the API is reachable; an unset URL is not "offline"."""
    import socket
    from types import SimpleNamespace

    import common.spark as spark
    from common.config import load_config

    def no_network(*a, **k):
        raise OSError("blocked")
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(spark, "_REACH", {})
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.hospital.local:8080")
    spark._check_online(load_config(), "baseten")                              # no OfflineError
    monkeypatch.delenv("HTTPS_PROXY")
    with pytest.raises(ValueError, match="base URL"):
        spark._check_online(SimpleNamespace(meta_api_base_url="", baseten_base_url=""), "baseten")


def test_offline_fails_fast_and_is_remembered(monkeypatch):
    """No internet: an OfflineError at once (callers show "review manually"), not two 20-second timeouts."""
    import socket
    import time

    import common.spark as spark
    from common.config import load_config
    tries = []
    def no_network(*a, **k):
        tries.append(a)
        raise OSError("network is unreachable")
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(spark, "_REACH", {})
    t0 = time.perf_counter()
    for _ in range(3):
        with pytest.raises(spark.OfflineError):
            spark._check_online(load_config(), "baseten")
    assert time.perf_counter() - t0 < 1 and len(tries) == 1                 # checked once, then remembered
