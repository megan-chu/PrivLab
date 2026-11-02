"""Shared fixtures: throwaway local keys and temp folders, so tests never touch real vault/profile files."""
import secrets
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

Path("runs").mkdir(exist_ok=True)      # pytest.ini puts temp folders in runs/pytest; a fresh clone has no runs/ yet

# ----------------------------------------------------------------------------------------------------------------
# One progress bar instead of a dot per test (quiet runs only; -v keeps the usual list). A failure still shows as
# "F", with its details in the summary.
# ----------------------------------------------------------------------------------------------------------------
_BAR = {"done": 0, "total": 0, "on": False, "tw": None}


def pytest_collection_modifyitems(session, config, items):
    _BAR.update(total=len(items), on=config.option.verbose < 0,
                tw=config.pluginmanager.get_plugin("terminalreporter"))


def pytest_report_teststatus(report, config):
    if _BAR["on"] and report.when == "call" and report.passed:
        return "passed", "", "PASSED"                                   # no dot: the bar shows progress


def pytest_runtest_logfinish(nodeid, location):
    if not _BAR["on"] or not _BAR["total"]:
        return
    _BAR["done"] += 1
    done, total, width = _BAR["done"], _BAR["total"], 30
    fill = round(width * done / total)
    _BAR["tw"].write(f"\r{'█' * fill}{'░' * (width - fill)} {done * 100 // total:>3}%  ({done}/{total})", flush=True)
    if done == total:
        _BAR["tw"].write("\n")


@pytest.fixture
def local_keys(monkeypatch, tmp_path):
    import common.spark as spark
    import deid.pipeline as pipeline
    import deid.vault as vault

    monkeypatch.setenv("PRIVLAB_SEAL_KEY", secrets.token_hex(32))
    monkeypatch.setenv("PRIVLAB_VAULT_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(vault, "VAULT_DIR", tmp_path / "vault")
    monkeypatch.setattr(pipeline, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(spark, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(spark, "USAGE_LOG", tmp_path / "usage.log")
    monkeypatch.setattr(spark, "USAGE_TOTALS", tmp_path / "usage.json")
    return tmp_path


@pytest.fixture
def no_names_model(monkeypatch):
    """Rules only: for tests of the hold rule, whose names the local name model would hide earlier."""
    import deid.patterns as patterns

    monkeypatch.setattr(patterns, "find_names", lambda text: [])


@pytest.fixture
def fake_spark(monkeypatch, local_keys):
    """Replace the network request with a scripted reply; records every prompt sent."""
    import json

    import common.spark as spark

    sent = []
    state = {"reply": {"findings": []}, "fail": None}

    def fake_request(settings, provider, prompt, schema, effort):
        if state["fail"]:
            raise state["fail"]
        sent.append(prompt)
        state.setdefault("providers", []).append(provider)
        reply = dict(state["reply"])
        if "findings" in reply:
            reply.setdefault("not_identifiers", [])      # the name check's newer field
        return json.dumps(reply), (100, 20), "json_schema"

    import deid.qualify as qualify
    monkeypatch.setattr(qualify, "QUAL_FILE", local_keys / "qualification.json")
    monkeypatch.setattr(qualify, "is_qualified", lambda provider: state.get("qualified", True))
    monkeypatch.setenv("META_API_KEY", "test-key")
    monkeypatch.setenv("BASETEN_API_KEY", "test-key")
    monkeypatch.setattr(spark, "_request", fake_request)
    return {"sent": sent, "state": state}
