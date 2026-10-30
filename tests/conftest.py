"""Shared fixtures: throwaway local keys and temp folders, so tests never touch real vault/profile files."""
import secrets

import pytest
from cryptography.fernet import Fernet


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
