"""Dataset integrity, config guards and de-id data types."""
import pytest

from common.config import ConfigError, load_config
from common.data import load_phi_truth, load_report, manifest
from deid.types import PHI_TYPES, PhiSpan


def test_splits():
    counts = {s: len(manifest(s)) for s in ("dev", "test", "extra")}
    assert counts == {"dev": 20, "test": 50, "extra": 130}


def test_all_truth_spans_match_report_text():
    # Checks offsets only; does not read report content for any split-specific purpose.
    n = 0
    for row in manifest():
        text = load_report(row["report_id"])
        truth = load_phi_truth(row["report_id"])
        assert len(truth["phi"]) == int(row["n_phi"])
        for s in truth["phi"]:
            assert text[s["start"]:s["end"]] == s["text"], (row["report_id"], s)
            assert s["type"] in PHI_TYPES
            n += 1
    assert n == 3143


def test_contributor_model_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("META_MODEL", "muse-spark-1.3-contributor")
    with pytest.raises(ConfigError):
        load_config(tmp_path / "none.env")


def test_require_names_missing_variable(monkeypatch, tmp_path):
    monkeypatch.delenv("META_API_KEY", raising=False)
    settings = load_config(tmp_path / "none.env")
    assert settings.meta_model == "muse-spark-1.3"
    with pytest.raises(ConfigError, match="META_API_KEY"):
        settings.require("meta_api_key")


def test_phispan_round_trip_and_validation():
    s = PhiSpan(id="p1", type="ID", start=128, end=137, text="G5987622N", source="pattern:sg_nric")
    assert PhiSpan.from_dict(s.to_dict()) == s
    with pytest.raises(ValueError):
        PhiSpan(id="p2", type="SECRET", start=0, end=1, text="x", source="user")
    with pytest.raises(ValueError):
        PhiSpan(id="p3", type="NAME", start=5, end=5, text="", source="user")


def test_labelled_ssn_gets_an_ssn_tag():
    from deid.mask import apply_tags
    from deid.patterns import find_pattern_phi
    text = "Patient: Daniel Martinez    SSN: 967-97-6295\n"
    tagged, _ = apply_tags(text, find_pattern_phi(text))
    assert "[SSN-1]" in tagged and "[ID-1]" not in tagged
