"""Extraction: define a field, find candidates, the four checks and the status. Fake Muse Spark only."""
import pytest

from common.data import load_report
from deid import pipeline
from deid.profile import new_profile
from extract.candidates import find_candidates
from extract.define import define_field
from extract.pack import load_pack

SIZE_DRAFT = {"key": "tumour_size", "meaning_question": "Largest tumour dimension on the surgical specimen?",
              "definition": "The largest invasive tumour dimension measured on the resection specimen.",
              "type": "number", "unit": "cm", "range_min": 0.1, "range_max": 30, "choices": []}


@pytest.fixture
def pack_path(local_keys):
    return local_keys / "pack.yaml"


def deid(text: str) -> str:
    r = pipeline.deidentify_report(text=text, profile=new_profile()[0])
    done = pipeline.confirm_deid(r["deid_id"], use_spark=False)
    if done["next"] == "hold":
        pipeline.confirm_deid(r["deid_id"], hold_answers={"keep": []}, use_spark=False)
    return r["deid_id"]


def define_size(fake_spark, pack_path):
    fake_spark["state"]["reply"] = SIZE_DRAFT
    draft = define_field("tumour size", "Measurement of the tumour taken from surgery", pack_path=pack_path)
    assert draft["next"] == "confirm" and draft["question"].startswith("Largest")
    define_field("tumour size", "Measurement of the tumour taken from surgery", accept=draft["draft"],
                 pack_path=pack_path)


def candidates(*cands):
    return {"fields": [{"key": "tumour_size", "candidates": [
        {"value_text": v, "unit_text": u, "evidence_text": e, "reason": "test", "confidence": c}
        for v, u, e, c in cands]}]}


def test_define_saves_field_and_sends_no_report_text(fake_spark, pack_path):
    define_size(fake_spark, pack_path)
    f = load_pack(pack_path)["fields"][0]
    assert (f.key, f.type, f.unit, f.range, f.readiness) == ("tumour_size", "number", "cm", [0.1, 30], "draft")
    assert "<<<REPORT" not in fake_spark["sent"][0]


def test_green_value_from_diagnosis(fake_spark, pack_path):
    define_size(fake_spark, pack_path)
    d = deid(load_report("PP0144"))                                   # extra split
    fake_spark["state"]["reply"] = candidates(("2.0", "cm", "2.0 x 2.0 x 0.9 cm subpleural mass", 0.9))
    r = find_candidates(d, "tumour_size", pack_path=pack_path)
    assert (r["status"], r["value"], r["unit"], r["section"]) == ("green", 2.0, "cm", "diagnosis")
    assert r["candidates"][0]["checks"] == {"evidence_found": True, "type_ok": True, "value_in_evidence": True,
                                            "section_ok": True, "no_identifiers": True}


def test_invented_number_is_amber(fake_spark, pack_path):
    define_size(fake_spark, pack_path)
    d = deid(load_report("PP0144"))
    fake_spark["state"]["reply"] = candidates(("3.5", "cm", "2.0 x 2.0 x 0.9 cm subpleural mass", 0.95))
    r = find_candidates(d, "tumour_size", pack_path=pack_path)
    assert r["status"] == "amber" and r["value"] is None and "value_in_evidence" in r["reason"]


def test_evidence_not_in_report_is_amber(fake_spark, pack_path):
    define_size(fake_spark, pack_path)
    d = deid(load_report("PP0144"))
    fake_spark["state"]["reply"] = candidates(("4.0", "cm", "a 4.0 cm mass in the left lobe", 0.9))
    assert find_candidates(d, "tumour_size", pack_path=pack_path)["reason"].endswith("(evidence_found)")


def test_history_only_and_conflict_and_missing(fake_spark, pack_path):
    define_size(fake_spark, pack_path)
    d = deid("CLINICAL HISTORY: 6 cm renal mass. DIAGNOSIS: Kidney, clear cell carcinoma, 5.7 cm and 4.1 cm.")
    fake_spark["state"]["reply"] = candidates(("6", "cm", "6 cm renal mass", 0.8))
    assert find_candidates(d, "tumour_size", pack_path=pack_path)["reason"] == "From the history section only"

    d = deid("DIAGNOSIS: Kidney, clear cell carcinoma, 5.7 cm and 4.1 cm.")
    fake_spark["state"]["reply"] = candidates(("5.7", "cm", "5.7 cm", 0.8), ("4.1", "cm", "4.1 cm", 0.7))
    assert find_candidates(d, "tumour_size", pack_path=pack_path)["reason"].startswith("Two different values")

    d = deid("DIAGNOSIS: Benign prostatic tissue.")
    fake_spark["state"]["reply"] = candidates()
    assert find_candidates(d, "tumour_size", pack_path=pack_path)["status"] == "missing"


def test_mm_is_converted_and_still_checked_against_the_text(fake_spark, pack_path):
    define_size(fake_spark, pack_path)
    d = deid("DIAGNOSIS: Invasive ductal carcinoma, 19 mm in greatest dimension.")
    fake_spark["state"]["reply"] = candidates(("19", "mm", "19 mm in greatest dimension", 0.9))
    r = find_candidates(d, "tumour_size", pack_path=pack_path)
    assert (r["status"], r["value"]) == ("green", 1.9)


def test_spark_failure_is_amber_not_a_crash(fake_spark, pack_path):
    define_size(fake_spark, pack_path)
    d = deid("DIAGNOSIS: Invasive ductal carcinoma, 19 mm.")
    fake_spark["state"]["fail"] = TimeoutError("slow")
    r = find_candidates(d, "tumour_size", pack_path=pack_path)
    assert r["status"] == "amber" and "review manually" in r["reason"]


def test_extraction_only_sees_deidentified_text(fake_spark, pack_path):
    define_size(fake_spark, pack_path)
    d = deid(load_report("PP0144"))
    fake_spark["state"]["reply"] = candidates()
    find_candidates(d, "tumour_size", pack_path=pack_path)
    prompt = fake_spark["sent"][-1]
    assert "Ananya" not in prompt and "G5987622N" not in prompt and "[PATIENT-1]" in prompt
