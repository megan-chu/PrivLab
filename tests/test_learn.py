"""Learning from confirmations, the test-then-learn log, readiness, history, export. Fake Muse Spark only."""
import pytest

from deid import pipeline
from deid.profile import new_profile
from eval.score_fields import score
from extract.candidates import find_candidates
from extract.define import define_field
from extract.export import export_rows
from extract.learn import confirm_field, read_log, show_pack
from extract.pack import load_pack

DRAFT = {"key": "tumour_size", "meaning_question": "Largest tumour dimension?", "type": "number",
         "unit": "cm", "range_min": 0.1, "range_max": 30, "choices": []}
REPORT = "DIAGNOSIS: Invasive ductal carcinoma, 2.1 cm. GROSS: specimen 9 x 6 x 3 cm, tumour 2.1 cm."


@pytest.fixture
def paths(fake_spark, local_keys):
    pack, log = local_keys / "pack.yaml", local_keys / "score_log.jsonl"
    fake_spark["state"]["reply"] = DRAFT
    define_field("tumour size", None, accept=DRAFT, pack_path=pack)
    return {"pack": pack, "log": log, "spark": fake_spark}


def run(paths, text, *cands):
    r = pipeline.deidentify_report(text=text, profile=new_profile()[0])
    pipeline.confirm_deid(r["deid_id"], use_spark=False)
    paths["spark"]["state"]["reply"] = {"fields": [{"key": "tumour_size", "candidates": [
        {"value_text": v, "unit_text": "cm", "evidence_text": e, "reason": "t", "confidence": 0.9}
        for v, e in cands]}]}
    return find_candidates(r["deid_id"], "tumour_size", pack_path=paths["pack"])


def confirm(paths, res, **kw):
    return confirm_field(res["deid_id"], "tumour_size", pack_path=paths["pack"], log_path=paths["log"], **kw)


def test_correct_is_logged_then_learned(paths):
    res = run(paths, REPORT, ("2.1", "Invasive ductal carcinoma, 2.1 cm"))
    out = confirm(paths, res, candidate_id=res["candidate_id"])
    assert out == {"status": "saved", "outcome": "correct", "value": 2.1, "readiness": "learning"}
    entry = read_log(paths["log"])[0]
    assert entry["predicted"]["value"] == 2.1 and entry["confirmed"]["value"] == 2.1
    assert entry["pack_version"] < load_pack(paths["pack"])["version"]       # scored before the pack changed
    assert len(load_pack(paths["pack"])["fields"][0].examples["good"]) == 1


def test_wrong_pick_becomes_a_bad_example(paths):
    res = run(paths, REPORT, ("9", "specimen 9 x 6 x 3 cm"), ("2.1", "tumour 2.1 cm"))
    assert res["value"] == 9.0                                               # both gross; model's order
    out = confirm(paths, res, candidate_id="c2")
    f = load_pack(paths["pack"])["fields"][0]
    assert out["outcome"] == "wrong" and f.examples["bad"][0]["reason"] == "not this one"
    assert "specimen 9 x 6 x 3 cm" in f.examples["bad"][0]["context"]


def test_typed_value_missed_and_extra(paths):
    res = run(paths, REPORT)                                                 # model found nothing
    assert confirm(paths, res, value="2.1", evidence_text="tumour 2.1 cm")["outcome"] == "missed"
    res = run(paths, "DIAGNOSIS: Benign tissue. Ink 2.0 cm from margin.", ("2.0", "Ink 2.0 cm from margin"))
    assert confirm(paths, res, missing=True)["outcome"] == "extra"
    res = run(paths, "DIAGNOSIS: Benign tissue.")
    assert confirm(paths, res, missing=True)["outcome"] == "true_empty"


def test_ready_after_four_of_five_and_examples_capped(paths):
    for i in range(6):
        res = run(paths, REPORT, ("2.1", "Invasive ductal carcinoma, 2.1 cm"))
        out = confirm(paths, res, candidate_id=res["candidate_id"])
    f = load_pack(paths["pack"])["fields"][0]
    assert out["readiness"] == "ready" and len(f.examples["good"]) == 3
    s = score(read_log(paths["log"]))["tumour_size"]
    assert (s["f1"], s["confirmations_to_ready"], s["learning_curve"][-1]) == (1.0, 5, 1.0)


def test_confirm_needs_a_prediction_first(paths):
    assert confirm_field("d_none", "tumour_size", candidate_id="c1", pack_path=paths["pack"],
                         log_path=paths["log"])["status"] == "error"


def test_history_and_export_have_no_free_text(paths):
    res = run(paths, REPORT, ("2.1", "Invasive ductal carcinoma, 2.1 cm"))
    confirm(paths, res, candidate_id=res["candidate_id"])
    hist = show_pack("tumour_size", pack_path=paths["pack"], log_path=paths["log"])
    assert hist["definition_history"][0]["what"] == "created"
    assert hist["extraction_history"][0]["outcome"] == "correct"
    rows, withheld = export_rows(pack_path=paths["pack"], log_path=paths["log"])
    assert withheld == 0 and rows[0]["value"] == 2.1 and rows[0]["case_id"]
    assert all("carcinoma" not in str(v) for row in rows for v in row.values())
    assert "carcinoma" not in str(hist["extraction_history"])


def test_learned_examples_reach_the_next_prompt(paths):
    res = run(paths, REPORT, ("9", "specimen 9 x 6 x 3 cm"), ("2.1", "tumour 2.1 cm"))
    confirm(paths, res, candidate_id="c2")
    run(paths, REPORT.replace("2.1", "3.4"))
    prompt = paths["spark"]["sent"][-1]
    assert "correct example" in prompt and "wrong example (do not pick)" in prompt


def test_confirm_refuses_null_or_failed_values(paths):
    res = run(paths, REPORT, ("99", "Invasive ductal carcinoma, 2.1 cm"))    # value not in its evidence
    assert "failed" in confirm(paths, res, candidate_id="c1")["reason"]
    assert "not a number" in confirm(paths, res, value="about two", evidence_text="tumour 2.1 cm")["reason"]
    assert confirm(paths, res, value=" ", evidence_text="tumour 2.1 cm")["status"] == "error"
    assert read_log(paths["log"]) == []                                       # nothing scored or learned


def test_teach_input_handling(paths, monkeypatch):
    import importlib
    tryer = importlib.import_module("extract.try")       # "try" is a keyword
    monkeypatch.setattr(tryer, "TRY_PACK", paths["pack"])
    monkeypatch.setattr(tryer, "TRY_LOG", paths["log"])
    res = run(paths, REPORT, ("2.1", "Invasive ductal carcinoma, 2.1 cm"))
    answers = iter(["0", "²", "v 2.1", "v2.1 | tumour 2.1 cm"])               # 3 rejected, then accepted
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert tryer.teach(res, "tumour_size").endswith("readiness now learning")
    missing = run(paths, "DIAGNOSIS: Benign tissue.")
    monkeypatch.setattr("builtins.input", lambda _: "")                       # Enter on "missing"
    assert "true_empty" in tryer.teach(missing, "tumour_size")


def test_example_with_identifier_is_not_learned_or_sent(paths, monkeypatch):
    import common.spark as spark
    from extract.pack import save_pack
    # A snippet that still holds an NRIC must never be stored, and never leave in a prompt.
    pack = load_pack(paths["pack"])
    pack["fields"][0].examples["good"].append({"report": "x", "value": 2.1, "context": "NRIC S1234567D tumour 2.1 cm"})
    save_pack(pack, paths["pack"])
    r = pipeline.deidentify_report(text=REPORT, profile=new_profile()[0])
    pipeline.confirm_deid(r["deid_id"], use_spark=False)
    res = find_candidates(r["deid_id"], "tumour_size", pack_path=paths["pack"])
    assert res["status"] == "amber" and paths["spark"]["sent"] == []         # refused before sending
