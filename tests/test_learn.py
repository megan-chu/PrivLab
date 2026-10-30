"""Learning from confirmations, the test-then-learn log, readiness, history, export. Fake Muse Spark only."""
import pytest

from deid import pipeline
from deid.profile import new_profile
from eval.score_fields import score
from extract.candidates import find_candidates
from extract.define import define_field
from extract.export import export_rows
from extract.learn import confirm_field, more_needed, read_log, show_pack, upper_bound, validation
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
    start, end = out.pop("start"), out.pop("end")
    assert out == {"status": "saved", "outcome": "correct", "value": 2.1, "readiness": "learning"}
    assert REPORT[start:end] == "Invasive ductal carcinoma, 2.1 cm"            # where the confirmed value is
    entry = read_log(paths["log"])[0]
    assert entry["predicted"]["value"] == 2.1 and entry["confirmed"]["value"] == 2.1
    assert entry["pack_version"] < load_pack(paths["pack"])["version"]       # scored before the pack changed
    assert len(load_pack(paths["pack"])["fields"][0].examples["good"]) == 1


def test_wrong_pick_becomes_a_bad_example(paths):
    res = run(paths, REPORT, ("9", "specimen 9 x 6 x 3 cm"), ("2.1", "tumour 2.1 cm"))
    assert res["value"] == 9.0                                               # both gross; model's order
    out = confirm(paths, res, candidate_id="c2")
    f = load_pack(paths["pack"])["fields"][0]
    assert out["outcome"] == "wrong"
    assert f.examples["bad"][0]["reason"] == "Picked 9 cm; the right value was 2.1 cm, elsewhere in the gross section."
    assert "specimen 9 x 6 x 3 cm" in f.examples["bad"][0]["context"]


def test_rejected_pick_explains_why_and_keeps_the_reviewer_note(paths):
    text = "DIAGNOSIS: specimen 9 x 6 x 3 cm. GROSS: tumour 2.1 cm."
    res = run(paths, text, ("9", "specimen 9 x 6 x 3 cm"))                    # PrivLab picks the wrong one
    confirm(paths, res, value="2.1", evidence_text="tumour 2.1 cm", note="whole specimen, not the tumour")
    reason = load_pack(paths["pack"])["fields"][0].examples["bad"][-1]["reason"]
    assert reason.startswith("Picked 9 cm from the diagnosis section; the right value was 2.1 cm in the gross section.")
    assert reason.endswith("Reviewer: whole specimen, not the tumour")
    res = run(paths, text.replace("2.1", "3.4"))                              # it reaches the next prompt
    assert "Reviewer: whole specimen, not the tumour" in paths["spark"]["sent"][-1]


def test_reviewer_note_with_an_identifier_is_dropped(paths):
    text = "DIAGNOSIS: specimen 9 x 6 x 3 cm. GROSS: tumour 2.1 cm."
    res = run(paths, text, ("9", "specimen 9 x 6 x 3 cm"))                    # PrivLab picks the wrong one
    confirm(paths, res, value="2.1", evidence_text="tumour 2.1 cm", note="ask Dr Lim on +65 9123 4567")
    reason = load_pack(paths["pack"])["fields"][0].examples["bad"][-1]["reason"]
    assert "9123" not in reason and "Reviewer" not in reason                 # identifiers never reach a prompt


def test_typed_value_missed_and_extra(paths):
    res = run(paths, REPORT)                                                 # model found nothing
    assert confirm(paths, res, value="2.1", evidence_text="tumour 2.1 cm")["outcome"] == "missed"
    res = run(paths, "DIAGNOSIS: Benign tissue. Ink 2.0 cm from margin.", ("2.0", "Ink 2.0 cm from margin"))
    assert confirm(paths, res, missing=True)["outcome"] == "extra"
    res = run(paths, "DIAGNOSIS: Benign tissue.")
    assert confirm(paths, res, missing=True)["outcome"] == "true_empty"


def test_six_right_is_not_ready_and_examples_capped(paths):
    for i in range(6):
        res = run(paths, REPORT, ("2.1", "Invasive ductal carcinoma, 2.1 cm"))
        out = confirm(paths, res, candidate_id=res["candidate_id"])
    f = load_pack(paths["pack"])["fields"][0]
    assert out["readiness"] == "learning" and len(f.examples["good"]) == 3   # a streak is not proof
    v = validation(f, read_log(paths["log"]))
    assert (v["checked"], v["green"], v["silent_errors"], v["more_needed"]) == (6, 6, 0, 53)
    s = score(read_log(paths["log"]))["tumour_size"]
    assert (s["f1"], s["confirmations_to_ready"], s["learning_curve"][-1]) == (1.0, None, 1.0)
    assert s["median_learn_ms"] is not None and s["median_review_s"]["green"] is not None


def test_ready_needs_the_95_percent_bound():
    # exact one-sided binomial: 59 / 93 / 124 green values with 0 / 1 / 2 errors prove <= 5% wrong
    for errors, n in [(0, 59), (1, 93), (2, 124)]:
        assert upper_bound(errors, n) <= 0.05 < upper_bound(errors, n - 1)
    assert more_needed(0, 0) == 59 and more_needed(1, 60) == 33 and more_needed(0, 59) == 0
    from extract.types import Field
    f = Field(key="t", name="t", description="", meaning="m", type="number")
    row = lambda status, outcome, **kw: {"field": "t", "predicted": {"status": status}, "outcome": outcome, **kw}
    clean = [row("green", "correct") for _ in range(59)]
    assert validation(f, clean)["ready"]
    assert not validation(f, clean[1:] + [row("green", "wrong")])["ready"]          # one silent error
    v = validation(f, clean[1:] + [row("amber", "wrong")])                           # caught by review, not silent
    assert (v["silent_errors"], v["green"], v["ready"]) == (0, 58, False)
    assert validation(f, [{**r, "field_version": 2} for r in clean])["checked"] == 0  # other version


def test_new_meaning_is_a_new_version(paths):
    res = run(paths, REPORT, ("2.1", "Invasive ductal carcinoma, 2.1 cm"))
    confirm(paths, res, candidate_id=res["candidate_id"])
    define_field("tumour size", "my words", accept=DRAFT, pack_path=paths["pack"])          # same meaning
    f = load_pack(paths["pack"])["fields"][0]
    assert f.version == 1 and f.examples["good"] and f.readiness == "learning"
    define_field("tumour size", None, accept={**DRAFT, "definition": "Invasive part only."}, pack_path=paths["pack"])
    f = load_pack(paths["pack"])["fields"][0]
    assert f.version == 2 and f.examples == {"good": [], "bad": []} and f.readiness == "draft"
    assert f.history[-1]["what"] == "redefined as v2; score and examples start again"
    assert validation(f, read_log(paths["log"]))["checked"] == 0                    # v1 results are not mixed in


def test_blind_check_hides_the_answer_until_the_user_has_answered(paths, monkeypatch):
    import extract.candidates as candidates
    monkeypatch.setattr(candidates, "_is_blind", lambda *a: True)
    r = pipeline.deidentify_report(text=REPORT, profile=new_profile()[0])
    pipeline.confirm_deid(r["deid_id"], use_spark=False)
    paths["spark"]["state"]["reply"] = {"fields": [{"key": "tumour_size", "candidates": [
        {"value_text": "2.1", "unit_text": "cm", "evidence_text": "Invasive ductal carcinoma, 2.1 cm", "reason": "t", "confidence": 0.9}]}]}
    res = find_candidates(r["deid_id"], "tumour_size", pack_path=paths["pack"], blind_checks=True)
    assert res["status"] == "blind" and res["value"] is None and res["candidates"] == []
    assert confirm(paths, res, candidate_id="c1")["status"] == "error"              # cannot pick the hidden answer
    out = confirm(paths, res, value="2.1", evidence_text="tumour 2.1 cm")
    assert (out["outcome"], out["blind"], out["predicted"]) == ("correct", True, 2.1)
    entry = read_log(paths["log"])[0]
    assert entry["blind"] and entry["field_version"] == 1 and entry["predicted"]["status"] == "green"
    assert validation(load_pack(paths["pack"])["fields"][0], read_log(paths["log"]))["blind_correct"] == 1
    assert find_candidates(r["deid_id"], "tumour_size", pack_path=paths["pack"])["status"] == "green"   # off by default


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


def test_review_and_model_time_are_recorded(paths):
    res = run(paths, REPORT, ("2.1", "Invasive ductal carcinoma, 2.1 cm"))
    confirm(paths, res, candidate_id=res["candidate_id"])
    entry = read_log(paths["log"])[0]
    assert entry["review_s"] >= 0 and entry["model_ms"] is not None and entry["learn_ms"] >= 0
    s = score(read_log(paths["log"]))["tumour_size"]
    assert s["median_review_s"]["amber"] is not None or s["median_review_s"]["green"] is not None


def test_deid_review_time_per_step(fake_spark):
    r = pipeline.deidentify_report(text="Next of kin: Rajesh Kumar phoned. Seen with Tan Ah Kow.", profile=new_profile()[0])
    held = pipeline.confirm_deid(r["deid_id"], use_spark=False)
    done = pipeline.confirm_deid(r["deid_id"], hold_answers={"keep": []}, use_spark=False) \
        if held["next"] == "hold" else held
    assert set(done["review_seconds"]) >= {"review"} and all(v >= 0 for v in done["review_seconds"].values())


def test_one_call_per_report_even_after_a_correction(paths):
    grade = {"key": "grade", "meaning_question": "Histological grade?", "type": "choice",
             "unit": None, "range_min": None, "range_max": None, "choices": ["1", "2", "3"]}
    paths["spark"]["state"]["reply"] = grade
    define_field("grade", None, accept=grade, pack_path=paths["pack"])
    r = pipeline.deidentify_report(text=REPORT + " Grade 2.", profile=new_profile()[0])
    pipeline.confirm_deid(r["deid_id"], use_spark=False)
    paths["spark"]["state"]["reply"] = {"fields": [
        {"key": "tumour_size", "candidates": [{"value_text": "2.1", "unit_text": "cm", "reason": "t",
                                                "evidence_text": "Invasive ductal carcinoma, 2.1 cm", "confidence": 0.9}]},
        {"key": "grade", "candidates": [{"value_text": "2", "unit_text": "", "reason": "t",
                                          "evidence_text": "Grade 2", "confidence": 0.9}]}]}
    sent = len(paths["spark"]["sent"])
    res = find_candidates(r["deid_id"], "tumour_size", pack_path=paths["pack"])
    confirm(paths, res, candidate_id=res["candidate_id"])                   # the pack changes here
    res2 = find_candidates(r["deid_id"], "grade", pack_path=paths["pack"])
    assert len(paths["spark"]["sent"]) == sent + 1                           # the second field reused the reply
    assert res2["value"] == "2" and res2["model_ms"] == 0                    # short choice codes pass the checks


def test_check_lists_only_results_to_recheck(paths):
    from eval.score_fields import to_check
    res = run(paths, REPORT, ("9", "specimen 9 x 6 x 3 cm"), ("2.1", "tumour 2.1 cm"))
    confirm(paths, res, candidate_id="c2")                                   # wrong pick
    res = run(paths, REPORT, ("2.1", "Invasive ductal carcinoma, 2.1 cm"))
    confirm(paths, res, candidate_id=res["candidate_id"])                   # correct
    lines = to_check(read_log(paths["log"]))
    assert lines[0].startswith("\n1 results") and "wrong" in lines[1] and len(lines) == 2


def test_redefined_field_gets_a_new_call_but_learning_does_not(paths):
    res = run(paths, REPORT, ("2.1", "Invasive ductal carcinoma, 2.1 cm"))
    sent = len(paths["spark"]["sent"])
    confirm(paths, res, candidate_id=res["candidate_id"])                   # learning: examples change
    find_candidates(res["deid_id"], "tumour_size", pack_path=paths["pack"])
    assert len(paths["spark"]["sent"]) == sent                               # same definition: no call
    paths["spark"]["state"]["reply"] = DRAFT
    define_field("tumour size", "invasive tumour only",
                 accept={**DRAFT, "meaning_question": "Largest invasive tumour dimension?"}, pack_path=paths["pack"])
    paths["spark"]["state"]["reply"] = {"fields": [{"key": "tumour_size", "candidates": []}]}
    find_candidates(res["deid_id"], "tumour_size", pack_path=paths["pack"])
    assert len(paths["spark"]["sent"]) == sent + 1                           # new meaning: new call


def test_remove_field(paths):
    from extract.pack import load_pack, remove_field, save_pack
    pack = load_pack(paths["pack"])
    assert remove_field(pack, "tumour_size") and not remove_field(pack, "tumour_size")
    save_pack(pack, paths["pack"])
    assert load_pack(paths["pack"])["fields"] == []
