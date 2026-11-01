"""Pattern recognisers: one hand-written example each, plus the edge cases found on dev/extra."""
import pytest

from common.data import load_phi_truth, load_report
from deid.mask import apply_tags
from deid.patterns import find_pattern_phi


def found(text: str) -> list[tuple[str, str]]:
    return [(s.type, s.text) for s in find_pattern_phi(text)]


@pytest.mark.parametrize("text, expected", [
    ("NRIC: S1234567D", ("ID", "S1234567D")),
    ("NHS No: 943 476 5919", ("ID", "943 476 5919")),
    ("SSN: 918-79-4700", ("ID", "918-79-4700")),
    ("sent to jo.tan12@mailbox.example today", ("EMAIL", "jo.tan12@mailbox.example")),
    ("Phone: +65 6999 8752", ("PHONE", "+65 6999 8752")),
    ("Phone: +44 7112 398477", ("PHONE", "+44 7112 398477")),
    ("contact (549) 555-2318.", ("PHONE", "(549) 555-2318")),
    ("Collected: 09/05/2025", ("DATE", "09/05/2025")),
    ("discussed on 14 May 2025.", ("DATE", "14 May 2025")),
    ("DOB: 26/01/1942", ("DOB", "26/01/1942")),
    ("Accession No: PA25-92841", ("ACCESSION", "PA25-92841")),
    ("ref MRN9980912", ("MRN", "MRN9980912")),
    ("MRN: H1088762", ("MRN", "H1088762")),
    ("ACCT # : 92071195", ("ACCOUNT", "92071195")),
    ("Patient: Ong Wei Ling    NRIC", ("NAME", "Ong Wei Ling")),
    ("Patient's husband Ahmad Faizal binte Hassan requested", ("NAME", "Ahmad Faizal binte Hassan")),
    ("signed by Dr Lee Mei Xuan, Consultant", ("DOCTOR", "Dr Lee Mei Xuan")),
    ("Address: 14 Hawthorn Close, Leeds LS6 2QT\n", ("ADDRESS", "14 Hawthorn Close, Leeds LS6 2QT")),
    ("Whitfield NHS Foundation Trust Hospital\n", ("HOSPITAL", "Whitfield NHS Foundation Trust Hospital")),
    ("Meet on the 1st Floor lobby", ("ADDRESS", "1st Floor")),
    ("Floor 2 lobby is closed", ("ADDRESS", "Floor 2")),
    ("First Floor lobby", ("ADDRESS", "First Floor")),
    ("Ground Floor lobby", ("ADDRESS", "Ground Floor")),
    ("2nd Fl lobby", ("ADDRESS", "2nd Fl")),
    ("3 Storey carpark", ("ADDRESS", "3 Storey")),
    ("Meet on the 1st\nFloor lobby", ("ADDRESS", "1st\nFloor")),
])
def test_recogniser_examples(text, expected):
    assert expected in found(text)


def test_checksum_rejects_unlabelled_lookalike_ids():
    assert found("code S1234567E") == []           # wrong check letter, no label
    assert found("ref 943 476 5918") == []         # wrong check digit, no label


def test_labelled_or_lowercase_ids_are_caught():
    assert found("NRIC: S1234567E") == [("ID", "S1234567E")]   # after a label: masked even if mistyped
    assert found("code s1234567d") == [("ID", "s1234567d")]    # valid checksum in lower case


@pytest.mark.parametrize("text", [
    "performed by Dr. In Surgical Pathology, the specimen",
    "Diagnosis called to Dr. GROSS DESCRIPTION: A. 4R",
    "Dictated by Dr. This information is subject",
])
def test_tcga_placeholders_are_not_doctors(text):
    assert not [t for t, _ in found(text) if t == "DOCTOR"]


def test_floor_words_outside_addresses_are_not_masked():
    assert not [t for t, _ in found("pelvic floor intact; floor of mouth clear.") if t == "ADDRESS"]
    assert not [t for t, _ in found("discharged to Miami, FL.") if t == "ADDRESS"]   # US state, not "Fl"
    assert not [t for t, _ in found("Level 2 lymph nodes involved.") if t == "ADDRESS"]   # neck levels


def test_doctor_name_stops_before_heading():
    assert ("DOCTOR", "Dr Oliver Turner") in found("by Dr Oliver Turner FROZEN SECTION DIAGNOSIS")


def test_footer_name_and_mrn():
    assert found("Page 1 of 1  |  Ananya Subramaniam  |  P5221936") == [
        ("NAME", "Ananya Subramaniam"), ("MRN", "P5221936")]


def test_pp0144_every_identifier_found_with_its_type():
    # PP0144 is in the extra split.
    text = load_report("PP0144")
    spans = find_pattern_phi(text)
    for s in load_phi_truth("PP0144")["phi"]:
        assert any(p.start == s["start"] and p.end == s["end"] and p.type == s["type"] for p in spans), s
    assert len(spans) == 17


@pytest.mark.parametrize("text, expected", [
    ("Next of kin: Tan Ah Wing (86743890), son of Mr.Tan.", [("NAME", "Tan Ah Wing"), ("PHONE", "86743890"), ("NAME", "Tan")]),
    ("NOK Siti Rahmah (daughter), tel 9123 4567.", [("NAME", "Siti Rahmah"), ("PHONE", "9123 4567")]),
    ("Emergency contact: husband, Ong Kok Wah, 6123-4567.", [("NAME", "Ong Kok Wah"), ("PHONE", "6123-4567")]),
    ("Her son, Kevin Tan, requested a copy.", [("NAME", "Kevin Tan")]),
    ("Prof Lim Boon Huat reviewed the slides.", [("DOCTOR", "Lim Boon Huat")]),
    ("Mdm Young reports that the young patient is her grandson.", [("NAME", "Young")]),
    ("Report reviewed by Dr A. Kumar.", [("DOCTOR", "Dr A. Kumar")]),
])
def test_uncommon_cues(text, expected):
    assert found(text) == expected


def test_repeat_mentions_of_a_found_name():
    text = "Mr Low Ten Young was seen. LOW TEN YOUNG consented later."
    assert found(text) == [("NAME", "Low Ten Young"), ("NAME", "LOW TEN YOUNG")]   # repeat in any case


def test_surname_that_is_a_report_word_is_not_repeated():
    text = "Patient: Low Pei Shan\nDIAGNOSIS: Low grade carcinoma. Pei Shan consented."
    hits = found(text)
    second = text.rindex("Pei Shan")
    assert ("NAME", "Low") not in hits                                       # "Low grade" stays
    assert any(s.start <= second + 4 < s.end for s in find_pattern_phi(text))   # "Pei Shan" masked


def test_doctor_name_stops_at_section_heading():
    assert ("DOCTOR", "Dr Michael Anderson") in found("by Dr Michael Anderson Light Microscopy: Light microscopic")


def test_learned_names_skip_single_words():
    learned = [{"text": "Tan", "type": "NAME"}, {"text": "Rajesh Kumar", "type": "NAME"}]
    text = "A tan-coloured nodule. Tan, firm. RAJESH KUMAR phoned."
    assert [(s.type, s.text) for s in find_pattern_phi(text, learned=learned)] == [("NAME", "RAJESH KUMAR")]


@pytest.mark.parametrize("text, expected", [
    ("DOB: 14-May-1952", ("DOB", "14-May-1952")),
    ("DOB (dd/mm/yyyy): 14/05/1952", ("DOB", "14/05/1952")),
    ("Age 72 (b. 14/05/1952).", ("DOB", "14/05/1952")),
    ("MRN-4903402", ("MRN", "4903402")),
    ("Medical record number 4903402.", ("MRN", "4903402")),
    ("Acct # 1234567", ("ACCOUNT", "1234567")),
    ("Call 6999.8752 for results.", ("PHONE", "6999.8752")),
    ("Contact hp91234567 after 5 pm.", ("PHONE", "91234567")),
    ("NHS number 401.023.2137.", ("ID", "401.023.2137")),
    ("NRIC S  7654321  F (double spaces).", ("ID", "S  7654321  F")),
    ("NRIC Ѕ7654321F (Cyrillic S).", ("ID", "Ѕ7654321F")),
    ("Results discussed with Mr TAN AH KOW.", ("NAME", "TAN AH KOW")),
    ("Signed out by DR LIM BOON HUAT.", ("DOCTOR", "DR LIM BOON HUAT")),
    ("Home: Block 88 Tampines Street 81, #03-214.", ("ADDRESS", "Block 88 Tampines Street 81, #03-214")),
])
def test_red_team_formats(text, expected):
    assert expected in found(text)


def test_capitalised_report_words_are_not_doctors():
    assert not [t for t, _ in found("REPORTED TO: DR. REPORTED BY: DR. # BLOCKS : 2.") if t == "DOCTOR"]


def test_form_labels_are_not_held_but_names_before_a_colon_are():
    from deid.hold import hold_flags
    text = ("Vehicle Plate: SLL 4921 G (Make: Honda Vezel 1.5A)\n"
            "Contact list. Mary Tan: 9123 4567. Next of kin Rajesh Kumar: brother.")
    held = [text[a:b] for a, b in hold_flags(text)]
    assert "Vehicle Plate" not in held and "Make" not in held                    # labels
    assert "Mary Tan" in held and "Rajesh Kumar" in held                         # listed names stay held


BANK = """CASE FILE REF: AML-2026-OCT-0941
PRIMARY RELATIONSHIP MANAGER: Denise Alcantara (Staff ID: PB-99412)
Account Name: Horizon Ventures Global Trust
Account Number: 884-2901-7724
Beneficial Owner: Victor Nikolai Voronin
Nationality: Cypriot / Passport No: K9044129
Tax Identification (TIN): 099-231-884A
Residential Address: Apartment 18A, Nassim Jade, 38 Nassim Road, Singapore 258399
Denise spoke with Victor. Voronin was screened. Shares bought via Broker ID: SGX-TR-4401.
"""


def test_any_report_type_labelled_ids_names_and_addresses():
    got = {t for _, t in found(BANK)}
    assert {"AML-2026-OCT-0941", "PB-99412", "884-2901-7724", "K9044129", "099-231-884A", "SGX-TR-4401",
            "Denise Alcantara", "Horizon Ventures Global Trust", "Victor Nikolai Voronin",
            "Apartment 18A, Nassim Jade, 38 Nassim Road, Singapore 258399",
            "Denise", "Victor", "Voronin"} <= got                                   # later mentions too
    assert ("ID", "SLL 4921 G") in found("Vehicle Plate: SLL 4921 G (Make: Honda)")
    assert ("NAME", "Tan Wei Ming") in found("Driver: Tan Wei Ming")


@pytest.mark.parametrize("text", ["LYMPH NODES EXAMINED: 12", "AGE/SX: 64/F", "Specimen No: 2",
                                  "Reference range: 3.5-5.0 mmol/L", "SWIFT: BCGECHGG"])
def test_labels_without_an_identifier_are_left_alone(text):
    assert not [t for _, t in found(text)]


def test_labels_next_to_an_id_are_not_held():
    from deid.hold import hold_flags
    text = "Manager: [PERSON-1] (Staff ID: [ID-1]). Passport No: [ID-2]. Tax Identification (TIN): [ID-3]."
    assert hold_flags(text) == []


def test_local_name_model_fills_gaps_only():
    pytest.importorskip("en_core_web_sm")
    text = ("DIAGNOSIS: Invasive ductal carcinoma, Gleason pattern noted; Gerota fascia intact.\n"
            "Escalated to Group Compliance Head Marcus Wong for review.\n"
            "by Dr Oliver Turner FROZEN SECTION DIAGNOSIS")
    hits = found(text)
    assert ("NAME", "Marcus Wong") in hits                                       # no cue, found by the model
    assert ("DOCTOR", "Dr Oliver Turner") in hits                                # a pattern is never overridden
    assert not any(t in ("Gleason", "Gerota", "DIAGNOSIS") for _, t in hits)    # report wording is not a name
    from deid.patterns import leftover_identifiers
    assert leftover_identifiers("Escalated to Marcus Wong for review.") == []    # final scan stays rules-only


def test_genomic_report_identifiers_caught_locally():
    """A US genomic (NGS) report: before 1 Nov these reached the Muse Spark name check unmasked."""
    from pathlib import Path

    from deid.mask import apply_tags
    text = (Path(__file__).parent / "genomic_report.txt").read_text(encoding="utf-8")
    tagged, _ = apply_tags(text, find_pattern_phi(text))
    for leaked in ("1.888.988.3639", "2202027531", "3402044309", "Foundation Medicine", "Metropolitan Health",
                   "Cancer Ctr", "Morrisville, NC 27560", "Cambridge, MA 02141"):
        assert leaked not in tagged, leaked
    names = {s.text for s in find_pattern_phi(text) if s.type in ("NAME", "DOCTOR")}
    assert not names & {"Vitrakvi®", "Keytruda®", "M.M. Sc"}                   # brands and degrees are not people
    assert {"J. Keith Killian", "Julia Elvin", "Shakti Ramkissoon"} <= names
    from deid.hold import hold_flags
    tagged, tags = apply_tags(text, find_pattern_phi(text))
    assert len(hold_flags(tagged, (), tags)) <= 1                             # was 23 questions on 1 Nov
    clinical = "Certified under CLIA '88 as qualified. Ki-67 index 12.5.2024 batch v1.2.3. Stage pT2, N0 12345."
    assert find_pattern_phi(clinical, names_model=False) == [] or all(
        s.type == "DATE" for s in find_pattern_phi(clinical, names_model=False))     # no CLIA, phone or address


@pytest.mark.parametrize("text, name", [
    ("Signed out by Kuganesan Thanabalasingam: no change.", "Kuganesan Thanabalasingam"),
    ("Comments from Marcus Zawadzki (surgeon): agree.", "Marcus Zawadzki"),
])
def test_a_name_before_a_colon_is_not_a_label(text, name):
    from deid.hold import hold_flags
    assert name in [text[a:b] for a, b in hold_flags(text)]                   # red-team round 2, H01 / H03


@pytest.mark.parametrize("text, name", [
    ("Results discussed with Marcus Zawadzki today.", "Marcus Zawadzki"),
    ("Collection Date confirmed by Kuganesan Thanabalasingam: no change.", "Kuganesan Thanabalasingam"),
    ("Sample from Tan Ah Wing Status: pending.", "Tan Ah Wing Status"),
    ("Reviewed by Sean McDonald and Ana DeSouza.", "Sean McDonald"),
])
def test_lab_wording_rules_still_hold_names(text, name):
    """The 1 Nov rules for lab and genomic wording (labels, units, codes, products) must not let a name through."""
    from deid.hold import hold_flags
    assert any(name in text[a:b] or text[a:b] in name for a, b in hold_flags(text))


@pytest.mark.parametrize("text, held", [
    ("Started on Osimertinib and Tagrisso after Biopsy Review.", []),          # drugs, terms: nothing says "person"
    ("Fusion partner Vitrakvi Larotrectinib noted.", []),
    ("Discussed with Okonkwo today.", ["Okonkwo"]),                             # a person cue
    ("The patient lives in Jurong West now.", ["Jurong West"]),                 # a place cue
    ("Slides sent to Raffles Medical for review.", ["Raffles Medical"]),       # an institution word
    ("Not identifiers: [PERSON-1], Siti Rahmah.", ["Siti Rahmah"]),             # listed beside a name
])
def test_hold_needs_evidence_of_a_person_or_place(text, held):
    from deid.hold import hold_flags
    assert [text[a:b] for a, b in hold_flags(text)] == held


@pytest.mark.parametrize("text,full,clia", [
    ("7010 Kit Creek Road, Morrisville, NC 27560 (CLIA: 3402044309)",
     "7010 Kit Creek Road, Morrisville, NC 27560", "3402044309"),
    ("150 Second St., 1st Floor, Cambridge, MA 02141 (CLIA: 2202027531)",
     "150 Second St., 1st Floor, Cambridge, MA 02141", "2202027531"),
])
def test_adjacent_address_parts_merge_into_one_span(text, full, clia):
    spans = find_pattern_phi(text)
    assert [(s.type, s.text) for s in spans if s.type == "ADDRESS"] == [("ADDRESS", full)]
    assert [s.text for s in spans if s.type == "ID"] == [clia]


def test_addresses_do_not_merge_across_lines_or_types():
    spans = find_pattern_phi("Address: 12 Jalan Besar\nPhone: +65 6123 4567")
    assert [(s.type, s.text) for s in spans if s.type == "ADDRESS"] == [("ADDRESS", "12 Jalan Besar")]
    assert [s.type for s in spans if s.type == "PHONE"]
    two = find_pattern_phi("12 Jalan Besar\n14 Hawthorn Close")
    assert [(s.type, s.text) for s in two if s.type == "ADDRESS"] == [
        ("ADDRESS", "12 Jalan Besar"), ("ADDRESS", "14 Hawthorn Close")]


def test_merged_address_masks_as_one_tag():
    text = "7010 Kit Creek Road, Morrisville, NC 27560 (CLIA: 3402044309)"
    tagged, _ = apply_tags(text, find_pattern_phi(text))
    assert tagged == "[ADDRESS-1] (CLIA: [ID-1])"
