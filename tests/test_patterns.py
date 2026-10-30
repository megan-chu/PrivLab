"""Pattern recognisers: one hand-written example each, plus the edge cases found on dev/extra."""
import pytest

from common.data import load_phi_truth, load_report
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
    assert ("NAME", "Low") not in hits and ("NAME", "Shan") in hits         # "Low grade" stays; "Pei Shan" masked


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
