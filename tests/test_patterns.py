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


def test_card_iban_and_bank_names():
    text = ("Paid by Visa 4111 1111 1111 1111. Refund to IBAN GB82 WEST 1234 5698 7654 32 at DBS Bank, "
            "Bank of America and OCBC Bank Ltd.")
    got = found(text)
    for item in [("ACCOUNT", "4111 1111 1111 1111"), ("ACCOUNT", "GB82 WEST 1234 5698 7654 32"), ("BANK", "DBS Bank"),
                 ("BANK", "Bank of America"), ("BANK", "OCBC Bank Ltd.")]:
        assert item in got


def test_statement_mailing_block_and_account_number():
    text = ("0000111122223333444\nJANE SAMPLE DOE                     Statement Date   1 Jan 2030\n"
            "45 EXAMPLE AVENUE                   Credit Limit     1,000.00\nSAMPLE TOWERS\n09-04\nSG 999999\n")
    got = found(text)
    for item in [("ACCOUNT", "0000111122223333444"), ("NAME", "JANE SAMPLE DOE"), ("ADDRESS", "45 EXAMPLE AVENUE"),
                 ("ADDRESS", "SAMPLE TOWERS"), ("ADDRESS", "09-04"), ("ADDRESS", "SG 999999")]:
        assert item in got
    assert "Statement Date" not in " ".join(t for _, t in got)          # the second column is not part of the address


def test_the_postcode_is_masked_wherever_it_appears_again():
    text = ("(999999B)              REF 0000111122223333444\n999999 0000111122223333444\n"
            "JANE SAMPLE DOE\n45 EXAMPLE AVENUE\nSG 999999\n")
    spans = find_pattern_phi(text)
    left = "".join("#" if any(s.start <= i < s.end for s in spans) else ch for i, ch in enumerate(text))
    assert "999999" not in left                                         # every copy, bare or with its letter
    assert ("ADDRESS", "999999B") in found(text)


@pytest.mark.parametrize("text, address", [
    ("Discharged to 21 Example Gateway, #12-03, Sample Towers, Singapore 999999.",
     "21 Example Gateway, #12-03, Sample Towers, Singapore 999999"),
    ("Home: 10, Jalan Contoh 3, Taman Sample, 50450 Kuala Lumpur, Malaysia.",
     "10, Jalan Contoh 3, Taman Sample, 50450 Kuala Lumpur, Malaysia"),
    ("Correspondence to Apt 3B, 77 Harbour Boulevard only.", "Apt 3B, 77 Harbour Boulevard"),
    ("Transferred from 46000 Petaling Jaya, Selangor last month.", "46000 Petaling Jaya, Selangor"),
])
def test_address_by_its_parts(text, address):
    assert ("ADDRESS", address) in found(text)


def test_one_address_word_alone_is_not_an_address():
    for text in ["Level 2 lymph nodes involved.", "Hill's criteria were met after a court-ordered review.",
                 "Tissue was sent overseas to Australia.", "The tumour bank received 12345 samples."]:
        assert not [t for t, _ in found(text) if t == "ADDRESS"], text


def test_heading_block_without_a_postcode_is_not_an_address():
    assert found("FINAL DIAGNOSIS\nInvasive ductal carcinoma, grade 2\nTumour size 21 mm\n") == []


def test_sixteen_digits_in_groups_masked_even_if_checksum_fails():
    assert found("Block 4111 1111 1111 1112 sections.") == [("ACCOUNT", "4111 1111 1111 1112")]
    assert found("bill 1234  1324  1234  1234 paid.") == [("ACCOUNT", "1234  1324  1234  1234")]


def test_invisible_and_full_width_characters_do_not_hide_identifiers():
    text = "Patient: Jane Sa­mple Doe\nNRIC: S12​34567D\nEmail: ｊａｎｅ＠ｅｘａｍｐｌｅ．ｃｏｍ\n"
    spans = find_pattern_phi(text)
    covered = lambda part: any(s.start <= text.find(part) and text.find(part) + len(part) <= s.end for s in spans)
    assert covered("Jane Sa­mple Doe") and covered("S12​34567D") and covered("ｊａｎｅ＠ｅｘａｍｐｌｅ．ｃｏｍ")


def test_addr_label():
    assert ("ADDRESS", "45 Example Avenue, #09-04") in found("Addr: 45 Example Avenue, #09-04\n")


def test_card_and_bank_lookalikes_are_not_masked():
    assert found("Ki-67 1234567890123.") == []
    assert found("GB82 WEST 1234 5698 7654 33") == []                                   # wrong IBAN check digits
    assert found("Sent to the tumour bank and the Tissue Bank; a swift recovery.") == []


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


def test_portal_is_a_word_not_a_name_but_portal_singh_still_held():
    from deid.hold import hold_flags
    assert hold_flags("Portal access verified for [EMAIL-1] with 2FA enabled.") == []
    text = "Portal Singh consented to the procedure."
    assert "Portal Singh" in [text[a:b] for a, b in hold_flags(text)]


def test_reversed_names_caught():
    assert ("NAME", "TAN, K. W.") in found('Specimen container labelled "TAN, K. W. / S-24-98432".')
    assert ("NAME", "Al-Mansoor, Tariq") in found("Seen in consultation with Al-Mansoor, Tariq.")
    assert found("Tan, firm cut surface.") == []
    assert all(t != "NAME" for t, _ in found("7010 Kit Creek Road, Morrisville, NC 27560"))


def test_lowercase_apostrophe_name_after_dr():
    assert ("DOCTOR", "d'Artois-Smith") in found("Post-op check with Dr d'Artois-Smith; dressing changed.")
    assert all(t != "DOCTOR" for t, _ in found("Dr ordered labs."))


def test_particle_names_caught_but_common_words_not():
    assert ("NAME", "van der Meer") in found("Biopsy reviewed by van der Meer, resident on duty.")
    assert found("de novo mutation noted.") == []
    assert found("transported by van to lab.") == []


def test_dotted_typo_name_caught():
    assert ("NAME", "Nur.Aisyah") in found("Follow-up arranged for Nur.Aisyah at polyclinic.")
    assert found("v2.0 released.") == []


def test_lab_ref_ids_after_cue_words():
    assert ("ID", "B24-10294A") in found("Block B24-10294A (Lim Ah Beng) received.")
    assert ("ID", "S24/912-1") in found("Cassette marked S24/912-1: section taken.")
    assert ("ACCESSION", "SURG-2024-001298") in found("Cross-reference histology accession SURG-2024-001298.")
    assert ("ID", "PRBC#88219-02") in found("Cross-match unit barcode PRBC#88219-02 verified.")
    assert ("ID", "ACC/2024/09/8812") in found("Billing reference ACC/2024/09/8812 attached.")
    assert all(t != "ID" for t, _ in found("Reference range 4.5-11.0."))
    assert all(t != "ID" for t, _ in found("Slide 5 of 6."))
    assert all(t != "ID" for t, _ in found("Specimen: BREAST, RIGHT."))


def test_mrn_bare_with_trailing_letter():
    assert found("MRN9823411A") == [("MRN", "MRN9823411A")]
    assert ("MRN", "MRN9823411A") in found("Specimen container stamped MRN9823411A without separator.")


def test_pager_cue_and_phone_extension_tail():
    assert ("PHONE", "b/p 9238-1120") in found("Urgent findings paged to b/p 9238-1120 at 1400 hrs.")
    assert ("PHONE", "6321-4311 loc 204") in found("reachable via hospital trunk line 6321-4311 loc 204.")
    assert all(t != "PHONE" for t, _ in found("BP 120/80 noted."))
    assert ("PHONE", "1.888.988.3639") in found("Call 1.888.988.3639 now.")


def test_report_header_dob_with_age_reversed_name_and_spaced_barcode():
    text = ("* P A 3 0 - 1 2 3 4 5 * S 1 2 3 4 5 6 7 D *\nPatient Name: Sample, Jane Ann\n"
            "DOB / Age: 01/02/1960 (Age 66)\n")
    got = found(text)
    for item in [("ID", "S 1 2 3 4 5 6 7 D"), ("ID", "P A 3 0 - 1 2 3 4 5"), ("NAME", "Sample, Jane Ann"),
                 ("DOB", "01/02/1960")]:
        assert item in got
    assert found("Margins: A B C D E clear; levels 1 2 3 4 5 examined.") == []          # spaced letters, few digits


def test_medical_words_are_not_names_but_listed_names_stay_names():
    for text in ["Lymph Nodes Macrometastases: 2 of 15", "Number of Lymph Nodes with Macrometastases (> 2 mm): 2"]:
        assert not [t for t, _ in found(text) if t in ("NAME", "DOCTOR")], text
    assert ("NAME", "Ramesh Sabapathy") in found("Discussed with Ramesh Sabapathy at the clinic.")   # "-pathy" surname


def test_name_stops_before_next_line_label():
    got = found("Patient: John Smith\nGender: Male")
    assert ("NAME", "John Smith") in got
    assert not [s for t, s in got if t == "NAME" and "Gender" in s]   # the label is not part of the name
    assert ("NAME", "John\nSmith") in found("Patient: John\nSmith")   # a wrapped name still joins


def test_middle_initials_stay_inside_the_name():
    assert ("DOCTOR", "Dr. Jane K. Example") in found("Surgeon: Dr. Jane K. Example")
    assert ("DOCTOR", "Dr Jane K Example") in found("Reported by Dr Jane K Example, consultant.")
    assert ("DOCTOR", "Dr Tan") in found("Dr Tan A Grade 2 tumour")              # "A" is a word, not an initial


def test_a_label_on_the_next_line_is_not_part_of_the_name():
    assert ("NAME", "Jane Sample") in found("Patient: Jane Sample\nDate Of Birth: 01/02/1960")
    assert ("NAME", "Jane Doe") in found("Name: Jane Doe\nAge 54")
    assert ("NAME", "Jane\nSample") in found("Patient: Jane\nSample")                  # a wrapped name still joins


def test_surnames_that_look_medical_are_not_dismissed():
    from deid.hold import medical_word
    assert not any(medical_word(w) for w in ("Pasupathy", "Angiolini", "Metaxas"))
    assert ("NAME", "Elena Angiolini") in found("Seen by Elena Angiolini today.")
