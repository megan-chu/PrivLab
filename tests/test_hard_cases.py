"""The hard-case file parses into text + spans whose offsets match the text."""
from eval.hard_cases import parse


def test_every_span_matches_its_text():
    cases = parse()
    assert len(cases) >= 30
    for c in cases:
        for s in c["phi"]:
            assert c["text"][s["start"]:s["end"]] == s["text"]
        assert "[[" not in c["text"]


def test_line_break_inside_marker():
    case = next(c for c in parse() if "\n" in c["text"])
    assert case["phi"][0]["text"] == "Tan\nAh Kow"


def test_negatives_have_no_identifiers():
    negatives = [c for c in parse() if c["category"].startswith("Negatives")]
    assert negatives and all(not c["phi"] for c in negatives)
