"""Checksums must accept every planted ID (dev + extra) and reject one-digit typos."""
import re

import pytest

from common.data import load_phi_truth, manifest
from deid.checksums import nhs_valid, nric_valid


def planted_ids() -> list[str]:
    return [s["text"] for r in manifest() if r["split"] != "test"
            for s in load_phi_truth(r["report_id"])["phi"] if s["type"] == "ID"]


def mutate(s: str, i: int) -> str:
    return s[:i] + str((int(s[i]) + 1) % 10) + s[i + 1:]


def test_all_planted_nrics_valid_and_typos_fail():
    nrics = [i for i in planted_ids() if re.fullmatch(r"[STFG]\d{7}[A-Z]", i)]
    assert len(nrics) > 100
    for n in nrics:
        assert nric_valid(n), n
        for i in range(1, 8):
            assert not nric_valid(mutate(n, i)), (n, i)


def test_all_planted_nhs_numbers_valid_and_typos_fail():
    nhs = [i for i in planted_ids() if re.fullmatch(r"\d{3} ?\d{3} ?\d{4}", i)]
    assert len(nhs) > 40
    for n in nhs:
        assert nhs_valid(n), n
        for i in (i for i, c in enumerate(n) if c.isdigit()):
            assert not nhs_valid(mutate(n, i)), (n, i)


@pytest.mark.parametrize("s", ["S1234567", "X1234567A", "abc", ""])
def test_malformed_nric(s):
    assert not nric_valid(s)


@pytest.mark.parametrize("s", ["12345", "943 476 5919 1", "abc", ""])
def test_malformed_nhs(s):
    assert not nhs_valid(s)


def test_known_examples():
    assert nric_valid("S1234567D")
    assert nhs_valid("943 476 5919")
    assert not nhs_valid("943 476 5918")
