"""The uncued-names benchmark is frozen: generated once, checked by hash before every score."""
import eval.uncued_benchmark as bench


def test_benchmark_is_balanced_and_frozen(tmp_path, monkeypatch):
    from eval.hard_cases import parse
    cases = parse(bench.OUT)
    groups = {c["category"]: 0 for c in cases}
    for c in cases:
        groups[c["category"]] += 1
    assert all(n == 12 for g, n in groups.items() if g.endswith("names")) and len(cases) == 53
    assert bench.check_frozen() is None                                     # the committed set matches its hash

    copy, digest = tmp_path / "uncued.txt", tmp_path / "uncued.sha256"
    copy.write_text(bench.OUT.read_text(encoding="utf-8") + "Tuned on [[NAME:Someone New]].\n", encoding="utf-8")
    digest.write_text(bench.HASH.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(bench, "OUT", copy)
    monkeypatch.setattr(bench, "HASH", digest)
    assert "changed since it was frozen" in bench.check_frozen()
    assert bench.main() == 1                                                # never overwritten
