"""
pipeline.py - the review-first de-identification flow (PLAN.md §3A, §3D).

  deidentify_report(path)              Layer 1 on the laptop -> highlights. Nothing is sent.
  confirm_deid(deid_id, unmark, add)   user corrections -> Layer 2 (Muse Spark) on sealed, masked text
  confirm_deid_flags(deid_id, keep)    only if Layer 2 found more: all pre-ticked "hide"

Every response carries "next": profile | review | flags | done | error. Muse Spark is never
called before confirm_deid. A Layer 2 failure turns the run amber ("review manually"): never a
crash, never a raw-text fallback. Corrections are learned in the profile; Muse Spark finds
the user keeps hidden go to always_remove, so Layer 1 catches them locally next time.
Run state is saved encrypted per deid_id (runs/state/) for audit and replay.
"""
import secrets
from collections import Counter
from pathlib import Path

from common.config import ROOT
from common.timing import timed
from deid import profile as prof
from deid.hold import hold_flags
from deid.mask import apply_tags, mask, merge_spans, to_original
from deid.names import find_names, locate
from deid.patterns import find_pattern_phi, leftover_identifiers
from deid.seal import make_seal
from deid.types import DeidResult, PhiSpan
from deid.vault import encrypt_json, save_vault

STATE_DIR = ROOT / "runs" / "state"
_RUNS: dict[str, dict] = {}


# --------------------------------------------------------------------------
# The three steps
# --------------------------------------------------------------------------
def deidentify_report(path: str | None = None, *, text: str | None = None, report_id: str | None = None,
                      profile=None, profile_choices: dict | None = None, show_text: bool = False) -> dict:
    """Layer 1 only. profile: pass one in (eval) or it is loaded from profiles/default.yaml."""
    if text is None:
        with open(path, encoding="utf-8", newline="") as f:
            text = f.read()
        report_id = report_id or Path(path).stem
    persist, warnings = profile is None, []
    if profile is None:
        profile = prof.load_profile()
        if profile is None:
            if profile_choices is None:
                return {"next": "profile", "checklist": prof.checklist(),
                        "message": "First use: choose what to remove (all ticked except other dates)."}
            profile, warnings = prof.new_profile(profile_choices)
            prof.save_profile(profile)

    with timed("deid_layer1", report_id=report_id):
        removed = prof.removed_types(profile)
        allow = {a.lower() for a in profile.allow_list}
        spans = [s for s in find_pattern_phi(text, report_id, learned=profile.always_remove)
                 if s.type in removed and s.text.lower() not in allow]
    deid_id = "d_" + secrets.token_hex(4)
    run = {"deid_id": deid_id, "report_id": report_id, "text": text, "profile": profile, "persist": persist,
           "step": "review", "spans": spans, "allowed": list(profile.allow_list), "layer2": "not run",
           "warnings": warnings, "trace": []}
    _trace(run, "layer1", f"Presidio found {len(spans)} identifiers on the laptop. Nothing sent.")
    _RUNS[deid_id] = run
    _save(run)
    return {"deid_id": deid_id, "next": "review", "counts": dict(Counter(s.type for s in spans)),
            "detections": [_detection(text, spans, s, show_text) for s in spans],
            "kept_by_choice": list(prof.kept_types(profile)), "warnings": warnings}


def confirm_deid(deid_id: str, unmark=(), add=(), hold_answers=None, *, use_spark: bool = True,
                 show_text: bool = False) -> dict:
    """Call 1: unmark (false-alarm ids), add ([{"text", "type"}] missed text).
    Call 2, only if the reply was next="hold": hold_answers={"keep": [ids of words that are not names]}.
    """
    run = _RUNS.get(deid_id)
    if run is None or run["step"] not in ("review", "hold"):
        return _error(deid_id, "nothing to confirm for this deid_id")
    text = run["text"]

    if run["step"] == "review":
        unmark = set(unmark)
        kept = [s for s in run["spans"] if s.id in unmark]
        spans = [s for s in run["spans"] if s.id not in unmark]
        for a in add:
            hits = locate(text, a["text"])
            if not hits:
                run["warnings"].append(f"Marked text not found in the report (type {a['type']}).")
            for start, end in hits:
                spans.append(PhiSpan(id=f"u{len(spans) + 1}", type=a["type"], start=start, end=end,
                                     text=text[start:end], source="user"))
        spans = merge_spans(spans)
        prof.learn(run["profile"], allow=[s.text for s in kept],
                   remove=[{"text": a["text"], "type": a["type"]} for a in add])
        run["allowed"] += [s.text for s in kept]
        run["confirmed"] = spans
        _trace(run, "review", f"You confirmed {len(spans)} identifiers ({len(kept)} unmarked, {len(add)} added).")

        # Hold rule: before any API call, pause on words that still look like names.
        tagged, tags = apply_tags(text, spans)
        holds = []
        for start, end in hold_flags(tagged, run["allowed"]):
            o_start, o_end = to_original(tags, start, end)
            holds.append(PhiSpan(id=f"h{len(holds) + 1}", type="NAME", start=o_start, end=o_end,
                                 text=text[o_start:o_end], source="hold"))
        if holds:
            run.update(step="hold", holds=holds)
            _trace(run, "hold", f"Paused before any API call: {len(holds)} words look like names.", ok=False)
            _save(run)
            return {"deid_id": deid_id, "next": "hold",
                    "hold": [_detection(text, spans + holds, h, show_text) for h in holds],
                    "note": "These look like names. All are ticked to hide; send the ids of any that are not names as keep."}
    else:
        keep = set((hold_answers or {}).get("keep", ()))
        hide = [h for h in run["holds"] if h.id not in keep]
        kept = [h for h in run["holds"] if h.id in keep]
        prof.learn(run["profile"], allow=[h.text for h in kept], remove=[{"text": h.text, "type": "NAME"} for h in hide])
        run["allowed"] += [h.text for h in kept]
        spans = merge_spans(run["confirmed"] + hide)
        _trace(run, "hold", f"You hid {len(hide)} and kept {len(kept)} of the held words.")

    run["layer1_final"] = spans          # what is masked when Muse Spark is called (exposure)

    if not use_spark:
        run["layer2"] = "skipped"
        return _finish(run, spans)
    masked = mask(deid_id, text, spans, kept_types=prof.kept_types(run["profile"]), allowed=run["allowed"])
    try:
        with timed("deid_layer2", report_id=run["report_id"]):
            found, not_found = find_names(masked, text, run["report_id"])
    except Exception as e:   # API down, timeout, bad JSON, missing key: amber, never a crash
        run["layer2"] = "unavailable"
        run["warnings"].append(f"Muse Spark check unavailable ({type(e).__name__}). Review the text manually.")
        _trace(run, "layer2", "Muse Spark check unavailable: review manually.", ok=False)
        return _finish(run, spans)

    new = [s for s in found if not _covered(s, spans)]
    if not new:
        run["layer2"] = "clean"
        _trace(run, "layer2", "Muse Spark double-checked the masked text: nothing new found.")
        return _finish(run, spans)
    run.update(step="flags", flags=new, layer2="flagged")
    _trace(run, "layer2", f"Muse Spark found {len(new)} more possible identifiers.", ok=False)
    _save(run)
    return {"deid_id": deid_id, "next": "flags",
            "flags": [_detection(text, spans + new, s, show_text) for s in new],
            "note": "All are ticked to hide. Send the ids of any false alarms as keep."}


def confirm_deid_flags(deid_id: str, keep=()) -> dict:
    run = _RUNS.get(deid_id)
    if run is None or run["step"] != "flags":
        return _error(deid_id, "no flags waiting for this deid_id")
    keep = set(keep)
    kept = [s for s in run["flags"] if s.id in keep]
    hide = [s for s in run["flags"] if s.id not in keep]
    # Layer 1 learns: hidden finds are caught locally next time.
    prof.learn(run["profile"], allow=[s.text for s in kept], remove=[{"text": s.text, "type": s.type} for s in hide])
    _trace(run, "flags", f"You hid {len(hide)} and kept {len(kept)}; hidden ones are learned for next time.")
    return _finish(run, merge_spans(run["layer1_final"] + hide))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _finish(run: dict, spans: list[PhiSpan]) -> dict:
    deid_id, text, profile = run["deid_id"], run["text"], run["profile"]
    tagged, tags = apply_tags(text, spans)
    kept, allowed = prof.kept_types(profile), tuple(sorted(run["allowed"]))

    # Output scan: nothing the patterns recognise may survive in the final text.
    leftovers = leftover_identifiers(tagged, kept, allowed)
    if leftovers:
        run["warnings"].append(f"{len(leftovers)} identifier pattern(s) still in the output. Review manually.")

    originals = {}
    for t in tags:
        originals.setdefault(t.surrogate, []).append(text[t.orig_start:t.orig_end])
    vault_path = save_vault(deid_id, {"case_id": secrets.token_hex(6), "report_id": run["report_id"],
                                      "tags": {k: sorted(set(v)) for k, v in originals.items()}})
    summary = {"total": len(tags), "by_type": dict(Counter(t.type for t in tags)),
               "by_source": dict(Counter(t.source.split(":")[0] for t in tags))}
    result = DeidResult(deid_id=deid_id, report_id=run["report_id"], status="confirmed", text=tagged, tags=tags,
                        summary=summary, profile_version=profile.version, vault_ref=str(vault_path),
                        kept_types=kept, allowed=allowed, seal=make_seal(deid_id, tagged, kept, allowed))
    run.update(step="done", final_spans=spans, result=result)
    if run["persist"]:
        prof.save_profile(profile)
    _trace(run, "apply", f"{len(tags)} identifiers replaced with tags; encrypted vault saved.", ok=not leftovers)
    _save(run)
    return {"deid_id": deid_id, "next": "done", "text": tagged, "summary": summary, "layer2": run["layer2"],
            "warnings": run["warnings"], "trace": run["trace"]}


def result_for(deid_id: str) -> DeidResult | None:
    """The sealed DeidResult of a finished run (input to extraction)."""
    run = _RUNS.get(deid_id)
    return run.get("result") if run else None


def _covered(span: PhiSpan, spans: list[PhiSpan]) -> bool:
    return any(s.start <= span.start and span.end <= s.end for s in spans)


def _detection(text: str, spans: list[PhiSpan], target: PhiSpan, show_text: bool, width: int = 40) -> dict:
    """id, type and masked context. The identifier itself only if show_text (local viewer)."""
    tagged, tags = apply_tags(text, spans if any(s.id == target.id for s in spans) else spans + [target])
    t = next((t for t in tags if t.orig_start <= target.start < t.orig_end), None)
    context = "" if t is None else (tagged[max(0, t.start - width):t.start] + ">>" + tagged[t.start:t.end]
                                    + "<<" + tagged[t.end:t.end + width]).replace("\n", " ")
    d = {"id": target.id, "type": target.type, "source": target.source, "context": context}
    if show_text:
        d["text"] = target.text
    return d


def _trace(run: dict, step: str, message: str, ok: bool = True) -> None:
    run["trace"].append({"step": step, "observe": message, "ok": ok})


def _error(deid_id: str, message: str) -> dict:
    return {"deid_id": deid_id, "next": "error", "error": message}


def _save(run: dict) -> None:
    """Encrypted state for audit and replay (contains report text and identifiers)."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = {k: run[k] for k in ("deid_id", "report_id", "step", "text", "allowed", "layer2", "warnings", "trace")}
    for key in ("spans", "confirmed", "holds", "layer1_final", "flags", "final_spans"):
        if key in run:
            state[key] = [s.to_dict() for s in run[key]]
    (STATE_DIR / f"{run['deid_id']}.enc").write_bytes(encrypt_json(state))
