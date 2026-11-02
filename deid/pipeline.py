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

The final response carries review_priority (high | medium | low, with reasons), so a reviewer
checks the riskiest reports first: high = a pattern left in the output, or the name check missing
or unqualified; medium = name check not run, or name-like words the user kept visible.
"""
import gc
import re
import secrets
import threading
import time
from collections import Counter
from pathlib import Path

from common import spark
from common.config import ROOT
from common.timing import record, timed
from deid import profile as prof
from deid.hold import hold_flags, name_list, suspects, vocabulary
from deid.mask import apply_tags, mask, merge_spans, to_original
from deid.names import find_names, locate
from deid.patterns import find_pattern_phi, leftover_identifiers
from deid.qualify import choose_provider
from deid.seal import make_seal
from deid.types import DeidResult, PhiSpan
from deid.vault import encrypt_json, save_vault

STATE_DIR = ROOT / "runs" / "state"
INSTITUTION = re.compile(r"\b(?:Hospital|Cent(?:er|re)|Clinic|Institute|Infirmary)$")   # a held hospital, not a person
_RUNS: dict[str, dict] = {}
_PROFILE_LOCK = threading.Lock()          # one finish at a time reads, merges and saves the privacy profile


def warm_up() -> None:
    """Call once at start-up (viewer, MCP server). Loads the vocabulary, name lists and Presidio
    recognisers, then freezes them for the garbage collector: a full collection would otherwise
    rescan ~170,000 long-lived objects and pause one report for 40-250 ms."""
    vocabulary(), name_list()
    find_pattern_phi("Patient: Mary Tan  NRIC: S1234567D  Phone: +65 6123 4567")   # compiles recognisers, loads NER
    gc.collect()
    gc.freeze()


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
    if report_id and find_pattern_phi(re.sub(r"[_.\-]+", " ", report_id)):
        # the report's name goes into every export (report_id column, masked file names): never personal details
        report_id = f"report_{secrets.token_hex(3)}"
        warnings.append(f"The file name looked like it held personal details, so this report is called "
                        f"\"{report_id}\" in every export.")
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
        reusable = prof.reusable_allow_list(profile)
        allow = {a.lower() for a in reusable}
        found = find_pattern_phi(text, report_id, learned=profile.always_remove)
        spans = [s for s in found if s.type in removed and s.text.lower() not in allow]
        kept_by_choice = [s.text for s in found if s.type not in removed]    # e.g. hospital names you keep
        # Words that still look like a person or place: shown with the rest, ticked, for one confirm (on device)
        spans += [h for h in _hold_spans(text, spans, list(reusable) + kept_by_choice) if h.type in removed]
    deid_id = "d_" + secrets.token_hex(4)
    run = {"deid_id": deid_id, "report_id": report_id, "text": text, "profile": profile, "persist": persist,
           "step": "review", "spans": spans, "allowed": list(reusable) + kept_by_choice, "layer2": "not run",
           "warnings": warnings, "trace": [], "shown_at": time.monotonic(), "review_s": {},
           "kept_names": 0, "unqualified": False, "priority": None,
           "learn": {"allow": [], "remove": []},       # staged; the profile learns only when the run finishes
           "allowed_n": len(reusable) + len(kept_by_choice)}     # how much of "allowed" came from the profile
    possible = sum(s.source == "hold" for s in spans)
    _trace(run, "layer1", f"Found {len(spans)} identifiers on this device" + (f", {possible} of them possible names "
           "for you to check" if possible else "") + ". Nothing sent.")
    _RUNS[deid_id] = run
    _save(run)
    return {"deid_id": deid_id, "next": "review", "counts": dict(Counter(s.type for s in spans)),
            "detections": [_learned_from(_detection(text, spans, s, show_text), s, profile) for s in spans],
            "kept_by_choice": list(prof.kept_types(profile)), "warnings": warnings}


def confirm_deid(deid_id: str, unmark=(), add=(), hold_answers=None, *, use_spark: bool = True,
                 show_text: bool = False, expect: str | None = None) -> dict:
    """Call 1: unmark (false-alarm ids), add ([{"text", "type"}] missed text).
    Call 2, only if the reply was next="hold": hold_answers={"keep": [ids of words that are not names]}.
    expect: the step the caller answered ("review" or "hold"); a run that has moved on refuses the answer,
    so a late or repeated review answer is never applied to the hold question.
    """
    run = _RUNS.get(deid_id)
    if run is None or run["step"] not in ("review", "hold"):
        return _error(deid_id, "nothing to confirm for this deid_id")
    if expect and run["step"] != expect:
        return {**_error(deid_id, f"this report is now at the {run['step']} step; reload it and answer that"),
                "step": run["step"]}
    _review_time(run)
    text = run["text"]

    if run["step"] == "review":
        unmark = set(unmark)
        given = {(a["text"], a["type"]) for a in add}        # text you added before Back counts again
        add = list(add) + [{"text": t, "type": ty} for t, ty in run.get("added", []) if (t, ty) not in given]
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
        _stage(run, allow=[s.text for s in kept],      # a model-found or possible name you kept hidden is
               remove=[{"text": a["text"], "type": a["type"]} for a in add]   # learned like a held one
               + [{"text": s.text, "type": s.type} for s in spans
                  if s.source == "ner:spacy" or s.source == "hold" and len(s.text.split()) > 1])   # not "Tan" from one click
        run["allowed"] += [s.text for s in kept]
        run["kept_names"] += sum(s.source == "hold" for s in kept)
        run["confirmed"] = spans
        _trace(run, "review", f"You confirmed {len(spans)} identifiers ({len(kept)} unmarked, {len(add)} added).")

        # The hold rule already ran on the review screen; it runs again only if you changed what is hidden,
        # so a word your change uncovered is still asked about before any API call.
        removed = prof.removed_types(run["profile"])
        holds = [h for h in _hold_spans(text, spans, run["allowed"], prefix="r") if h.type in removed] if unmark or add else []
        if holds:
            run.update(step="hold", holds=holds, shown_at=time.monotonic())
            _trace(run, "hold", f"Paused before any API call: {len(holds)} words look like names.", ok=False)
            _save(run)
            return {"deid_id": deid_id, "next": "hold",
                    "hold": [_detection(text, spans + holds, h, show_text) for h in holds],
                    "note": "These look like names. All are ticked to hide; send the ids of any that are not names as keep."}
    else:
        keep = set((hold_answers or {}).get("keep", ()))
        hide = [h for h in run["holds"] if h.id not in keep]
        kept = [h for h in run["holds"] if h.id in keep]
        _stage(run, allow=[h.text for h in kept], remove=[{"text": h.text, "type": h.type} for h in hide])
        run["allowed"] += [h.text for h in kept]
        run["kept_names"] += len(kept)
        spans = merge_spans(run["confirmed"] + hide)
        _trace(run, "hold", f"You hid {len(hide)} and kept {len(kept)} of the held words.")

    run["layer1_final"] = spans          # what is masked when Muse Spark is called (exposure)

    if not use_spark:
        run["layer2"] = "skipped"
        return _finish(run, spans)
    masked = mask(deid_id, text, spans, kept_types=prof.kept_types(run["profile"]), allowed=run["allowed"])
    provider, warning = choose_provider()
    if warning:
        run["warnings"].append(warning)
    run["unqualified"] = warning is not None and provider == spark.STEP_PROVIDER["names"]
    try:
        with timed("deid_layer2", report_id=run["report_id"], provider=provider):
            found, not_found, unresolved = find_names(
                masked, text, run["report_id"], provider=provider,
                suspects=suspects(masked.text, run["allowed"], masked.tags))
    except Exception as e:   # API down, timeout, bad JSON, missing key: amber, never a crash
        run["layer2"] = "unavailable"
        run["warnings"].append(f"{spark.OFFLINE}: the second check did not run. De-identification on this device is "
                               "complete; read the text once before sharing it." if isinstance(e, spark.OfflineError)
                               else f"Muse Spark check unavailable ({type(e).__name__}). Review the text manually.")
        _trace(run, "layer2", "Muse Spark check unavailable: review manually.", ok=False)
        return _finish(run, spans)
    run["layer2_provider"] = provider

    new = [s for s in found if not _covered(s, spans)]
    unresolved = [q for q in unresolved if not _covered(q, spans) and not _covered(q, new)]
    if not new and not unresolved:
        run["layer2"] = "clean"
        _trace(run, "layer2", "Muse Spark double-checked the masked text: nothing new found.")
        return _finish(run, spans)
    run.update(step="flags", flags=new + unresolved, layer2="flagged", shown_at=time.monotonic())
    _trace(run, "layer2", f"Muse Spark found {len(new)} more possible identifiers"
                          + (f"; {len(unresolved)} words it did not rule on are shown for you to check" if unresolved else "")
                          + ".", ok=False)
    _save(run)
    every = spans + new + unresolved
    return {"deid_id": deid_id, "next": "flags",
            "flags": [{**_detection(text, every, s, show_text), "ticked": s.source != "suspect"} for s in new + unresolved],
            "note": ("Model finds are ticked to hide; send their ids as keep if they are false alarms. "
                     "Words the model did not rule on are unticked; send their ids as hide if they are identifiers.")}


def back_to_review(deid_id: str) -> dict:
    """From the hold question back to the highlights, to change your answers. Only before any API call:
    after the name check, going back would mean sending again (start the report again instead)."""
    run = _RUNS.get(deid_id)
    if run is None or run["step"] != "hold":
        return _error(deid_id, "can only go back from the hold question")
    # A true undo: nothing from the first pass is learned or allowed any more. Text you added stays hidden
    # (the viewer of a report started in chat does not know it), so going back never un-hides it.
    run["added"] = sorted({(s.text, s.type) for s in run.get("confirmed", []) if s.source == "user"} | set(run.get("added", [])))
    run.update(step="review", shown_at=time.monotonic(), learn={"allow": [], "remove": []},
               allowed=run["allowed"][:run["allowed_n"]])
    run.pop("holds", None)
    _trace(run, "review", "Back to the highlights; nothing was sent.")
    _save(run)
    return {"deid_id": deid_id, "next": "review"}


def confirm_deid_flags(deid_id: str, keep=(), hide=()) -> dict:
    """keep: ids of ticked model finds that are false alarms. hide: ids of unticked suspects that are identifiers."""
    run = _RUNS.get(deid_id)
    if run is None or run["step"] != "flags":
        return _error(deid_id, "no flags waiting for this deid_id")
    _review_time(run)
    keep, hide_ids = set(keep), set(hide)
    hidden = [s for s in run["flags"] if (s.source != "suspect" and s.id not in keep)
              or (s.source == "suspect" and s.id in hide_ids)]
    kept = [s for s in run["flags"] if s not in hidden]
    # Learning: hidden finds are caught locally next time; kept ones are not asked again.
    _stage(run, allow=[s.text for s in kept], remove=[{"text": s.text, "type": s.type} for s in hidden])
    run["allowed"] += [s.text for s in kept]
    run["kept_names"] += len(kept)
    _trace(run, "flags", f"You hid {len(hidden)} and kept {len(kept)}; hidden ones are learned for next time.")
    return _finish(run, merge_spans(run["layer1_final"] + hidden))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _finish(run: dict, spans: list[PhiSpan]) -> dict:
    deid_id, text, profile = run["deid_id"], run["text"], run["profile"]
    tagged, tags = apply_tags(text, spans)
    kept, allowed = prof.kept_types(profile), tuple(sorted(run["allowed"]))

    # Output scan: nothing the patterns recognise may survive in the final text.
    leftovers = leftover_identifiers(tagged, kept, allowed, tags)
    if leftovers:
        run["warnings"].append(f"{len(leftovers)} identifier pattern(s) still in the output. Review manually.")

    originals = {}
    for t in tags:
        originals.setdefault(t.surrogate, []).append(text[t.orig_start:t.orig_end])
    case_id = run.setdefault("case_id", secrets.token_hex(6))      # one per report, however often it is finished
    vault_path = save_vault(deid_id, {"case_id": case_id, "report_id": run["report_id"],
                                      "tags": {k: sorted(set(v)) for k, v in originals.items()}})
    summary = {"total": len(tags), "by_type": dict(Counter(t.type for t in tags)),
               "by_source": dict(Counter(t.source.split(":")[0] for t in tags))}
    result = DeidResult(deid_id=deid_id, report_id=run["report_id"], status="confirmed", text=tagged, tags=tags,
                        summary=summary, profile_version=profile.version, vault_ref=str(vault_path),
                        kept_types=kept, allowed=allowed, seal=make_seal(deid_id, tagged, kept, allowed))
    run.update(step="done", final_spans=spans, result=result, priority=review_priority(run, len(leftovers)),
               leftovers=leftovers)
    # commit what this run staged (idempotent on a re-finish); each learned identifier keeps the report it came from
    # Merged into the profile as it is on disk now, not this run's copy from when it started: other reports may
    # have learned, and the user may have changed choices or removed an entry since.
    with _PROFILE_LOCK:
        if run["persist"]:
            profile = prof.load_profile() or profile
        new = prof.learn(profile, allow=run["learn"]["allow"],
                         remove=[{**x, "from": run["report_id"]} for x in run["learn"]["remove"]])
        if run["persist"]:
            prof.save_profile(profile)
    run["profile"] = profile
    run["learned"] = {"allow": run.get("learned", {}).get("allow", []) + new["allow"],
                      "remove": run.get("learned", {}).get("remove", []) + new["remove"]}
    _trace(run, "apply", f"{len(tags)} identifiers replaced with tags; encrypted vault saved.", ok=not leftovers)
    _save(run)
    return {"deid_id": deid_id, "next": "done", "text": tagged, "summary": summary, "layer2": run["layer2"],
            "warnings": run["warnings"], "trace": run["trace"], "review_seconds": run["review_s"],
            "review_priority": run["priority"],
            # what this report taught the profile (the viewer says so; the MCP chat reply leaves it out: it holds names)
            "learned": {"hide": [{"text": x["text"], "type": x["type"]} for x in run["learned"]["remove"]],
                        "keep": run["learned"]["allow"]},
            "leftovers": [{"id": x.id, "type": x.type, "rule": x.source, "start": x.start, "end": x.end}
                          for x in leftovers]}         # positions in the tagged text; the value only in the viewer


def resolve_leftover(deid_id: str, leftover_id: str, action: str) -> dict:
    """One click on a pattern the output scan still found: "mask" hides it as a tag (learned for next
    time), "allow" keeps it as a clinical term (learned too). The text is scanned again before anything
    else can use it; a report stays blocked until no pattern is left."""
    run = _RUNS.get(deid_id)
    if run is None or run["step"] != "done":
        return _error(deid_id, "nothing to fix for this deid_id")
    hit = next((x for x in run.get("leftovers", []) if x.id == leftover_id), None)
    if hit is None or action not in ("mask", "allow"):
        return _error(deid_id, "unknown leftover or action")
    spans = run["final_spans"]
    if action == "allow":
        run["allowed"].append(hit.text)
        _stage(run, allow=[hit.text])
    else:
        start, end = to_original(run["result"].tags, hit.start, hit.end)
        span = PhiSpan(id=f"x{len(spans) + 1}", type=hit.type, start=start, end=end, text=run["text"][start:end],
                       source="user")
        _stage(run, remove=[{"text": span.text, "type": span.type}])
        spans = merge_spans(spans + [span])
    run["warnings"] = [w for w in run["warnings"] if "still in the output" not in w]
    _trace(run, "fix", f"You {'masked' if action == 'mask' else 'allowed'} a {hit.type} pattern the final scan found.")
    return _finish(run, spans)


def review_priority(run: dict, leftovers: int) -> dict:
    """Which finished reports a reviewer should check first, and why."""
    high, medium = [], []
    if leftovers:
        high.append(f"{leftovers} identifier pattern(s) left in the output")
    if run["layer2"] == "unavailable":
        high.append("name check unavailable")
    elif run["layer2"] == "skipped":
        medium.append("name check not run")
    if run["unqualified"] and run["layer2"] != "skipped":
        high.append("name-check model not qualified")
    if run["kept_names"]:
        medium.append(f"{run['kept_names']} name-like word(s) you kept visible")
    return {"level": "high" if high else "medium" if medium else "low", "reasons": high + medium}


def report_state(deid_id: str, show_text: bool = False) -> dict:
    """Where a report is up to, to resume it (MCP privlab_get_report, viewer). "items" are what still
    needs an answer: detections (review), held words (hold) or second-check finds (flags). With
    show_text (local viewer only), "confirmed" holds the spans already hidden, for highlighting."""
    run = _RUNS.get(deid_id)
    if run is None:
        return _error(deid_id, "unknown deid_id: de-identify the report first")
    text, step = run["text"], run["step"]
    base = {"deid_id": deid_id, "report_id": run["report_id"], "next": step, "warnings": run["warnings"]}
    if step == "done":
        r = run["result"]
        out = {**base, "text": r.text, "summary": r.summary, "layer2": run["layer2"], "trace": run["trace"],
               "review_priority": run["priority"]}
        if show_text:                   # local viewer only: the original and what was hidden in it
            spans = run.get("final_spans", [])
            out.update(original=text, confirmed=[_detection(text, spans, s, True) for s in spans])
        return out
    if step == "review":
        done, items = [], [_detection(text, run["spans"], s, show_text) for s in run["spans"]]
    elif step == "hold":
        done = run["confirmed"]
        items = [_detection(text, done + run["holds"], h, show_text) for h in run["holds"]]
    else:                                                   # flags
        done = run["layer1_final"]
        items = [{**_detection(text, done + run["flags"], s, show_text), "ticked": s.source != "suspect"}
                 for s in run["flags"]]
    out = {**base, "items": items}
    if show_text:
        out.update(text=text, confirmed=[_detection(text, done, s, True) for s in done])
    return out


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
    if show_text:                        # local viewer only: the identifier and where it is
        d.update(text=target.text, start=target.start, end=target.end)
    return d


def _learned_from(d: dict, span: PhiSpan, profile) -> dict:
    """A find made by a learned entry says so, and which report taught it ("Learned from report_1")."""
    if span.source == "pattern:learned":
        entry = next((r for r in profile.always_remove if r["text"].lower() == span.text.lower()), {})
        d.update(learned=True, learned_from=entry.get("from"))
    return d


def _review_time(run: dict) -> None:
    """Seconds the user spent on the step just answered (review, hold, flags), for speed tracking."""
    seconds = round(time.monotonic() - run["shown_at"], 1)
    run["review_s"][run["step"]] = seconds
    record("deid_review", seconds * 1000, report_id=run["report_id"], step=run["step"])


def _trace(run: dict, step: str, message: str, ok: bool = True) -> None:
    run["trace"].append({"step": step, "observe": message, "ok": ok})


def _stage(run: dict, allow=(), remove=()) -> None:
    """What this run will teach the profile; committed in _finish, discarded by back_to_review."""
    run["learn"]["allow"] += list(allow)
    run["learn"]["remove"] += list(remove)


def _hold_spans(text: str, spans: list[PhiSpan], allowed, prefix: str = "h") -> list[PhiSpan]:
    """The hold rule on the text with these spans masked: words that still look like a person or place.
    prefix: "h" on the review screen, "r" for the re-check at confirm, so ids never collide."""
    tagged, tags = apply_tags(text, spans)
    holds = []
    for start, end in hold_flags(tagged, allowed, tags):
        o_start, o_end = to_original(tags, start, end)
        kind = "HOSPITAL" if INSTITUTION.search(text[o_start:o_end]) else "NAME"
        holds.append(PhiSpan(id=f"{prefix}{len(holds) + 1}", type=kind, start=o_start, end=o_end,
                             text=text[o_start:o_end], source="hold"))
    return holds


def _error(deid_id: str, message: str) -> dict:
    return {"deid_id": deid_id, "next": "error", "error": message}


def _save(run: dict) -> None:
    """Encrypted state for audit and replay (contains report text and identifiers)."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = {k: run[k] for k in ("deid_id", "report_id", "step", "text", "allowed", "layer2", "warnings", "trace",
                                 "review_s", "priority")}
    for key in ("spans", "confirmed", "holds", "layer1_final", "flags", "final_spans"):
        if key in run:
            state[key] = [s.to_dict() for s in run[key]]
    (STATE_DIR / f"{run['deid_id']}.enc").write_bytes(encrypt_json(state))
