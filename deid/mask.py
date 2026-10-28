"""
mask.py - replace identifier spans with tags, map positions back, and build sealed MaskedText.

  apply_tags(text, spans) -> (tagged_text, tags)     one code path for masking and the final output
  merge_spans(spans)       -> non-overlapping spans  overlapping finds are united, never half-masked
  to_original(tags, start, end)                      offsets in tagged text -> offsets in the original
"""
from deid.seal import make_seal
from deid.surrogates import assign_tags
from deid.types import MaskedText, PhiSpan, TagSpan


def merge_spans(spans: list[PhiSpan]) -> list[PhiSpan]:
    """Unite overlapping spans so no identifier is left partly visible. The longest span's type wins."""
    out: list[PhiSpan] = []
    for s in sorted(spans, key=lambda s: (s.start, -(s.end - s.start))):
        if out and s.start < out[-1].end:
            last = out[-1]
            if s.end > last.end:
                longest = last if (last.end - last.start) >= (s.end - s.start) else s
                out[-1] = PhiSpan(id=last.id, type=longest.type, start=last.start, end=s.end,
                                  text=last.text + s.text[last.end - s.start:], source=longest.source,
                                  score=max(last.score, s.score), status=last.status)
            continue
        out.append(s)
    return out


def apply_tags(text: str, spans: list[PhiSpan]) -> tuple[str, tuple[TagSpan, ...]]:
    spans = merge_spans(spans)
    tag_for = assign_tags(spans)
    parts, tags, pos, out_len = [], [], 0, 0
    for s in spans:
        parts.append(text[pos:s.start])
        out_len += s.start - pos
        tag = tag_for[s.id]
        tags.append(TagSpan(id=s.id, type=s.type, source=s.source, surrogate=tag,
                            orig_start=s.start, orig_end=s.end, start=out_len, end=out_len + len(tag)))
        parts.append(tag)
        out_len += len(tag)
        pos = s.end
    parts.append(text[pos:])
    return "".join(parts), tuple(tags)


def mask(deid_id: str, text: str, spans: list[PhiSpan], kept_types=(), allowed=()) -> MaskedText:
    tagged, tags = apply_tags(text, spans)
    kept_types, allowed = tuple(sorted(kept_types)), tuple(sorted(allowed))
    return MaskedText(deid_id=deid_id, text=tagged, tags=tags, kept_types=kept_types, allowed=allowed,
                      seal=make_seal(deid_id, tagged, kept_types, allowed))


def inside_tag(tags, start: int, end: int) -> bool:
    return any(start < t.end and t.start < end for t in tags)


def to_original(tags, start: int, end: int) -> tuple[int, int]:
    """Map a span outside any tag from tagged-text offsets to original offsets."""
    delta = 0
    for t in tags:
        if t.end <= start:
            delta += (t.end - t.start) - (t.orig_end - t.orig_start)
    return start - delta, end - delta
