"""
spark.py - the ONLY module that talks to a cloud model (Muse Spark 1.3, Meta Model API).

Safety: call() accepts only MaskedText / DeidResult built inside deid/, never a str,
and re-scans the outgoing text with the pattern recognisers before sending.
Cost ($150 of credits): every reply is cached per step and text; every call is
logged to runs/spark_usage.log with tokens and a running total (no report text).

  call(safe, instructions=..., schema=MyModel, step="names", report_id="PP0001") -> MyModel

Verified by eval/smoke_spark.py (27 Oct): strict json_schema output works; the
response shape is OpenAI-style (choices[0].message.content, usage.prompt_tokens /
completion_tokens). Latency ~8-11 s per call; output tokens ~1.3-1.7k even for a
tiny reply (likely hidden reasoning), so output tokens dominate cost.
"""
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, ValidationError

from common.config import ROOT, load_config
from common.timing import timed

CACHE_DIR = ROOT / "runs" / "cache"
USAGE_LOG = ROOT / "runs" / "spark_usage.log"
USAGE_TOTALS = ROOT / "runs" / "spark_usage.json"
SGT = timezone(timedelta(hours=8))

# PLACEHOLDER: USD per 1M (input, output) tokens. Fill in from the Meta Model API docs.
PRICE_PER_M = (None, None)
WARN_AT_USD = (75, 120)

SYSTEM = ("You are a careful assistant for pathology report processing. "
          "Reply with JSON only, matching the requested schema. Do not give medical advice.")

_client = None


class PhiLeakError(RuntimeError):
    pass


class SparkReplyError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def call(safe, *, instructions: str, schema: type[BaseModel], step: str,
         report_id: str | None = None, effort: str = "minimal") -> BaseModel:
    """Send de-identified or masked text to Muse Spark and return a validated reply."""
    from deid.types import DeidResult, MaskedText   # deid imports nothing from here

    if not isinstance(safe, (MaskedText, DeidResult)):
        raise TypeError(f"spark.call accepts MaskedText or DeidResult, not {type(safe).__name__}")
    # Day 2: verify safe.seal here (HMAC with PRIVLAB_SEAL_KEY) once deid/ creates seals.
    check_no_leak(safe.text)
    prompt = f"{instructions}\n\nText:\n{safe.text}"
    return _call_raw(prompt, schema, step=step, report_id=report_id, effort=effort)


def check_no_leak(text: str) -> None:
    """Refuse to send text in which the pattern recognisers still find an identifier."""
    from deid.patterns import find_pattern_phi

    hits = find_pattern_phi(text)
    if hits:
        kinds = ", ".join(sorted({h.type for h in hits}))
        raise PhiLeakError(f"refusing to send: {len(hits)} identifier(s) still in text ({kinds})")


# --------------------------------------------------------------------------
# Raw call: cache, structured output, usage log
# --------------------------------------------------------------------------
def _call_raw(prompt: str, schema: type[BaseModel], *, step: str, report_id: str | None = None,
              use_cache: bool = True, effort: str = "minimal") -> BaseModel:
    """Only call() and eval/smoke_spark.py (hand-written text) may use this directly.

    effort: reasoning_effort sent to the API. Muse Spark always reasons ("none" is
    rejected); reasoning tokens are billed as output, so default to "minimal".
    """
    settings = load_config()
    settings.require("meta_api_key")
    key = hashlib.sha256(json.dumps(
        [settings.meta_model, step, effort, SYSTEM, prompt, schema.model_json_schema()], sort_keys=True
    ).encode("utf-8")).hexdigest()
    cache_path = CACHE_DIR / step / f"{key}.json"

    if use_cache and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        _log_usage(step, report_id, effort, cached["mode"], 0, 0, cached=True)
        return schema.model_validate(cached["reply"])

    with timed("spark", step=step, report_id=report_id, effort=effort):
        content, usage, mode = _request(settings, prompt, schema, effort)
    try:
        reply = schema.model_validate_json(_strip_fence(content))
    except ValidationError as e:
        raise SparkReplyError(f"reply does not match {schema.__name__}: {e}") from e

    in_tok, out_tok = usage
    _log_usage(step, report_id, effort, mode, in_tok, out_tok, cached=False)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"mode": mode, "reply": reply.model_dump()}, indent=2), encoding="utf-8")
    return reply


def _request(settings, prompt: str, schema: type[BaseModel], effort: str) -> tuple[str, tuple[int, int], str]:
    """Try strict json_schema output first; fall back to json_object if the API rejects it."""
    import openai

    global _client
    if _client is None:
        # max_retries=1: at most one retry on a network error (CLAUDE.md cost rules)
        _client = openai.OpenAI(base_url=settings.meta_api_base_url, api_key=settings.meta_api_key,
                                max_retries=1, timeout=60)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    formats = [
        ("json_schema", {"type": "json_schema", "json_schema": {
            "name": schema.__name__, "strict": True, "schema": _strict_schema(schema)}}),
        ("json_object", {"type": "json_object"}),
    ]
    last_error = None
    for mode, response_format in formats:
        msgs = messages if mode == "json_schema" else [
            messages[0],
            {"role": "user", "content": f"{prompt}\n\nJSON schema:\n{json.dumps(schema.model_json_schema())}"},
        ]
        try:
            resp = _client.chat.completions.create(model=settings.meta_model, messages=msgs,
                                                   response_format=response_format, temperature=0,
                                                   reasoning_effort=effort)
        except openai.BadRequestError as e:   # format not supported: try the next one
            last_error = e
            continue
        usage = getattr(resp, "usage", None)
        tokens = (getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0)
        return resp.choices[0].message.content, tokens, mode
    raise SparkReplyError(f"no supported response_format: {last_error}")


def _strict_schema(schema: type[BaseModel]) -> dict:
    """OpenAI strict mode needs additionalProperties: false on every object."""
    s = schema.model_json_schema()

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(s)
    return s


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text


def _log_usage(step: str, report_id: str | None, effort: str, mode: str, in_tok: int, out_tok: int,
               cached: bool) -> None:
    USAGE_LOG.parent.mkdir(exist_ok=True)
    totals = json.loads(USAGE_TOTALS.read_text(encoding="utf-8")) if USAGE_TOTALS.exists() else \
        {"calls": 0, "cached_hits": 0, "input_tokens": 0, "output_tokens": 0}
    if cached:
        totals["cached_hits"] += 1
    else:
        totals["calls"] += 1
        totals["input_tokens"] += in_tok
        totals["output_tokens"] += out_tok
    cost = estimate_usd(totals["input_tokens"], totals["output_tokens"])
    USAGE_TOTALS.write_text(json.dumps(totals, indent=2), encoding="utf-8")

    line = (f"{datetime.now(SGT).isoformat(timespec='seconds')}\t{step}\t{report_id or '-'}\t"
            f"effort={effort}\t{mode}\t"
            f"{'cached' if cached else 'live'}\tin={in_tok}\tout={out_tok}\t"
            f"total_in={totals['input_tokens']}\ttotal_out={totals['output_tokens']}\t"
            f"total_usd={'?' if cost is None else f'{cost:.2f}'}\n")
    with open(USAGE_LOG, "a", encoding="utf-8") as f:
        f.write(line)
    if cost is not None:
        for limit in WARN_AT_USD:
            if cost >= limit:
                print(f"WARNING: Muse Spark spend estimate ${cost:.2f} has passed ${limit}", file=sys.stderr)


def estimate_usd(input_tokens: int, output_tokens: int) -> float | None:
    price_in, price_out = PRICE_PER_M
    if price_in is None or price_out is None:
        return None
    return input_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out
