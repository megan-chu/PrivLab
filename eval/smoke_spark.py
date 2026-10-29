"""
smoke_spark.py - check a provider (Meta: muse-spark-1.3, or Baseten: Muse Spark Open)
answers with valid structured JSON, and record latency, tokens and which JSON mode worked.

Sends only the hand-written snippet below (no dataset text). Makes 3 live calls,
deliberately skipping the cache to test reliability; cost is a few hundred tokens.

Usage:
  python -m eval.smoke_spark --provider meta
  python -m eval.smoke_spark --provider baseten
"""
import argparse
import sys
import time
from typing import Literal

from pydantic import BaseModel

from common.config import ConfigError
from common.spark import SparkReplyError, _call_raw

SNIPPET = ("COMMENT: Patient's daughter Mary Tan called [PHONE-1] on [DATE-1]. "
           "Results discussed with Prof Lim Boon Huat and signed by Dr John Lim.")
EXPECTED = ["Mary Tan", "Lim Boon Huat", "John Lim"]

PROMPT = ("List every person, organisation and place named in the text below. Copy each name exactly "
          "as written. Ignore tags in square brackets.\n\nText:\n" + SNIPPET)


class Entity(BaseModel):
    text: str
    type: Literal["PERSON", "ORG", "PLACE"]


class Entities(BaseModel):
    entities: list[Entity]


def main(runs: int = 3) -> int:
    ap = argparse.ArgumentParser(description="Live smoke test: 3 calls to one provider.")
    ap.add_argument("--provider", required=True, choices=["meta", "baseten"])
    provider = ap.parse_args().provider
    print(f"provider={provider}")
    ok = 0
    for i in range(1, runs + 1):
        t0 = time.perf_counter()
        try:
            reply = _call_raw(PROMPT, Entities, step="smoke", use_cache=False, provider=provider)
        except ConfigError as e:
            print(e)
            return 2
        except SparkReplyError as e:
            print(f"run {i}: {e}")
            continue
        except Exception as e:   # surface API errors (auth, model id, URL) plainly
            print(f"run {i}: {type(e).__name__}: {e}")
            return 1
        ms = (time.perf_counter() - t0) * 1000
        names = [e.text for e in reply.entities]
        in_snippet = all(n in SNIPPET for n in names)
        found = [x for x in EXPECTED if any(x in n for n in names)]
        good = in_snippet and len(found) == len(EXPECTED)
        ok += good
        print(f"run {i}: {ms:.0f} ms  schema-valid  names={names}  "
              f"all copied exactly={in_snippet}  expected found={len(found)}/{len(EXPECTED)}")

    print(f"\n{ok}/{runs} runs fully correct. Mode, tokens and running total: runs/spark_usage.log")
    return 0 if ok == runs else 1


if __name__ == "__main__":
    sys.exit(main())
