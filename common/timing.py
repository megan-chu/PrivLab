"""
timing.py - append step timings to runs/timings.log.

Measure before optimising. Context values are ids and counts only, never report text.

  with timed("presidio", report_id="PP0144"):
      ...
"""
import logging
import time
from contextlib import contextmanager

from common.config import ROOT

LOG_PATH = ROOT / "runs" / "timings.log"
_logger: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        LOG_PATH.parent.mkdir(exist_ok=True)
        _logger = logging.getLogger("privlab.timing")
        _logger.setLevel(logging.INFO)
        _logger.propagate = False
        handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s\t%(message)s"))
        _logger.addHandler(handler)
    return _logger


@contextmanager
def timed(name: str, /, **context):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        ms = (time.perf_counter() - t0) * 1000
        extra = " ".join(f"{k}={v}" for k, v in context.items())
        _get_logger().info("%s\t%.1f ms\t%s", name, ms, extra)
