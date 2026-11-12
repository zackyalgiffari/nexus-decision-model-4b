"""DeepSeek teacher client with an on-disk cache, a spend ledger and a hard budget cap.

* Every request is keyed by a hash of its payload and cached in SQLite, so re-running any data
  step never pays twice.
* Each billed call is priced from the response usage (cache-hit / cache-miss / output tokens) at
  the peak or off-peak rate in effect, and appended to the ledger.
* ``BudgetExceeded`` is raised before a call would start once the ledger total reaches the cap.
* ``offpeak_only=True`` blocks until DeepSeek's off-peak window.

Environment: DEEPSEEK_API_KEY (required), NEXUS_TEACHER_MODEL (default deepseek-flash),
NEXUS_BUDGET_USD (default 18).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
CACHE_DB = ROOT / "artifacts" / "cache" / "teacher.sqlite"

# USD per 1M tokens, from https://api-docs.deepseek.com/quick_start/pricing (checked 2026-11-12).
PRICES = {
    "deepseek-flash": {"hit": 0.006, "miss": 0.30, "out": 1.20},
    "deepseek-v4-pro": {"hit": 0.044, "miss": 1.32, "out": 3.96},
}
OFFPEAK_FACTOR = 0.5


def is_peak(now: datetime | None = None) -> bool:
    """Peak: 01:00-04:00 and 06:00-10:00 UTC, Monday-Friday (holidays ignored -> conservative)."""
    now = now or datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    h = now.hour
    return 1 <= h < 4 or 6 <= h < 10


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class TeacherResult:
    content: str
    reasoning: str | None
    top_logprobs: list[dict[str, float]] | None  # first output position: {token: logprob}
    usage: dict[str, int]
    cost_usd: float
    cached: bool


class Teacher:
    def __init__(
        self,
        model: str | None = None,
        budget_usd: float | None = None,
        offpeak_only: bool = False,
        db_path: Path = CACHE_DB,
    ):
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv(ROOT / ".env")
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set (put it in .env)")
        self.model = model or os.environ.get("NEXUS_TEACHER_MODEL", "deepseek-flash")
        self.budget = float(budget_usd or os.environ.get("NEXUS_BUDGET_USD", 18))
        self.offpeak_only = offpeak_only
        self.client = OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=600)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS ledger (ts TEXT, step TEXT, model TEXT, peak INT, "
                "hit INT, miss INT, out INT, cost REAL)"
            )
            self._db.commit()

    # ------------------------------------------------------------------ ledger
    def spent(self, step: str | None = None) -> float:
        q, args = "SELECT COALESCE(SUM(cost),0) FROM ledger", ()
        if step:
            q, args = q + " WHERE step=?", (step,)
        with self._lock:
            return float(self._db.execute(q, args).fetchone()[0])

    def ledger_summary(self) -> list[tuple]:
        with self._lock:
            return self._db.execute(
                "SELECT step, COUNT(*), SUM(hit), SUM(miss), SUM(out), ROUND(SUM(cost),4) "
                "FROM ledger GROUP BY step ORDER BY MIN(ts)"
            ).fetchall()

    def _price(self, usage: dict[str, int], peak: bool) -> float:
        p = PRICES.get(self.model, PRICES["deepseek-flash"])
        f = 1.0 if peak else OFFPEAK_FACTOR
        return f * (
            usage["hit"] * p["hit"] + usage["miss"] * p["miss"] + usage["out"] * p["out"]
        ) / 1e6

    # ------------------------------------------------------------------ calls
    def _wait_offpeak(self) -> None:
        while self.offpeak_only and is_peak():
            time.sleep(60)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        step: str,
        thinking: bool = False,
        max_tokens: int = 1024,
        temperature: float | None = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
        reasoning_effort: str | None = None,
        json_mode: bool = False,
        seed_tag: str = "",
    ) -> TeacherResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "extra_body": {"thinking": {"type": "enabled" if thinking else "disabled"}},
        }
        if temperature is not None and not thinking:
            payload["temperature"] = temperature
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = top_logprobs
        if thinking and reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        key = hashlib.sha256(
            json.dumps({**payload, "_tag": seed_tag}, sort_keys=True).encode()
        ).hexdigest()

        with self._lock:
            row = self._db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        if row:
            d = json.loads(row[0])
            return TeacherResult(**d, cost_usd=0.0, cached=True)

        if self.spent() >= self.budget:
            raise BudgetExceeded(f"spent ${self.spent():.2f} >= budget ${self.budget:.2f}")
        self._wait_offpeak()

        resp = None
        for attempt in range(6):
            try:
                resp = self.client.chat.completions.create(**payload)
                break
            except Exception as e:  # rate limits, timeouts, 5xx
                if attempt == 5:
                    raise
                wait = min(60, 2 ** attempt + random.random())
                print(f"[teacher] {type(e).__name__}: {e}; retry in {wait:.1f}s", flush=True)
                time.sleep(wait)

        msg = resp.choices[0].message
        u = resp.usage
        hit = int(getattr(u, "prompt_cache_hit_tokens", 0) or 0)
        miss = int(getattr(u, "prompt_cache_miss_tokens", 0) or 0)
        if hit + miss == 0:
            miss = int(u.prompt_tokens)
        usage = {"hit": hit, "miss": miss, "out": int(u.completion_tokens)}
        peak = is_peak()
        cost = self._price(usage, peak)

        tlp = None
        lp = getattr(resp.choices[0], "logprobs", None)
        if lp is not None and lp.content:
            tlp = [{t.token: t.logprob for t in pos.top_logprobs} for pos in lp.content[:4]]

        result = {
            "content": msg.content or "",
            "reasoning": getattr(msg, "reasoning_content", None),
            "top_logprobs": tlp,
            "usage": usage,
        }
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(result))
            )
            self._db.execute(
                "INSERT INTO ledger VALUES (?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), step, self.model, int(peak),
                 usage["hit"], usage["miss"], usage["out"], cost),
            )
            self._db.commit()
        return TeacherResult(**result, cost_usd=cost, cached=False)


def run_concurrent(
    fn: Callable[[Any], Any], items: Iterable[Any], workers: int = 32, desc: str = ""
) -> list[Any]:
    """Run ``fn`` over ``items`` with a thread pool. Stops cleanly on BudgetExceeded.

    Returns results in input order; failed items yield ``None``.
    """
    from tqdm import tqdm

    items = list(items)
    results: list[Any] = [None] * len(items)
    stop = threading.Event()

    def wrapped(i_item):
        i, item = i_item
        if stop.is_set():
            return i, None
        try:
            return i, fn(item)
        except BudgetExceeded as e:
            if not stop.is_set():
                print(f"\n[budget] {e} -- stopping", flush=True)
            stop.set()
            return i, None
        except Exception as e:
            print(f"\n[error] item {i}: {type(e).__name__}: {e}", flush=True)
            return i, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(wrapped, (i, it)) for i, it in enumerate(items)]
        for f in tqdm(as_completed(futs), total=len(futs), desc=desc):
            i, r = f.result()
            results[i] = r
    return results


def letter_distribution(top_logprobs: dict[str, float], n: int) -> list[float]:
    """Turn the first-position top-k logprobs into a distribution over the first ``n`` letters.

    Tokens such as "B", " B", "B)" all count toward letter B. Probability mass outside the
    returned top-k is spread uniformly over the letters that did not appear.
    """
    letters = [chr(ord("A") + i) for i in range(n)]
    mass = dict.fromkeys(letters, 0.0)
    seen_total = 0.0
    for tok, lp in top_logprobs.items():
        p = math.exp(lp)
        seen_total += p
        s = tok.strip().lstrip("(")
        if s[:1] in mass and (len(s) == 1 or not s[1:2].isalpha()):
            mass[s[:1]] += p
    residual = max(0.0, 1.0 - seen_total)
    missing = [k for k, v in mass.items() if v == 0.0]
    if missing and residual > 0:
        for k in missing:
            mass[k] = residual / len(missing)
    total = sum(mass.values())
    if total <= 0:
        return [1.0 / n] * n
    return [mass[k] / total for k in letters]
