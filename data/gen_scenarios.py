"""Generate synthetic decision scenarios (state + typed questions) with the teacher.

Output: artifacts/data/scenarios.jsonl, one scenario per line:
    {"id", "source": "synthetic", "domain", "difficulty", "language", "state", "questions"}

Usage:
    uv run python -m data.gen_scenarios --calls 20            # pilot (~80 scenarios)
    uv run python -m data.gen_scenarios --calls 900 --offpeak-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re

import yaml

from data.teacher import ROOT, Teacher, run_concurrent
from nexus_decision.schema import MAX_OPTIONS, parse_question

OUT = ROOT / "artifacts" / "data" / "scenarios.jsonl"
SEEDS = ROOT / "data" / "seeds" / "domains.yaml"
PER_CALL = 4

SYSTEM = """You write realistic evaluation data for a typed decision model used inside software.
Each scenario is an application STATE plus 1-4 typed QUESTIONS about that state.

Question types (JSON):
- choice: {"type": "choice", "instructions": str, "options": {"<snake_case_label>": "<criteria: when this option is right>", ...}}  2-%d options
- score:  {"type": "score", "instructions": str, "rubric": {"1": "<level meaning>", ..., "5": "..."}}  3-7 integer levels
- binary: {"type": "binary", "instructions": "<a yes/no proposition about the state>"}

Rules:
- The state must look like real production data: concrete names, numbers, timestamps, IDs, messy text, typos where natural. No placeholders like "John Doe" or "Lorem".
- The answer must be decidable from the state and question alone (except for "ambiguous" difficulty, where reasonable people could disagree).
- Option criteria must be specific and mutually exclusive. Most choice questions should include an abstain option such as "needs_review", "unknown" or "none_of_these" with clear criteria.
- Vary option count, question types and wording. Never reveal the answer in the question.
- Do NOT include answers or labels anywhere.
Return JSON: {"scenarios": [{"state": ..., "questions": {"<snake_case_name>": {...}, ...}}, ...]}"""

USER = """Domain: {domain_id} - {description}
Question ideas (use as inspiration, do not copy): {ideas}
Difficulty: {difficulty}
State format: {state_format}
Language of the state text and question wording: {language} (option labels stay snake_case English)
Write {n} diverse scenarios. Scenario seed: {seed}"""


def _weighted(d: dict[str, float], rng: random.Random) -> str:
    keys = list(d)
    return rng.choices(keys, weights=[d[k] for k in keys])[0]


def _norm_hash(state) -> str:
    s = json.dumps(state, sort_keys=True, ensure_ascii=False).lower()
    s = re.sub(r"[\W_]+", " ", s)
    s = re.sub(r"\d+", "0", s)
    return hashlib.sha1(s.encode()).hexdigest()


def validate(scn: dict) -> dict | None:
    state, qs = scn.get("state"), scn.get("questions")
    if state in (None, "", {}, []) or not isinstance(qs, dict) or not qs:
        return None
    clean = {}
    for name, q in list(qs.items())[:4]:
        try:
            if isinstance(q, dict) and q.get("type") == "score" and isinstance(q.get("rubric"), dict):
                q = {**q, "rubric": {int(k): v for k, v in q["rubric"].items()}}
            parsed = parse_question(q)
        except Exception:
            continue
        if parsed.type == "choice" and len(parsed.options) > MAX_OPTIONS:
            continue
        clean[re.sub(r"\W+", "_", str(name)).strip("_") or "q"] = parsed.model_dump()
    if not clean:
        return None
    return {"state": state, "questions": clean}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--offpeak-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(SEEDS.read_text())
    domains = cfg["domains"]
    var = cfg["variation"]
    rng = random.Random(args.seed)
    jobs = []
    for i in range(args.calls):
        d = rng.choices(domains, weights=[x["weight"] for x in domains])[0]
        jobs.append({
            "i": i, "domain": d,
            "difficulty": _weighted(var["difficulty"], rng),
            "state_format": _weighted(var["state_format"], rng),
            "language": _weighted(var["language"], rng),
        })

    teacher = Teacher(budget_usd=args.budget_usd, offpeak_only=args.offpeak_only)

    def call(job):
        d = job["domain"]
        msg = USER.format(
            domain_id=d["id"], description=d["description"], ideas="; ".join(d["question_ideas"]),
            difficulty=job["difficulty"], state_format=job["state_format"],
            language=job["language"], n=PER_CALL, seed=f"{args.seed}-{job['i']}",
        )
        r = teacher.chat(
            [{"role": "system", "content": SYSTEM % MAX_OPTIONS}, {"role": "user", "content": msg}],
            step="gen_scenarios", thinking=False, max_tokens=6000, temperature=1.1, json_mode=True,
            seed_tag=f"{args.seed}-{job['i']}",
        )
        try:
            data = json.loads(r.content)
        except json.JSONDecodeError:
            return []
        out = []
        for k, scn in enumerate(data.get("scenarios", [])):
            v = validate(scn) if isinstance(scn, dict) else None
            if v:
                out.append({
                    "id": f"syn-{args.seed}-{job['i']}-{k}", "source": "synthetic",
                    "domain": d["id"], "difficulty": job["difficulty"],
                    "language": job["language"], **v,
                })
        return out

    results = run_concurrent(call, jobs, workers=args.workers, desc="gen_scenarios")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            seen.add(_norm_hash(json.loads(line)["state"]))
    n_new = n_q = 0
    with OUT.open("a") as f:
        for batch in results:
            for scn in batch or []:
                h = _norm_hash(scn["state"])
                if h in seen:
                    continue
                seen.add(h)
                f.write(json.dumps(scn, ensure_ascii=False) + "\n")
                n_new += 1
                n_q += len(scn["questions"])
    print(f"wrote {n_new} scenarios / {n_q} questions -> {OUT}")
    print(f"spent this step: ${teacher.spent('gen_scenarios'):.4f}  total: ${teacher.spent():.4f}")


if __name__ == "__main__":
    main()
