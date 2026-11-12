"""Build a tiny dataset for the Phase 0 spike from the public gold-only splits.

Think records get a placeholder reasoning string; they only exercise the LM-loss code path.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = [json.loads(x) for x in (ROOT / "artifacts/data/train.jsonl").read_text().splitlines()]
rng = random.Random(0)
rng.shuffle(src)
out = src[:48]
for r in src[48:64]:
    letter_idx = max(range(len(r["target"])), key=r["target"].__getitem__)
    reasoning = (
        "The state is a single short message. I compare it against each option's criteria. "
        f"Option {r['labels'][letter_idx]} matches the request best; the others do not fit. " * 6
    )
    out.append({**r, "id": r["id"].replace("fast0", "think"), "kind": "think", "reasoning": reasoning})
d = ROOT / "artifacts/spike"
d.mkdir(parents=True, exist_ok=True)
(d / "train.jsonl").write_text("\n".join(json.dumps(r) for r in out) + "\n")
(d / "val.jsonl").write_text("\n".join(json.dumps(r) for r in src[64:84]) + "\n")
print(len(out), "spike records")
