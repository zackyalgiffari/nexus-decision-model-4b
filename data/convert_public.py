"""Convert public intent datasets into typed decision scenarios with gold labels.

Sources (both on the Hugging Face Hub):
  * PolyAI/banking77  (CC-BY-4.0)  - 77 banking intents
  * clinc/clinc_oos "plus" (CC-BY-3.0) - 150 intents + out-of-scope

Each text becomes one scenario with one question:
  * choice (80%): the gold intent plus 3-11 distractors (half "hard": sharing a word with the gold
    label), always with a ``none_of_these`` option. In 15% of in-scope cases the gold intent is
    removed, so the answer becomes ``none_of_these``. CLINC out-of-scope texts answer
    ``none_of_these`` too.
  * binary (20%): "The message is about <intent>." with 50% positives.

Output: artifacts/data/public.jsonl (scenario format + "gold" + "split").
"""

from __future__ import annotations

import argparse
import json
import random

from huggingface_hub import HfApi

from data.teacher import ROOT

OUT = ROOT / "artifacts" / "data" / "public.jsonl"
NONE = "none_of_these"

SOURCES = {
    "banking77": {
        "repo": "PolyAI/banking77", "config": "default", "text": "text", "label": "label",
        "splits": {"train": "train", "test": "test"},
        "state": lambda t: {"channel": "mobile_banking_chat", "customer_message": t},
        "instructions": "Which intent best describes the customer's message?",
        "binary": "The customer's message is about: {intent}.",
        "oos": None,
    },
    "clinc150": {
        "repo": "clinc/clinc_oos", "config": "plus", "text": "text", "label": "intent",
        "splits": {"train": "train", "test": "test"},
        "state": lambda t: {"product": "general_virtual_assistant", "user_utterance": t},
        "instructions": (
            "Which intent should the assistant handle this utterance as? Choose none_of_these if "
            "no listed intent fits."
        ),
        "binary": "The user's utterance is a request of type: {intent}.",
        "oos": "oos",
    },
}


def humanize(label: str) -> str:
    return label.replace("_", " ").strip()


def label_names(repo: str, config: str, column: str) -> list[str]:
    card = HfApi().dataset_info(repo).card_data.to_dict()
    infos = card["dataset_info"]
    infos = infos if isinstance(infos, list) else [infos]
    info = next((i for i in infos if i.get("config_name", "default") == config), infos[0])
    for feat in info["features"]:
        if feat["name"] == column:
            names = feat["dtype"]["class_label"]["names"]
            return [names[k] for k in sorted(names, key=int)] if isinstance(names, dict) else names
    raise KeyError(column)


def load_split(repo: str, config: str, split: str) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo, f"{config}/{split}/0000.parquet", repo_type="dataset",
        revision="refs/convert/parquet",
    )
    return pq.read_table(path).to_pylist()


def make_question(text, gold, names, src, rng: random.Random):
    in_scope = [n for n in names if n != src["oos"]]
    if gold != src["oos"] and rng.random() < 0.2:
        positive = rng.random() < 0.5
        intent = gold if positive else rng.choice([n for n in in_scope if n != gold])
        q = {"type": "binary", "instructions": src["binary"].format(intent=humanize(intent))}
        return q, ("yes" if positive else "no")

    k = rng.randint(4, 12)
    drop_gold = gold == src["oos"] or rng.random() < 0.15
    pool = [n for n in in_scope if n != gold]
    gold_words = set(gold.split("_")) - {"card", "my", "the", "a"} if gold != src["oos"] else set()
    hard = [n for n in pool if gold_words & set(n.split("_"))]
    rng.shuffle(hard)
    n_dist = k - 1 if drop_gold else k - 2
    distractors = hard[: n_dist // 2]
    rest = [n for n in pool if n not in distractors]
    distractors += rng.sample(rest, n_dist - len(distractors))
    labels = distractors + ([] if drop_gold else [gold])
    rng.shuffle(labels)
    options = {lbl: f"The message is about: {humanize(lbl)}." for lbl in labels}
    options[NONE] = "None of the listed intents fits the message."
    q = {"type": "choice", "instructions": src["instructions"], "options": options}
    return q, (NONE if drop_gold else gold)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-source", type=int, default=1500)
    ap.add_argument("--test-per-source", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with OUT.open("w") as f:
        for name, src in SOURCES.items():
            names = label_names(src["repo"], src["config"], src["label"])
            for split, hub_split in src["splits"].items():
                ds = load_split(src["repo"], src["config"], hub_split)
                want = args.train_per_source if split == "train" else args.test_per_source
                idx = rng.sample(range(len(ds)), min(want, len(ds)))
                for i in idx:
                    row = ds[i]
                    gold = names[int(row[src["label"]])]
                    q, ans = make_question(row[src["text"]], gold, names, src, rng)
                    rec = {
                        "id": f"{name}-{split}-{i}", "source": name, "split": split,
                        "domain": "intent_" + name, "difficulty": "public", "language": "English",
                        "state": src["state"](row[src["text"]]),
                        "questions": {"intent": q}, "gold": {"intent": ans},
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
    print(f"wrote {n} public scenarios -> {OUT}")


if __name__ == "__main__":
    main()
