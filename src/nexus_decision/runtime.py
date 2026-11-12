"""Backend-agnostic decision runtime.

A backend only needs to expose three things:

* ``tokenizer`` with ``apply_chat_template`` and ``encode``
* ``next_token_logits(text) -> np.ndarray`` giving full-vocabulary logits after ``text``
* ``generate_until(text, stop, max_tokens, temperature, top_p, seed) -> str``

Everything else (rendering, letter readout, rotation debiasing, calibration) lives here so the MLX
and Transformers backends behave identically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .calibrate import entropy, load_config, softmax
from .prompt import (
    THINK_CLOSE,
    RenderedQuestion,
    close_think,
    generation_prefix,
    render_question,
    rotations,
)
from .schema import LETTERS, Answer, DecisionRequest, DecisionResponse, Mode, Score


class Backend(Protocol):
    tokenizer: Any
    model_id: str

    def next_token_logits(self, text: str) -> np.ndarray: ...

    def generate_until(
        self, text: str, stop: str, max_tokens: int, temperature: float, top_p: float, seed: int
    ) -> str: ...


def letter_token_ids(tokenizer: Any, n: int = len(LETTERS)) -> list[int]:
    """Token id of each answer letter. Each letter must encode to exactly one token."""
    ids = []
    for letter in LETTERS[:n]:
        enc = tokenizer.encode(letter, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"letter {letter!r} is not a single token: {enc}")
        ids.append(enc[0])
    return ids


class NexusDecision:
    """High-level API.

    >>> nd = NexusDecision.from_pretrained("<user>/nexus-decision-4b", backend="mlx")
    >>> nd.decide({"state": {...}, "questions": {"route": {"type": "choice", ...}}})
    """

    def __init__(self, backend: Backend, config_path: str | Path | None = None):
        self.backend = backend
        self.config = load_config(config_path)
        self._letter_ids = letter_token_ids(backend.tokenizer)

    @classmethod
    def from_pretrained(cls, path: str, backend: str = "mlx", **kwargs: Any) -> NexusDecision:
        if backend == "mlx":
            from .runtime_mlx import MLXBackend

            b = MLXBackend(path, **kwargs)
        elif backend in ("hf", "transformers"):
            from .runtime_hf import HFBackend

            b = HFBackend(path, **kwargs)
        else:
            raise ValueError(f"unknown backend {backend!r}")
        return cls(b, config_path=b.local_dir / "nexus_config.json")

    # ------------------------------------------------------------------ core readout
    def letter_logits(self, text: str, n: int) -> np.ndarray:
        logits = self.backend.next_token_logits(text)
        return np.asarray(logits, dtype=np.float64)[self._letter_ids[:n]]

    def question_logits(
        self, state: Any, name: str, question, order: list[int] | None, mode: Mode, **gen: Any
    ) -> tuple[RenderedQuestion, np.ndarray, str | None]:
        """Raw (uncalibrated) letter logits in *display order* for one rendering."""
        rq = render_question(state, name, question, order)
        prefix = generation_prefix(self.backend.tokenizer, rq.user_text, mode)
        reasoning = None
        if mode == "think":
            out = self.backend.generate_until(
                prefix,
                stop=THINK_CLOSE,
                max_tokens=gen.get("max_think_tokens", 1024),
                temperature=gen.get("temperature", 0.6),
                top_p=gen.get("top_p", 0.95),
                seed=gen.get("seed", 0),
            )
            reasoning = out.split(THINK_CLOSE)[0].strip()
            prefix = prefix + close_think(reasoning)
        return rq, self.letter_logits(prefix, len(rq.labels)), reasoning

    def answer(
        self,
        state: Any,
        name: str,
        question,
        mode: Mode = "fast",
        debias: bool = False,
        n_rotations: int | None = None,
        **gen: Any,
    ) -> Answer:
        items = question.option_items()
        labels = [label for label, _ in items]
        n = len(labels)
        temp = float(self.config["temperature"][mode])

        # Reasoning refers to letters, so think mode always uses a single (canonical) order.
        orders = rotations(n, n_rotations) if (debias and mode == "fast") else [list(range(n))]
        acc = np.zeros(n)
        reasoning = None
        for order in orders:
            rq, z, reasoning = self.question_logits(state, name, question, order, mode, **gen)
            p = softmax(z, temp)
            for i, label in enumerate(rq.labels):
                acc[labels.index(label)] += p[i]
        probs = acc / acc.sum()

        top = int(np.argmax(probs))
        ans = Answer(
            question=name,
            type=question.type,
            choice=labels[top],
            probabilities={label: float(p) for label, p in zip(labels, probs)},
            confidence=float(probs[top]),
            entropy=entropy(probs),
            mode=mode,
            reasoning=reasoning,
            model_id=self.backend.model_id,
        )
        if isinstance(question, Score):
            ans.expected_score = float(sum(int(lbl) * p for lbl, p in zip(labels, probs)))
        if question.type == "binary":
            ans.p_yes = float(probs[labels.index("yes")])
        return ans

    def decide(
        self,
        request: DecisionRequest | dict,
        mode: Mode = "fast",
        debias: bool = False,
        n_rotations: int | None = None,
        **gen: Any,
    ) -> DecisionResponse:
        req = request if isinstance(request, DecisionRequest) else DecisionRequest.model_validate(request)
        answers = {
            name: self.answer(req.state, name, q, mode, debias, n_rotations, **gen)
            for name, q in req.questions.items()
        }
        return DecisionResponse(answers=answers, model_id=self.backend.model_id)
