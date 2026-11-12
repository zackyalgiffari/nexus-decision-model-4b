import pytest
from pydantic import ValidationError

from nexus_decision.prompt import render_question, rotations
from nexus_decision.schema import Binary, Choice, DecisionRequest, Score


def test_choice_validation():
    with pytest.raises(ValidationError):
        Choice(instructions="x", options={"only": "one"})
    with pytest.raises(ValidationError):
        Choice(instructions="x", options={f"o{i}": "" for i in range(27)})


def test_request_discriminator():
    req = DecisionRequest.model_validate(
        {
            "state": {"msg": "hi"},
            "questions": {
                "a": {"type": "choice", "instructions": "pick", "options": {"x": "", "y": ""}},
                "b": {"type": "score", "instructions": "rate", "rubric": {1: "low", 2: "high"}},
                "c": {"type": "binary", "instructions": "is it spam?"},
            },
        }
    )
    assert isinstance(req.questions["a"], Choice)
    assert isinstance(req.questions["b"], Score)
    assert isinstance(req.questions["c"], Binary)


def test_render_order_and_labels():
    q = Choice(instructions="Route it.", options={"refund": "money back", "faq": "info", "human": ""})
    rq = render_question({"text": "hello"}, "route", q, order=[2, 0, 1])
    assert rq.labels == ["human", "refund", "faq"]
    assert "A) human" in rq.user_text and "B) refund: money back" in rq.user_text
    assert rq.letter_for("faq") == "C"
    with pytest.raises(ValueError):
        render_question({}, "route", q, order=[0, 0, 1])


def test_score_canonical_order():
    q = Score(instructions="Rate", rubric={3: "c", 1: "a", 2: "b"})
    assert [lbl for lbl, _ in q.option_items()] == ["1", "2", "3"]


def test_rotations():
    assert rotations(3) == [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    assert len(rotations(10, 2)) == 2
