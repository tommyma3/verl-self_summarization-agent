from typing import Literal


Outcome = Literal["correct_answer", "wrong_answer"]


def answer_reward(outcome: Outcome) -> float:
    if outcome == "correct_answer":
        return 1.0
    if outcome == "wrong_answer":
        return -1.0
    raise ValueError(f"Unknown answer outcome: {outcome}")


def incorrect_reward() -> float:
    return -1.0
