from __future__ import annotations

from typing import Any

from self_summarization_agent.dataset import QueryExample
from self_summarization_agent.judge import RewardJudge
from self_summarization_agent.rewards import answer_reward, incorrect_reward


def evaluate_episode_artifacts(
    *,
    task: dict[str, Any],
    artifacts: dict[str, Any],
    judge: RewardJudge,
) -> dict[str, Any]:
    status = str(artifacts.get("status") or "")
    final_answer = artifacts.get("final_answer")
    if status != "completed" or not isinstance(final_answer, str):
        return {
            "reward": incorrect_reward(),
            "is_correct": False,
            "metrics": {"status": status, "judge_called": False},
        }

    example = QueryExample(
        query_id=str(task["query_id"]),
        query=str(task["query"]),
        answer=str(task["answer"]) if task.get("answer") is not None else None,
    )
    decision = judge.evaluate(example, status, final_answer)
    is_correct = decision.outcome == "correct_answer"
    return {
        "reward": answer_reward("correct_answer" if is_correct else "wrong_answer"),
        "is_correct": is_correct,
        "metrics": {
            "status": status,
            "judge_called": True,
            "judge_parse_error": decision.parse_error,
            "judge_outcome": decision.outcome,
        },
    }


def build_rllm_evaluator(judge: RewardJudge):
    try:
        import rllm
        from rllm.experimental.eval.types import EvalOutput, Signal
    except ImportError as exc:
        raise ImportError(
            "rLLM is required to build the evaluator. Install the rllm extra in the training environment."
        ) from exc

    @rllm.evaluator
    def score(task: dict[str, Any], episode: Any) -> Any:
        artifacts = dict(getattr(episode, "artifacts", {}) or {})
        result = evaluate_episode_artifacts(task=task, artifacts=artifacts, judge=judge)
        return EvalOutput(
            reward=result["reward"],
            is_correct=result["is_correct"],
            signals=[
                Signal(name="accuracy", value=1.0 if result["is_correct"] else 0.0),
                Signal(name="reward", value=result["reward"]),
            ],
        )

    return score
