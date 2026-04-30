from dataclasses import dataclass

from self_summarization_agent.judge import JudgeDecision
from self_summarization_agent.rllm_evaluator import evaluate_episode_artifacts


@dataclass(slots=True)
class FakeJudge:
    outcome: str

    def evaluate(self, example, status: str, response: str) -> JudgeDecision:
        return JudgeDecision(
            outcome=self.outcome,
            judge_prompt="judge prompt",
            judge_response="correct: yes" if self.outcome == "correct_answer" else "correct: no",
            parse_error=False,
        )


def test_evaluator_returns_positive_reward_for_correct_answer() -> None:
    output = evaluate_episode_artifacts(
        task={"query_id": "q1", "query": "question", "answer": "answer"},
        artifacts={"status": "completed", "final_answer": "answer"},
        judge=FakeJudge("correct_answer"),
    )

    assert output["reward"] == 1.0
    assert output["is_correct"] is True
    assert output["metrics"]["status"] == "completed"


def test_evaluator_returns_negative_reward_for_malformed_action_without_judge() -> None:
    output = evaluate_episode_artifacts(
        task={"query_id": "q1", "query": "question", "answer": "answer"},
        artifacts={"status": "malformed_tool_call", "final_answer": None},
        judge=FakeJudge("correct_answer"),
    )

    assert output["reward"] == -1.0
    assert output["is_correct"] is False
    assert output["metrics"]["judge_called"] is False
