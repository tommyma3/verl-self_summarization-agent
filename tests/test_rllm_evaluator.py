from dataclasses import dataclass
import sys
import types

from self_summarization_agent.judge import JudgeDecision
from self_summarization_agent.rllm_evaluator import build_rllm_evaluator, evaluate_episode_artifacts


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


def test_build_rllm_evaluator_falls_back_when_top_level_decorator_is_missing(monkeypatch) -> None:
    class FakeSignal:
        def __init__(self, *, name: str, value: float) -> None:
            self.name = name
            self.value = value

    class FakeEvalOutput:
        def __init__(self, *, reward: float, is_correct: bool, signals: list[FakeSignal]) -> None:
            self.reward = reward
            self.is_correct = is_correct
            self.signals = signals

    class FakeEvaluator:
        def __init__(self, func) -> None:
            self.func = func

        def evaluate(self, task, episode):
            return self.func(task, episode)

    rllm_module = types.ModuleType("rllm")
    rllm_module.__path__ = []
    eval_module = types.ModuleType("rllm.eval")
    eval_module.__path__ = []
    decorator_module = types.ModuleType("rllm.eval.rollout_decorator")
    decorator_module.evaluator = FakeEvaluator
    types_module = types.ModuleType("rllm.eval.types")
    types_module.EvalOutput = FakeEvalOutput
    types_module.Signal = FakeSignal

    monkeypatch.setitem(sys.modules, "rllm", rllm_module)
    monkeypatch.setitem(sys.modules, "rllm.eval", eval_module)
    monkeypatch.setitem(sys.modules, "rllm.eval.rollout_decorator", decorator_module)
    monkeypatch.setitem(sys.modules, "rllm.eval.types", types_module)

    evaluator = build_rllm_evaluator(FakeJudge("correct_answer"))
    episode = types.SimpleNamespace(metadata={"artifacts": {"status": "completed", "final_answer": "answer"}})

    output = evaluator.evaluate(
        {"query_id": "q1", "query": "question", "answer": "answer"},
        episode,
    )

    assert output.reward == 1.0
    assert output.is_correct is True
