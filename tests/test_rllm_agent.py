from dataclasses import dataclass
from types import SimpleNamespace
import sys
import types

from self_summarization_agent.backend import FakeBackend
from self_summarization_agent.config import RuntimeConfig
from self_summarization_agent.rllm_agent import OpenAICompatibleGenerator, build_rllm_rollout, run_self_summarization_episode


class FakeChoiceMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeChoiceMessage(content)


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [FakeChoice(content)]


class FakeChatCompletions:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.prompts: list[str] = []

    def create(self, *, model, messages, temperature, top_p, max_tokens):
        del model, temperature, top_p, max_tokens
        self.prompts.append(messages[-1]["content"])
        return FakeResponse(self.outputs.pop(0))


class FakeChat:
    def __init__(self, outputs: list[str]) -> None:
        self.completions = FakeChatCompletions(outputs)


class FakeClient:
    def __init__(self, outputs: list[str]) -> None:
        self.chat = FakeChat(outputs)


def tool_output(json_text: str) -> str:
    return f"<think>reason</think>\n{json_text}"


def test_run_self_summarization_episode_records_all_generation_steps() -> None:
    client = FakeClient(
        [
            tool_output('{"tool_name": "search", "arguments": {"query": "q"}}'),
            tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}'),
        ]
    )
    generator = OpenAICompatibleGenerator(
        client=client,
        model="policy",
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.95,
    )

    result = run_self_summarization_episode(
        task={"query_id": "q1", "query": "question", "answer": "done"},
        generator=generator,
        backend=FakeBackend(search_index={"q": ["d1"]}, documents={"d1": "fact"}),
        runtime_config=RuntimeConfig(context_threshold_tokens=1000, max_context_tokens=4096, tool_budget=3),
    )

    assert result["artifacts"]["status"] == "completed"
    assert result["artifacts"]["final_answer"] == "done"
    assert [step["kind"] for step in result["artifacts"]["generation_steps"]] == ["action", "action"]
    assert all(step["is_trainable"] for step in result["artifacts"]["generation_steps"])


def test_run_self_summarization_episode_accepts_task_metadata_object() -> None:
    client = FakeClient([tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}')])
    generator = OpenAICompatibleGenerator(
        client=client,
        model="policy",
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.95,
    )
    task = SimpleNamespace(metadata={"query_id": "q1", "query": "question", "answer": "done"})

    result = run_self_summarization_episode(
        task=task,
        generator=generator,
        backend=FakeBackend(search_index={}, documents={}),
        runtime_config=RuntimeConfig(context_threshold_tokens=1000, max_context_tokens=4096, tool_budget=3),
    )

    assert result["artifacts"]["query_id"] == "q1"
    assert result["artifacts"]["answer"] == "done"


def test_artifacts_keep_raw_generation_steps_without_turn_records() -> None:
    client = FakeClient(
        [
            tool_output('{"tool_name": "search", "arguments": {"query": "q"}}'),
            tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}'),
        ]
    )
    generator = OpenAICompatibleGenerator(
        client=client,
        model="policy",
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.95,
    )

    result = run_self_summarization_episode(
        task={"query_id": "q1", "query": "question", "answer": "done"},
        generator=generator,
        backend=FakeBackend(search_index={"q": ["d1"]}, documents={"d1": "fact"}),
        runtime_config=RuntimeConfig(context_threshold_tokens=1000, max_context_tokens=4096, tool_budget=3),
    )

    artifacts = result["artifacts"]
    completions = [step["completion"] for step in artifacts["generation_steps"]]
    assert all("<think>reason</think>" in completion for completion in completions)
    assert "turn_records" not in artifacts


def test_openai_compatible_generator_uses_cached_tokenizer_for_counts(monkeypatch) -> None:
    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return list(text)

    monkeypatch.setattr(
        "self_summarization_agent.rllm_agent._load_cached_tokenizer",
        lambda model_path, trust_remote_code: FakeTokenizer(),
    )
    generator = OpenAICompatibleGenerator(
        client=FakeClient([]),
        model="policy",
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.95,
        tokenizer_path="tokenizer",
    )

    assert generator.count_tokens("abc") == 3


def test_rllm_rollout_returns_trainable_steps(monkeypatch) -> None:
    @dataclass
    class FakeStep:
        chat_completions: list[dict[str, str]]
        observation: str
        action: object
        model_response: str
        info: dict[str, object]
        done: bool

    @dataclass
    class FakeTrajectory:
        name: str
        task: dict[str, str]
        steps: list[FakeStep]
        info: dict[str, object]

    @dataclass
    class FakeEpisode:
        task: dict[str, str]
        trajectories: list[FakeTrajectory]
        artifacts: dict[str, object]

    rllm_module = types.ModuleType("rllm")
    rllm_module.__path__ = []
    rllm_module.rollout = lambda func: func
    agents_module = types.ModuleType("rllm.agents")
    agents_module.__path__ = []
    agent_module = types.ModuleType("rllm.agents.agent")
    agent_module.Episode = FakeEpisode
    agent_module.Trajectory = FakeTrajectory
    agent_module.Step = FakeStep
    monkeypatch.setitem(sys.modules, "rllm", rllm_module)
    monkeypatch.setitem(sys.modules, "rllm.agents", agents_module)
    monkeypatch.setitem(sys.modules, "rllm.agents.agent", agent_module)

    openai_module = types.ModuleType("openai")
    openai_module.OpenAI = lambda **kwargs: FakeClient(
        [
            tool_output('{"tool_name": "search", "arguments": {"query": "q"}}'),
            tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}'),
        ]
    )
    monkeypatch.setitem(sys.modules, "openai", openai_module)

    config = SimpleNamespace(
        model=SimpleNamespace(
            model_path="policy",
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.95,
        ),
        runtime=RuntimeConfig(context_threshold_tokens=1000, max_context_tokens=4096, tool_budget=3),
    )
    backend_calls = 0

    def build_backend():
        nonlocal backend_calls
        backend_calls += 1
        return FakeBackend(search_index={"q": ["d1"]}, documents={"d1": "fact"})

    rollout = build_rllm_rollout(
        config=config,
        backend_factory=build_backend,
    )
    assert backend_calls == 0

    episode = rollout(
        SimpleNamespace(metadata={"query_id": "q1", "query": "question", "answer": "done"}),
        {"base_url": "http://localhost:8000/v1"},
    )

    steps = episode.trajectories[0].steps
    assert episode.task["query_id"] == "q1"
    assert len(steps) == 2
    assert all(step.info["is_trainable"] for step in steps)
    assert steps[0].action["tool_name"] == "search"
    assert steps[-1].done is True
    assert episode.artifacts["generation_steps"][0]["completion"] == steps[0].model_response
    assert backend_calls == 1


def test_rllm_rollout_falls_back_when_top_level_decorator_is_missing(monkeypatch) -> None:
    @dataclass
    class FakeStep:
        chat_completions: list[dict[str, str]]
        observation: str
        action: object
        model_response: str
        metadata: dict[str, object]
        done: bool

    @dataclass
    class FakeTrajectory:
        name: str
        task: dict[str, str]
        steps: list[FakeStep]
        metadata: dict[str, object]

    @dataclass
    class FakeEpisode:
        task: dict[str, str]
        trajectories: list[FakeTrajectory]
        artifacts: dict[str, object]
        metadata: dict[str, object]

    rllm_module = types.ModuleType("rllm")
    rllm_module.__path__ = []
    agents_module = types.ModuleType("rllm.agents")
    agents_module.__path__ = []
    agent_module = types.ModuleType("rllm.agents.agent")
    agent_module.Episode = FakeEpisode
    agent_module.Trajectory = FakeTrajectory
    agent_module.Step = FakeStep
    eval_module = types.ModuleType("rllm.eval")
    eval_module.__path__ = []
    decorator_module = types.ModuleType("rllm.eval.rollout_decorator")
    decorator_module.rollout = lambda func: func
    monkeypatch.setitem(sys.modules, "rllm", rllm_module)
    monkeypatch.setitem(sys.modules, "rllm.agents", agents_module)
    monkeypatch.setitem(sys.modules, "rllm.agents.agent", agent_module)
    monkeypatch.setitem(sys.modules, "rllm.eval", eval_module)
    monkeypatch.setitem(sys.modules, "rllm.eval.rollout_decorator", decorator_module)

    openai_module = types.ModuleType("openai")
    openai_module.OpenAI = lambda **kwargs: FakeClient(
        [tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}')]
    )
    monkeypatch.setitem(sys.modules, "openai", openai_module)

    config = SimpleNamespace(
        model=SimpleNamespace(
            model_path="policy",
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.95,
        ),
        runtime=RuntimeConfig(context_threshold_tokens=1000, max_context_tokens=4096, tool_budget=3),
    )

    rollout = build_rllm_rollout(
        config=config,
        backend=FakeBackend(search_index={}, documents={}),
    )
    episode = rollout(
        {"query_id": "q1", "query": "question", "answer": "done"},
        {"base_url": "http://localhost:8000/v1"},
    )

    assert episode.trajectories[0].steps[0].metadata["is_trainable"] is True
    assert episode.metadata["artifacts"]["status"] == "completed"
