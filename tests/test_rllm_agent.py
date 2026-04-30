from self_summarization_agent.backend import FakeBackend
from self_summarization_agent.config import RuntimeConfig
from self_summarization_agent.rllm_agent import OpenAICompatibleGenerator, run_self_summarization_episode


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
