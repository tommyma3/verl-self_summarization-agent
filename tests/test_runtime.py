from self_summarization_agent.backend import FakeBackend
from self_summarization_agent.runtime import EpisodeRuntime, ScriptedModel, extract_summary_output, parse_model_tool_call


def tool_output(json_text: str, thinking: str = "thinking") -> str:
    return f"<think>{thinking}</think>\n{json_text}"


def test_parse_model_tool_call_requires_completed_thinking() -> None:
    assert parse_model_tool_call('{"tool_name": "search", "arguments": {"query": "q"}}') is None


def test_parse_model_tool_call_ignores_json_inside_thinking() -> None:
    raw = '<think>{"tool_name": "finish", "arguments": {"answer": "bad"}}</think>\n{"tool_name": "search", "arguments": {"query": "good"}}'
    parsed = parse_model_tool_call(raw)
    assert parsed is not None
    payload, _ = parsed
    assert payload == {"tool_name": "search", "arguments": {"query": "good"}}


def test_extract_summary_output_uses_post_think_body() -> None:
    summary = extract_summary_output("<think>summary reasoning</think>\nsummary body")

    assert summary.thinking == "summary reasoning"
    assert summary.summary == "summary body"


def test_extract_summary_output_rejects_incomplete_thinking() -> None:
    summary = extract_summary_output("<think>unfinished summary reasoning")

    assert summary.thinking == ""
    assert summary.summary == ""


def test_action_prompt_includes_remaining_budget() -> None:
    model = ScriptedModel([tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}')])
    runtime = EpisodeRuntime(
        model=model,
        backend=FakeBackend(search_index={}, documents={}),
        context_threshold_tokens=1000,
        max_context_tokens=4096,
        max_tool_calls=5,
    )

    result = runtime.run("q1", "question")

    assert result.status == "completed"
    assert "Remaining search/get_document calls: 5" in result.generation_steps[0].prompt
    assert result.generation_steps[0].kind == "action"


def test_search_decrements_budget_and_finish_does_not() -> None:
    model = ScriptedModel(
        [
            tool_output('{"tool_name": "search", "arguments": {"query": "q"}}'),
            tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}'),
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=FakeBackend(search_index={"q": ["d1"]}, documents={"d1": "fact"}),
        context_threshold_tokens=1000,
        max_context_tokens=4096,
        max_tool_calls=2,
    )

    result = runtime.run("q1", "question")

    assert result.status == "completed"
    assert result.tool_call_counts == {"search": 1, "get_document": 0}
    assert "Remaining search/get_document calls: 2" in result.generation_steps[0].prompt
    assert "Remaining search/get_document calls: 1" in result.generation_steps[1].prompt


def test_budget_exhaustion_adds_forced_answer_step() -> None:
    model = ScriptedModel(
        [
            tool_output('{"tool_name": "search", "arguments": {"query": "q"}}'),
            tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}'),
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=FakeBackend(search_index={"q": ["d1"]}, documents={"d1": "fact"}),
        context_threshold_tokens=1000,
        max_context_tokens=4096,
        max_tool_calls=1,
    )

    result = runtime.run("q1", "question")

    assert result.status == "completed"
    assert [step.kind for step in result.generation_steps] == ["action", "forced_answer"]
    assert "You must now submit the final answer" in result.generation_steps[-1].prompt


def test_forced_answer_rejects_more_tool_calls() -> None:
    model = ScriptedModel(
        [
            tool_output('{"tool_name": "search", "arguments": {"query": "q"}}'),
            tool_output('{"tool_name": "search", "arguments": {"query": "again"}}'),
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=FakeBackend(search_index={"q": ["d1"], "again": []}, documents={"d1": "fact"}),
        context_threshold_tokens=1000,
        max_context_tokens=4096,
        max_tool_calls=1,
    )

    result = runtime.run("q1", "question")

    assert result.status == "malformed_forced_answer"
    assert result.final_answer is None
    assert result.generation_steps[-1].kind == "forced_answer"


def test_summary_is_trainable_step_but_uses_summary_state() -> None:
    model = ScriptedModel(
        [
            tool_output('{"tool_name": "search", "arguments": {"query": "first"}}'),
            tool_output('{"tool_name": "search", "arguments": {"query": "second"}}'),
            "<think>summary reasoning</think>\nsummary body",
            tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}'),
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=FakeBackend(search_index={"first": ["old"], "second": ["new"]}, documents={}),
        context_threshold_tokens=1,
        max_context_tokens=4096,
        max_tool_calls=5,
        token_counter=lambda text: text.count("new"),
    )

    result = runtime.run("q1", "question")

    assert [step.kind for step in result.generation_steps] == ["action", "action", "summary", "action"]
    assert result.generation_steps[2].is_trainable is True
    assert "summary reasoning" not in result.generation_steps[3].prompt
    assert "### SUMMARY\nsummary body" in result.generation_steps[3].prompt


def test_incomplete_summary_does_not_enter_later_context() -> None:
    model = ScriptedModel(
        [
            tool_output('{"tool_name": "search", "arguments": {"query": "first"}}'),
            tool_output('{"tool_name": "search", "arguments": {"query": "second"}}'),
            "<think>unfinished summary reasoning",
            tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}'),
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=FakeBackend(search_index={"first": ["old"], "second": ["new"]}, documents={}),
        context_threshold_tokens=1,
        max_context_tokens=4096,
        max_tool_calls=5,
        token_counter=lambda text: text.count("new"),
    )

    result = runtime.run("q1", "question")

    assert result.status == "completed"
    assert "unfinished summary reasoning" not in result.generation_steps[3].prompt
    assert "### SUMMARY" not in result.generation_steps[3].prompt
    assert result.summary_turns == []
