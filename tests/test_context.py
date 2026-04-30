from self_summarization_agent.context import ContextManager
from self_summarization_agent.launcher_utils import serialize_runtime_result
from self_summarization_agent.models import EpisodeState, GenerationStep, Message, RuntimeResult, ToolCallRecord, ToolRound


def test_summary_context_includes_remaining_budget() -> None:
    manager = ContextManager(token_counter=lambda text: len(text.split()), max_context_tokens=1000)
    state = EpisodeState(query_id="q1", user_prompt="question", context_threshold_tokens=10)
    state.rounds.append(
        ToolRound(
            assistant_message=Message(role="assistant", content='{"tool_name": "search", "arguments": {"query": "q"}}'),
            tool_call=ToolCallRecord(tool_name="search", arguments={"query": "q"}, raw_output="{}"),
            tool_result=Message(role="tool", content='[{"docid": "d1"}]'),
        )
    )

    prompt = manager.build_summary_context(state, remaining_tool_calls=3)

    assert "### TOOL_BUDGET" in prompt
    assert "Remaining search/get_document calls: 3" in prompt
    assert "context summarization AI agent" in prompt


def test_serialized_runtime_result_includes_generation_steps() -> None:
    result = RuntimeResult(
        query_id="q1",
        status="completed",
        final_answer="done",
        summary_turns=[],
        retrieved_docids=[],
        generation_steps=[
            GenerationStep(
                step_id="step-1",
                kind="action",
                prompt="prompt",
                completion="completion",
                parsed_tool_name="finish",
            )
        ],
    )

    serialized = serialize_runtime_result(result, query_text="question")

    assert serialized["generation_steps"] == [
        {
            "step_id": "step-1",
            "kind": "action",
            "prompt": "prompt",
            "completion": "completion",
            "parsed_tool_name": "finish",
            "is_trainable": True,
        }
    ]
