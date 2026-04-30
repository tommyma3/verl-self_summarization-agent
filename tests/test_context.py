from self_summarization_agent.context import ContextManager
from self_summarization_agent.models import EpisodeState, Message, ToolCallRecord, ToolRound


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
