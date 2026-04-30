def build_budget_block(remaining_tool_calls: int) -> str:
    return f"### TOOL_BUDGET\nRemaining search/get_document calls: {remaining_tool_calls}"


def build_system_prompt() -> str:
    return """You are a deep research AI agent.

Your response must be exactly one JSON object for one tool call.
After any internal reasoning, the final visible action must be only one JSON tool call.

Available tools:
- search: find candidate documents for a search query. Use {"tool_name": "search", "arguments": {"query": "..."}}
- get_document: read one retrieved document by id. Use {"tool_name": "get_document", "arguments": {"doc_id": "..."}}
- finish: submit the final answer. Use {"tool_name": "finish", "arguments": {"answer": "..."}}

Tool strategy:
- Start with search unless the answer is already fully supported by the conversation.
- Use get_document only with docid values returned by search.
- Call finish only when the evidence is sufficient."""


def build_summary_system_prompt() -> str:
    return """You are a context summarization AI agent.

Your task is to summarize the previous research context so another step of the same agent can continue the task.
Return only the summary text after thinking."""


def build_summary_prompt() -> str:
    return (
        "Write a clean summary containing only the essential information needed "
        "to continue solving the task. Preserve normalized facts, unresolved "
        "questions, evidence-grounded facts tied to doc_id citations, and useful next steps."
    )


def build_normal_next_action_prompt() -> str:
    return (
        "### NEXT_ACTION\n"
        "Return exactly one JSON object for the next tool call. "
        "After any thinking, the final visible action must be only the JSON object. "
        "Return one action only."
    )


def build_forced_answer_prompt() -> str:
    return (
        "### NEXT_ACTION\n"
        "You must now submit the final answer. "
        "Return exactly one JSON object: "
        '{"tool_name": "finish", "arguments": {"answer": "..."}}. '
        "Do not call search or get_document."
    )
