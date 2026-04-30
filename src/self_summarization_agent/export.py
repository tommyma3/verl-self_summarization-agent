from self_summarization_agent.models import RuntimeResult


def build_run_record(result: RuntimeResult) -> dict[str, object]:
    return {
        "query_id": result.query_id,
        "status": result.status,
        "retrieved_docids": result.retrieved_docids,
        "result": [{"type": "output_text", "output": result.final_answer or ""}],
        "tool_call_counts": result.tool_call_counts,
    }
