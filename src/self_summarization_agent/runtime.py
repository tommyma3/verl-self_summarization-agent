from dataclasses import dataclass

from self_summarization_agent.models import RuntimeResult


@dataclass(slots=True)
class ScriptedModel:
    outputs: list[str]
    cursor: int = 0

    def generate(self, prompt: str) -> str:
        del prompt
        output = self.outputs[self.cursor]
        self.cursor += 1
        return output


def build_smoke_result() -> RuntimeResult:
    return RuntimeResult(
        query_id="smoke-q1",
        status="completed",
        final_answer="smoke answer",
        summary_turns=[],
        retrieved_docids=["smoke-doc"],
        tool_call_counts={"search": 1, "get_document": 0},
    )
