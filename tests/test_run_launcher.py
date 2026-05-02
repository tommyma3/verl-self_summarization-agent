from __future__ import annotations

import json
from pathlib import Path

from self_summarization_agent.config import load_run_config
from self_summarization_agent.dataset import QueryExample
from self_summarization_agent.run_launcher import run_experiment


class FakeGenerator:
    def generate(self, prompt: str) -> str:
        del prompt
        return '<think>ok</think> {"tool_name": "finish", "arguments": {"answer": "done"}}'

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class FakeBackend:
    def search(self, query: str) -> list[dict[str, str]]:
        del query
        return []

    def get_document(self, doc_id: str) -> str:
        raise KeyError(doc_id)


def test_run_experiment_writes_trajectory_jsonl(tmp_path) -> None:
    config = load_run_config(Path("configs/run/default.yaml"))
    config.experiment.output_root = str(tmp_path)

    output_path = run_experiment(
        config,
        explicit_examples=[("q1", "question")],
        backend=FakeBackend(),
        generator=FakeGenerator(),
    )

    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["query_id"] == "q1"
    assert record["query"] == "question"


def test_run_experiment_preserves_answers_in_jsonl(tmp_path) -> None:
    config = load_run_config(Path("configs/run/default.yaml"))
    config.experiment.output_root = str(tmp_path)

    output_path = run_experiment(
        config,
        examples=[QueryExample(query_id="q1", query="question", answer="gold")],
        backend=FakeBackend(),
        generator=FakeGenerator(),
    )

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["answer"] == "gold"
    assert records[0]["status"] == "completed"
    assert records[0]["final_answer"] == "done"
