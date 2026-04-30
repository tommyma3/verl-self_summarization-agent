from pathlib import Path

from self_summarization_agent.config import DatasetConfig
from self_summarization_agent.rllm_dataset import build_rllm_tasks


def test_build_rllm_tasks_from_decrypted_jsonl(tmp_path: Path) -> None:
    data_path = tmp_path / "browsecomp_plus_decrypted.jsonl"
    data_path.write_text(
        '{"query_id": "q1", "query": "question 1", "answer": "answer 1"}\n'
        '{"query_id": "q2", "query": "question 2", "answer": "answer 2"}\n',
        encoding="utf-8",
    )

    tasks = build_rllm_tasks(
        bc_plus_root=tmp_path,
        dataset_config=DatasetConfig(decrypted_path=str(data_path), train_limit=1),
        seed=7,
    )

    assert tasks == [{"query_id": "q1", "query": "question 1", "answer": "answer 1"}]
