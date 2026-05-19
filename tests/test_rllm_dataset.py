from pathlib import Path
import sys
import types

from self_summarization_agent.config import DatasetConfig
from self_summarization_agent.rllm_dataset import build_rllm_dataset, build_rllm_tasks


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


def test_build_rllm_dataset_registers_tasks(monkeypatch, tmp_path: Path) -> None:
    data_path = tmp_path / "browsecomp_plus_decrypted.jsonl"
    data_path.write_text(
        '{"query_id": "q1", "query": "question 1", "answer": "answer 1"}\n',
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class FakeDatasetRegistry:
        @staticmethod
        def register_dataset(**kwargs):
            captured.update(kwargs)
            return "dataset"

    rllm_module = types.ModuleType("rllm")
    rllm_module.__path__ = []
    data_module = types.ModuleType("rllm.data")
    data_module.__path__ = []
    dataset_module = types.ModuleType("rllm.data.dataset")
    dataset_module.DatasetRegistry = FakeDatasetRegistry
    monkeypatch.setitem(sys.modules, "rllm", rllm_module)
    monkeypatch.setitem(sys.modules, "rllm.data", data_module)
    monkeypatch.setitem(sys.modules, "rllm.data.dataset", dataset_module)

    dataset = build_rllm_dataset(
        bc_plus_root=tmp_path,
        dataset_config=DatasetConfig(decrypted_path=str(data_path)),
        seed=7,
        name="training",
        split="train",
    )

    assert dataset == "dataset"
    assert captured["name"] == "training"
    assert captured["split"] == "train"
    assert captured["data"] == [{"query_id": "q1", "query": "question 1", "answer": "answer 1"}]
