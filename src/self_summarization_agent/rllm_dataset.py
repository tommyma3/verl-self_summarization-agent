from __future__ import annotations

from pathlib import Path
from typing import Any

from self_summarization_agent.config import DatasetConfig
from self_summarization_agent.dataset import load_query_examples


def build_rllm_tasks(
    *,
    bc_plus_root: str | Path,
    dataset_config: DatasetConfig,
    seed: int,
) -> list[dict[str, Any]]:
    examples = load_query_examples(
        bc_plus_root,
        dataset_config,
        require_answers=True,
        seed=seed,
    )
    train_examples = examples if dataset_config.train_limit is None else examples[: dataset_config.train_limit]
    return [
        {"query_id": example.query_id, "query": example.query, "answer": example.answer}
        for example in train_examples
    ]


def build_rllm_dataset(
    *,
    bc_plus_root: str | Path,
    dataset_config: DatasetConfig,
    seed: int,
    name: str = "self_summarization_agent_bcplus",
    split: str = "train",
) -> Any:
    """Build and register an rLLM Dataset for verl training."""
    try:
        from rllm.data.dataset import DatasetRegistry
    except ImportError as exc:
        raise ImportError(
            "rLLM is required to build the verl training dataset. Install the rllm extra in the training environment."
        ) from exc

    tasks = build_rllm_tasks(
        bc_plus_root=bc_plus_root,
        dataset_config=dataset_config,
        seed=seed,
    )
    return DatasetRegistry.register_dataset(
        name=name,
        data=tasks,
        split=split,
        source="browsecomp-plus",
        description="Self-summarization agent training tasks for BrowseComp-Plus.",
        category="self-summarization",
    )
