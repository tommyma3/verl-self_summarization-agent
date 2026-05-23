from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from self_summarization_agent.config import DatasetConfig
from self_summarization_agent.dataset import QueryExample, load_query_examples

LOGGER = logging.getLogger(__name__)


def _load_tokenizer(
    tokenizer_path: str | Path | None,
    *,
    trust_remote_code: bool,
) -> Any | None:
    if tokenizer_path is None:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        LOGGER.warning("transformers is not installed; rLLM dataset token filtering is disabled.")
        return None
    try:
        return AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=trust_remote_code)
    except Exception as exc:
        LOGGER.warning(
            "Failed to load tokenizer from %s; rLLM dataset token filtering is disabled. %s: %s",
            tokenizer_path,
            type(exc).__name__,
            exc,
        )
        return None


def _query_token_count(example: QueryExample, tokenizer: Any) -> int:
    return len(tokenizer.encode(example.query, add_special_tokens=False))


def _filter_examples_by_query_tokens(
    examples: list[QueryExample],
    *,
    tokenizer: Any | None,
    max_query_tokens: int | None,
) -> list[QueryExample]:
    if tokenizer is None or max_query_tokens is None or max_query_tokens <= 0:
        return examples
    filtered: list[QueryExample] = []
    skipped = 0
    for example in examples:
        if _query_token_count(example, tokenizer) <= max_query_tokens:
            filtered.append(example)
        else:
            skipped += 1
    if skipped:
        LOGGER.warning(
            "Skipped %s rLLM training task(s) whose query exceeded max_query_tokens=%s.",
            skipped,
            max_query_tokens,
        )
    return filtered


def build_rllm_tasks(
    *,
    bc_plus_root: str | Path,
    dataset_config: DatasetConfig,
    seed: int,
    tokenizer_path: str | Path | None = None,
    max_query_tokens: int | None = None,
    trust_remote_code: bool = False,
) -> list[dict[str, Any]]:
    examples = load_query_examples(
        bc_plus_root,
        dataset_config,
        require_answers=True,
        seed=seed,
    )
    tokenizer = _load_tokenizer(tokenizer_path, trust_remote_code=trust_remote_code)
    filtered_examples = _filter_examples_by_query_tokens(
        examples,
        tokenizer=tokenizer,
        max_query_tokens=max_query_tokens,
    )
    train_examples = (
        filtered_examples
        if dataset_config.train_limit is None
        else filtered_examples[: dataset_config.train_limit]
    )
    return [
        {"query_id": example.query_id, "query": example.query, "answer": example.answer}
        for example in train_examples
    ]


def build_rllm_dataset(
    *,
    bc_plus_root: str | Path,
    dataset_config: DatasetConfig,
    seed: int,
    tokenizer_path: str | Path | None = None,
    max_query_tokens: int | None = None,
    trust_remote_code: bool = False,
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
        tokenizer_path=tokenizer_path,
        max_query_tokens=max_query_tokens,
        trust_remote_code=trust_remote_code,
    )
    return DatasetRegistry.register_dataset(
        name=name,
        data=tasks,
        split=split,
        source="browsecomp-plus",
        description="Self-summarization agent training tasks for BrowseComp-Plus.",
        category="self-summarization",
    )
