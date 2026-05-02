from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from self_summarization_agent.config import load_run_config, parse_cli_overrides
from self_summarization_agent.dataset import QueryExample, load_query_examples
from self_summarization_agent.launcher_utils import build_runtime, seed_everything, serialize_runtime_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run self-summarization agent rollouts.")
    parser.add_argument("--config", required=True, help="Path to a run config.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config values as key=value.")
    return parser.parse_args()


def _explicit_examples_to_query_examples(explicit_examples: Iterable[tuple[str, str]]) -> list[QueryExample]:
    return [
        QueryExample(query_id=str(query_id), query=str(query))
        for query_id, query in explicit_examples
    ]


def _resolve_examples(
    config: Any,
    *,
    examples: list[QueryExample] | None,
    explicit_examples: Iterable[tuple[str, str]] | None,
) -> list[QueryExample]:
    if explicit_examples is not None:
        return _explicit_examples_to_query_examples(explicit_examples)
    if examples is not None:
        return list(examples)
    return load_query_examples(
        config.experiment.bc_plus_root,
        config.dataset,
        require_answers=False,
        seed=config.experiment.seed,
    )


def run_experiment(
    config: Any,
    *,
    examples: list[QueryExample] | None = None,
    backend: Any = None,
    generator: Any = None,
    explicit_examples: Iterable[tuple[str, str]] | None = None,
) -> Path:
    seed_everything(config.experiment.seed)
    loaded_examples = _resolve_examples(config, examples=examples, explicit_examples=explicit_examples)

    if backend is None:
        from self_summarization_agent.bcplus_backend import build_backend

        backend = build_backend(config.experiment.bc_plus_root, config.retrieval)
    if generator is None:
        from self_summarization_agent.generation import build_generator

        generator = build_generator(config.model)

    runtime = build_runtime(generator, backend, config.runtime)
    output_path = Path(config.experiment.output_root) / config.experiment.name / "trajectories.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for example in loaded_examples:
            result = runtime.run(example.query_id, example.query)
            record = serialize_runtime_result(result, query_text=example.query)
            if example.answer is not None:
                record["answer"] = example.answer
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

    return output_path


def main() -> None:
    args = parse_args()
    config = load_run_config(args.config, parse_cli_overrides(args.overrides))
    output_path = run_experiment(config)
    print(output_path)


if __name__ == "__main__":
    main()
