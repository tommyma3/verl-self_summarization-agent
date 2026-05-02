from __future__ import annotations

import argparse
from typing import Any

from self_summarization_agent.config import load_train_config, parse_cli_overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the self-summarization agent with rLLM/verl.")
    parser.add_argument("--config", required=True, help="Path to an rLLM training config.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config values as key=value.")
    return parser.parse_args()


def build_trainer(config: Any) -> Any:
    try:
        from rllm.experimental.unified_trainer import AgentTrainer
    except ImportError as exc:
        raise ImportError(
            "rLLM is required for train_rllm.py. Install the rllm extra in the training environment."
        ) from exc

    from self_summarization_agent.bcplus_backend import build_backend
    from self_summarization_agent.generation import build_generator
    from self_summarization_agent.judge import RewardJudge
    from self_summarization_agent.rllm_agent import build_rllm_rollout
    from self_summarization_agent.rllm_dataset import build_rllm_tasks
    from self_summarization_agent.rllm_evaluator import build_rllm_evaluator

    tasks = build_rllm_tasks(
        bc_plus_root=config.experiment.bc_plus_root,
        dataset_config=config.dataset,
        seed=config.experiment.seed,
    )
    backend = build_backend(config.experiment.bc_plus_root, config.retrieval)
    judge = RewardJudge(build_generator(config.model, judge_config=config.judge))
    agent_flow = build_rllm_rollout(config=config, backend=backend)
    evaluator = build_rllm_evaluator(judge)
    trainer_config = {
        "backend": config.rllm.backend,
        "algorithm": config.rllm.algorithm,
        "train_batch_size": config.rllm.train_batch_size,
        "rollout_group_size": config.rllm.rollout_group_size,
        "max_prompt_length": config.rllm.max_prompt_length,
        "max_response_length": config.rllm.max_response_length,
        "total_training_steps": config.rllm.total_training_steps,
        "save_freq": config.rllm.save_freq,
        "eval_freq": config.rllm.eval_freq,
        "project_name": config.rllm.project_name,
        "experiment_name": config.rllm.experiment_name,
    }

    return AgentTrainer(
        backend=config.rllm.backend,
        agent_flow=agent_flow,
        evaluator=evaluator,
        config=trainer_config,
        train_dataset=tasks,
    )


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config, parse_cli_overrides(args.overrides))
    if config.training.backend != "rllm_verl":
        raise ValueError(f"train_rllm.py requires training.backend='rllm_verl', got {config.training.backend!r}")
    trainer = build_trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
