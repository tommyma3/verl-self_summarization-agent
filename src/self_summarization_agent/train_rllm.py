from __future__ import annotations

import argparse
from typing import Any

from self_summarization_agent.config import load_train_config, parse_cli_overrides


RLLM_CONFIG_MODULE = "rllm.experimental.config"
RLLM_CONFIG_NAME = "unified"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the self-summarization agent with rLLM/verl.")
    parser.add_argument("--config", required=True, help="Path to an rLLM training config.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config values as key=value.")
    return parser.parse_args()


def build_trainer_config(config: Any) -> Any:
    try:
        from hydra import compose, initialize_config_module
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf, open_dict
    except ImportError as exc:
        raise ImportError(
            "Hydra and OmegaConf are required to compose the rLLM/verl trainer config."
        ) from exc

    if GlobalHydra.instance().is_initialized():
        trainer_config = compose(config_name=RLLM_CONFIG_NAME)
    else:
        with initialize_config_module(config_module=RLLM_CONFIG_MODULE, version_base=None):
            trainer_config = compose(config_name=RLLM_CONFIG_NAME)

    rollout_n = config.rllm.rollout_group_size
    sampling = {
        "temperature": config.rollout.temperature,
        "top_p": config.rollout.top_p,
        "top_k": -1,
        "max_tokens": config.rllm.max_response_length,
    }
    updates = {
        "model.name": config.model.model_path,
        "data.train_batch_size": config.rllm.train_batch_size,
        "data.max_prompt_length": config.rllm.max_prompt_length,
        "data.max_response_length": config.rllm.max_response_length,
        "data.trust_remote_code": config.model.trust_remote_code,
        "actor_rollout_ref.model.path": config.model.model_path,
        "actor_rollout_ref.model.trust_remote_code": config.model.trust_remote_code,
        "actor_rollout_ref.actor.optim.lr": config.training.learning_rate,
        "actor_rollout_ref.rollout.dtype": config.model.dtype,
        "actor_rollout_ref.rollout.gpu_memory_utilization": config.rllm.gpu_memory_utilization,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": config.rollout.tensor_parallel_size,
        "actor_rollout_ref.rollout.max_model_len": config.rollout.max_model_len,
        "actor_rollout_ref.rollout.prompt_length": config.rllm.max_prompt_length,
        "actor_rollout_ref.rollout.response_length": config.rllm.max_response_length,
        "actor_rollout_ref.rollout.temperature": config.rollout.temperature,
        "actor_rollout_ref.rollout.top_p": config.rollout.top_p,
        "actor_rollout_ref.rollout.do_sample": config.rollout.do_sample,
        "actor_rollout_ref.rollout.n": rollout_n,
        "actor_rollout_ref.rollout.val_kwargs.temperature": config.judge.temperature,
        "actor_rollout_ref.rollout.val_kwargs.top_p": config.judge.top_p,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": config.judge.do_sample,
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "rllm.backend": config.rllm.backend,
        "rllm.algorithm.adv_estimator": config.rllm.algorithm,
        "rllm.rollout.n": rollout_n,
        "rllm.rollout.n_val": 1,
        "rllm.rollout.train": sampling,
        "rllm.rollout.val": {
            "temperature": config.judge.temperature,
            "top_p": config.judge.top_p,
            "top_k": -1,
            "max_tokens": config.judge.max_new_tokens,
        },
        "rllm.workflow.n_parallel_tasks": config.rollout.max_concurrent_episodes,
        "rllm.workflow.retry_limit": 0,
        "rllm.trainer.logger": config.rllm.logger,
        "rllm.trainer.project_name": config.rllm.project_name,
        "rllm.trainer.experiment_name": config.rllm.experiment_name,
        "rllm.trainer.save_freq": config.rllm.save_freq,
        "rllm.trainer.test_freq": config.rllm.eval_freq,
        "rllm.trainer.total_batches": config.rllm.total_training_steps,
    }
    with open_dict(trainer_config):
        for key, value in updates.items():
            if value is not None:
                OmegaConf.update(trainer_config, key, value, merge=False)
    return trainer_config


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
    trainer_config = build_trainer_config(config)

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
