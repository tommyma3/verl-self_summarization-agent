from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from self_summarization_agent.launcher_utils import build_runtime


@dataclass(slots=True)
class OpenAICompatibleGenerator:
    client: Any
    model: str
    max_new_tokens: int
    temperature: float
    top_p: float

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )
        return response.choices[0].message.content or ""

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def _generation_step_to_dict(step: Any) -> dict[str, Any]:
    return {
        "step_id": step.step_id,
        "kind": step.kind,
        "prompt": step.prompt,
        "completion": step.completion,
        "parsed_tool_name": step.parsed_tool_name,
        "is_trainable": step.is_trainable,
    }


def run_self_summarization_episode(
    task: dict[str, Any],
    generator: Any,
    backend: Any,
    runtime_config: Any,
) -> dict[str, Any]:
    runtime = build_runtime(generator, backend, runtime_config)
    query_id = str(task["query_id"])
    query = str(task["query"])
    result = runtime.run(query_id, query)

    return {
        "artifacts": {
            "query_id": result.query_id,
            "query": query,
            "answer": task.get("answer"),
            "status": result.status,
            "final_answer": result.final_answer,
            "retrieved_docids": list(result.retrieved_docids),
            "tool_call_counts": dict(result.tool_call_counts),
            "summary_turn_count": len(result.summary_turns),
            "generation_steps": [_generation_step_to_dict(step) for step in result.generation_steps],
        }
    }


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _load_rllm_types(rllm: Any) -> tuple[type[Any], type[Any]]:
    try:
        from rllm.types import Episode, Trajectory
    except ImportError:
        Episode = getattr(rllm, "Episode", None)
        Trajectory = getattr(rllm, "Trajectory", None)
        if Episode is None or Trajectory is None:
            raise
    return Episode, Trajectory


def build_rllm_rollout(config: Any, backend: Any) -> Any:
    try:
        import rllm
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "rLLM and openai are required to build the rollout. Install the rllm extra in the training environment."
        ) from exc

    try:
        Episode, Trajectory = _load_rllm_types(rllm)
    except ImportError as exc:
        raise ImportError(
            "rLLM Episode and Trajectory types are required to build the rollout. "
            "Install a compatible rLLM version in the training environment."
        ) from exc

    @rllm.rollout
    def solve(task: Any, agent_config: Any) -> Any:
        task_data = _get_attr(task, "data", task)
        client = OpenAI(base_url=_get_attr(agent_config, "base_url"), api_key=_get_attr(agent_config, "api_key", "EMPTY"))
        generator = OpenAICompatibleGenerator(
            client=client,
            model=_get_attr(agent_config, "model", config.model.model_path),
            max_new_tokens=_get_attr(agent_config, "max_new_tokens", config.model.max_new_tokens),
            temperature=_get_attr(agent_config, "temperature", config.model.temperature),
            top_p=_get_attr(agent_config, "top_p", config.model.top_p),
        )
        payload = run_self_summarization_episode(
            task=task_data,
            generator=generator,
            backend=backend,
            runtime_config=config.runtime,
        )
        return Episode(
            trajectories=[Trajectory(name="self_summarization_agent", steps=[])],
            artifacts=payload["artifacts"],
        )

    return solve
