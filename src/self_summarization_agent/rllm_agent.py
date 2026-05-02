from __future__ import annotations

from dataclasses import dataclass
import inspect
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


def _load_rllm_types(rllm: Any) -> tuple[type[Any], type[Any], type[Any]]:
    try:
        from rllm.agents.agent import Episode, Step, Trajectory
    except ImportError:
        try:
            from rllm.types import Episode, Step, Trajectory
        except ImportError:
            Episode = getattr(rllm, "Episode", None)
            Trajectory = getattr(rllm, "Trajectory", None)
            Step = getattr(rllm, "Step", None)
            if Episode is None or Trajectory is None or Step is None:
                raise
    return Episode, Trajectory, Step


def _build_step_action(step: dict[str, Any]) -> Any:
    parsed_tool_name = step.get("parsed_tool_name")
    if parsed_tool_name is None:
        return None
    return {
        "tool_name": parsed_tool_name,
        "kind": step.get("kind"),
    }


def _generation_steps_to_rllm_steps(Step: type[Any], generation_steps: list[dict[str, Any]]) -> list[Any]:
    rllm_steps: list[Any] = []
    for index, step in enumerate(generation_steps):
        prompt = str(step.get("prompt") or "")
        completion = str(step.get("completion") or "")
        rllm_steps.append(
            Step(
                chat_completions=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
                observation=prompt,
                action=_build_step_action(step),
                model_response=completion,
                info={
                    "step_id": step.get("step_id"),
                    "kind": step.get("kind"),
                    "parsed_tool_name": step.get("parsed_tool_name"),
                    "is_trainable": bool(step.get("is_trainable")),
                },
                done=index == len(generation_steps) - 1,
            )
        )
    return rllm_steps


def _construct_supported(cls: type[Any], **kwargs: Any) -> Any:
    parameters = inspect.signature(cls).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return cls(**kwargs)
    return cls(**{key: value for key, value in kwargs.items() if key in parameters})


def _build_episode(
    Episode: type[Any],
    Trajectory: type[Any],
    *,
    task_data: dict[str, Any],
    trajectory_steps: list[Any],
    artifacts: dict[str, Any],
) -> Any:
    trajectory = _construct_supported(
        Trajectory,
        name="self_summarization_agent",
        task=task_data,
        steps=trajectory_steps,
        reward=None,
        info={"status": artifacts["status"]},
    )
    episode = _construct_supported(
        Episode,
        task=task_data,
        trajectories=[trajectory],
        metrics={},
        info={"artifacts": artifacts},
        artifacts=artifacts,
    )
    if not hasattr(episode, "artifacts"):
        try:
            setattr(episode, "artifacts", artifacts)
        except (AttributeError, TypeError):
            pass
    return episode


def build_rllm_rollout(config: Any, backend: Any) -> Any:
    try:
        import rllm
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "rLLM and openai are required to build the rollout. Install the rllm extra in the training environment."
        ) from exc

    try:
        Episode, Trajectory, Step = _load_rllm_types(rllm)
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
        generation_steps = payload["artifacts"]["generation_steps"]
        trajectory_steps = _generation_steps_to_rllm_steps(Step, generation_steps)
        return _build_episode(
            Episode,
            Trajectory,
            task_data=task_data,
            trajectory_steps=trajectory_steps,
            artifacts=payload["artifacts"],
        )

    return solve
