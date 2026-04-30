# rLLM Self-Summarization Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the rLLM/verl training version of the BrowseComp-Plus self-summarization agent from the approved design.

**Architecture:** Port the reference runtime and BrowseComp support code, then adapt the runtime to full-trajectory rLLM training. rLLM owns rollout tracing, trajectory-level advantages, and verl policy updates; this repo owns prompts, tool execution, forced-answer behavior, final-answer judging, and human-readable trace artifacts.

**Tech Stack:** Python 3.11, pytest, PyYAML, transformers/vLLM-compatible generation helpers, optional rLLM imports, OpenAI-compatible model gateway, BrowseComp-Plus retrieval assets, verl through rLLM.

---

## File Structure

Create or modify these files:

- Modify: `pyproject.toml` - package metadata, dependencies, pytest path.
- Modify: `README.md` - runnable smoke, eval, and rLLM training commands.
- Modify: `main.py` - smoke CLI entrypoint.
- Create: `src/self_summarization_agent/__init__.py` - package marker.
- Create: `src/self_summarization_agent/backend.py` - backend protocol and fake backend.
- Create: `src/self_summarization_agent/bcplus_backend.py` - BrowseComp search/get_document adapter.
- Create: `src/self_summarization_agent/config.py` - run/train config dataclasses and YAML loader.
- Create: `src/self_summarization_agent/context.py` - context counting, summary-context packing, and budget block injection.
- Create: `src/self_summarization_agent/dataset.py` - BrowseComp query loading and slicing.
- Create: `src/self_summarization_agent/export.py` - BrowseComp-style run record export.
- Create: `src/self_summarization_agent/generation.py` - local transformers/vLLM generator helpers for eval and judge.
- Create: `src/self_summarization_agent/judge.py` - existing LLM judge and parser.
- Create: `src/self_summarization_agent/models.py` - runtime dataclasses, including generation-step traces.
- Create: `src/self_summarization_agent/prompts.py` - action, forced-answer, summary, and budget prompt builders.
- Create: `src/self_summarization_agent/rewards.py` - answer-level reward constants/helpers.
- Create: `src/self_summarization_agent/runtime.py` - agent state machine, full-step tracing, forced answer.
- Create: `src/self_summarization_agent/launcher_utils.py` - shared file, JSONL, runtime builder utilities.
- Create: `src/self_summarization_agent/run_launcher.py` - non-training benchmark/eval launcher.
- Create: `src/self_summarization_agent/rllm_dataset.py` - rLLM task loader.
- Create: `src/self_summarization_agent/rllm_evaluator.py` - rLLM evaluator wrapper.
- Create: `src/self_summarization_agent/rllm_agent.py` - rLLM rollout/AgentFlow wrapper.
- Create: `src/self_summarization_agent/train_rllm.py` - rLLM AgentTrainer launcher.
- Create: `configs/run/default.yaml` - local eval config.
- Create: `configs/train/rllm_verl.yaml` - rLLM/verl training config.
- Create: `tests/test_runtime.py` - parser, prompt, summary, budget, and forced-answer tests.
- Create: `tests/test_rllm_dataset.py` - rLLM task dataset tests.
- Create: `tests/test_rllm_evaluator.py` - terminal reward/evaluator tests.
- Create: `tests/test_rllm_agent.py` - fake-client rollout wrapper tests.
- Create: `tests/test_cli.py` - smoke CLI test.

Do not copy the reference custom FSDP2 trainer or `train_step.py` into the active path.

### Task 1: Package Scaffold And Smoke CLI

**Files:**
- Modify: `pyproject.toml`
- Modify: `main.py`
- Create: `src/self_summarization_agent/__init__.py`
- Create: `src/self_summarization_agent/backend.py`
- Create: `src/self_summarization_agent/models.py`
- Create: `src/self_summarization_agent/prompts.py`
- Create: `src/self_summarization_agent/runtime.py`
- Create: `src/self_summarization_agent/export.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write the failing CLI smoke test**

Create `tests/test_cli.py`:

```python
import contextlib
import io
import json

import main


def test_main_prints_smoke_run_record() -> None:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        main.main()

    record = json.loads(stdout.getvalue())

    assert record["query_id"] == "smoke-q1"
    assert record["status"] == "completed"
    assert record["result"] == [{"type": "output_text", "output": "smoke answer"}]
    assert record["tool_call_counts"] == {"search": 1, "get_document": 0}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_cli.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'self_summarization_agent'` or JSON mismatch from the existing hello-world CLI.

- [ ] **Step 3: Update packaging**

Replace `pyproject.toml` with:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "verl-self-summarization-agent"
version = "0.1.0"
description = "rLLM/verl self-summarization agent for BrowseComp-Plus"
readme = "README.md"
requires-python = ">=3.11,<3.12"
dependencies = [
  "PyYAML>=6.0.2",
  "datasets>=4.0.0",
  "transformers>=4.57.0",
  "torch>=2.7.0",
  "openai>=1.0.0",
]

[dependency-groups]
dev = [
  "pytest>=8.4.0",
]

[project.optional-dependencies]
retrieval = [
  "faiss-cpu>=1.13.2",
  "pyserini>=2.0.0",
  "tevatron",
]
serving = [
  "vllm>=0.17.0",
]
rllm = [
  "rllm",
]

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = [".", "src"]
```

- [ ] **Step 4: Add minimal core files**

Create `src/self_summarization_agent/__init__.py`:

```python
"""Self-summarization agent runtime and rLLM training integration."""
```

Create `src/self_summarization_agent/backend.py`:

```python
from dataclasses import dataclass
from typing import Any, Protocol


SearchResult = dict[str, Any]


class BrowseCompBackend(Protocol):
    def search(self, query: str) -> list[SearchResult]:
        ...

    def get_document(self, doc_id: str) -> str:
        ...


@dataclass(slots=True)
class FakeBackend:
    search_index: dict[str, list[str]]
    documents: dict[str, str]

    def search(self, query: str) -> list[SearchResult]:
        return [
            {"docid": doc_id, "snippet": self.documents.get(doc_id, "")}
            for doc_id in self.search_index.get(query, [])
        ]

    def get_document(self, doc_id: str) -> str:
        return self.documents[doc_id]
```

Create `src/self_summarization_agent/models.py`:

```python
from dataclasses import dataclass, field
from typing import Literal


Role = Literal["system", "user", "assistant", "tool"]
StepKind = Literal["action", "summary", "forced_answer"]


@dataclass(slots=True)
class Message:
    role: Role
    content: str


@dataclass(slots=True)
class ToolCallRecord:
    tool_name: str
    arguments: dict[str, str]
    raw_output: str
    is_valid: bool = True


@dataclass(slots=True)
class ToolRound:
    assistant_message: Message
    tool_call: ToolCallRecord
    tool_result: Message


@dataclass(slots=True)
class GenerationStep:
    step_id: str
    kind: StepKind
    prompt: str
    completion: str
    parsed_tool_name: str | None = None
    is_trainable: bool = True


@dataclass(slots=True)
class EpisodeState:
    query_id: str
    user_prompt: str
    context_threshold_tokens: int
    latest_summary: str | None = None
    summary_count: int = 0
    summarized_round_count: int = 0
    rounds: list[ToolRound] = field(default_factory=list)


@dataclass(slots=True)
class RuntimeResult:
    query_id: str
    status: str
    final_answer: str | None
    summary_turns: list[str]
    retrieved_docids: list[str]
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    generation_steps: list[GenerationStep] = field(default_factory=list)
    turn_records: list[dict[str, str]] = field(default_factory=list)
```

Create `src/self_summarization_agent/prompts.py`:

```python
def build_budget_block(remaining_tool_calls: int) -> str:
    return f"### TOOL_BUDGET\nRemaining search/get_document calls: {remaining_tool_calls}"


def build_system_prompt() -> str:
    return """You are a deep research AI agent.

Your response must be exactly one JSON object for one tool call.
After any internal reasoning, the final visible action must be only one JSON tool call.

Available tools:
- search: find candidate documents for a search query. Use {"tool_name": "search", "arguments": {"query": "..."}}
- get_document: read one retrieved document by id. Use {"tool_name": "get_document", "arguments": {"doc_id": "..."}}
- finish: submit the final answer. Use {"tool_name": "finish", "arguments": {"answer": "..."}}

Tool strategy:
- Start with search unless the answer is already fully supported by the conversation.
- Use get_document only with docid values returned by search.
- Call finish only when the evidence is sufficient."""


def build_summary_system_prompt() -> str:
    return """You are a context summarization AI agent.

Your task is to summarize the previous research context so another step of the same agent can continue the task.
Return only the summary text after thinking."""


def build_summary_prompt() -> str:
    return (
        "Write a clean summary containing only the essential information needed "
        "to continue solving the task. Preserve normalized facts, unresolved "
        "questions, evidence-grounded facts tied to doc_id citations, and useful next steps."
    )


def build_normal_next_action_prompt() -> str:
    return (
        "### NEXT_ACTION\n"
        "Return exactly one JSON object for the next tool call. "
        "After any thinking, the final visible action must be only the JSON object. "
        "Return one action only."
    )


def build_forced_answer_prompt() -> str:
    return (
        "### NEXT_ACTION\n"
        "You must now submit the final answer. "
        "Return exactly one JSON object: "
        '{"tool_name": "finish", "arguments": {"answer": "..."}}. '
        "Do not call search or get_document."
    )
```

Create `src/self_summarization_agent/export.py`:

```python
from self_summarization_agent.models import RuntimeResult


def build_run_record(result: RuntimeResult) -> dict[str, object]:
    return {
        "query_id": result.query_id,
        "status": result.status,
        "retrieved_docids": result.retrieved_docids,
        "result": [{"type": "output_text", "output": result.final_answer or ""}],
        "tool_call_counts": result.tool_call_counts,
    }
```

Create a temporary minimal `src/self_summarization_agent/runtime.py` sufficient for the smoke test:

```python
from dataclasses import dataclass

from self_summarization_agent.backend import BrowseCompBackend
from self_summarization_agent.models import RuntimeResult


@dataclass(slots=True)
class ScriptedModel:
    outputs: list[str]
    cursor: int = 0

    def generate(self, prompt: str) -> str:
        del prompt
        output = self.outputs[self.cursor]
        self.cursor += 1
        return output


def build_smoke_result() -> RuntimeResult:
    return RuntimeResult(
        query_id="smoke-q1",
        status="completed",
        final_answer="smoke answer",
        summary_turns=[],
        retrieved_docids=["smoke-doc"],
        tool_call_counts={"search": 1, "get_document": 0},
    )
```

Replace `main.py` with:

```python
import json

from self_summarization_agent.export import build_run_record
from self_summarization_agent.runtime import build_smoke_result


def main() -> None:
    print(json.dumps(build_run_record(build_smoke_result()), ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the smoke test**

Run: `python -m pytest tests/test_cli.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add pyproject.toml main.py src/self_summarization_agent tests/test_cli.py
git commit -m "feat: scaffold self-summarization package"
```

### Task 2: Port Core Reference Modules

**Files:**
- Create: `src/self_summarization_agent/config.py`
- Create: `src/self_summarization_agent/context.py`
- Create: `src/self_summarization_agent/dataset.py`
- Create: `src/self_summarization_agent/generation.py`
- Create: `src/self_summarization_agent/judge.py`
- Create: `src/self_summarization_agent/rewards.py`
- Create: `src/self_summarization_agent/launcher_utils.py`
- Create: `src/self_summarization_agent/bcplus_backend.py`
- Modify: `src/self_summarization_agent/models.py`
- Test: `tests/test_context.py`
- Test: `tests/test_rewards.py`

- [ ] **Step 1: Copy stable reference modules**

Copy these files from `reference/src/self_summarization_agent/` into `src/self_summarization_agent/` with the noted edits:

```text
config.py          copy, then add RLLMConfig dataclass later in Task 6
dataset.py         copy unchanged
generation.py      copy unchanged
judge.py           copy unchanged
launcher_utils.py  copy, then adjust RuntimeResult serialization for generation_steps
bcplus_backend.py  copy unchanged
```

Do not copy `trainer.py`, `train_step.py`, `iteration_launcher.py`, `openrlhf_agent.py`, `openrlhf_dataset.py`, or `openrlhf_judge_server.py`.

- [ ] **Step 2: Replace rewards with answer-level helper**

Create `src/self_summarization_agent/rewards.py`:

```python
from typing import Literal


Outcome = Literal["correct_answer", "wrong_answer"]


def answer_reward(outcome: Outcome) -> float:
    if outcome == "correct_answer":
        return 1.0
    if outcome == "wrong_answer":
        return -1.0
    raise ValueError(f"Unknown answer outcome: {outcome}")


def incorrect_reward() -> float:
    return -1.0
```

- [ ] **Step 3: Update context to include budget block in summary prompts**

Create `src/self_summarization_agent/context.py` based on the reference file, with this public method signature:

```python
def build_summary_context(self, state: EpisodeState, *, remaining_tool_calls: int) -> str:
    pieces = [
        build_summary_system_prompt(),
        build_budget_block(remaining_tool_calls),
        state.user_prompt,
    ]
    if state.latest_summary:
        pieces.append(state.latest_summary)
    for round_record in state.rounds:
        pieces.extend(
            [
                round_record.assistant_message.content,
                self._serialize_tool_call(round_record.tool_call),
                round_record.tool_result.content,
            ]
        )
    pieces.append(build_summary_prompt())
    return "\n".join(pieces)
```

Keep `current_token_count`, `should_summarize`, `_serialize_tool_call`, and `assert_fits` from the reference implementation.

- [ ] **Step 4: Write context and rewards tests**

Create `tests/test_context.py`:

```python
from self_summarization_agent.context import ContextManager
from self_summarization_agent.models import EpisodeState, Message, ToolCallRecord, ToolRound


def test_summary_context_includes_remaining_budget() -> None:
    manager = ContextManager(token_counter=lambda text: len(text.split()), max_context_tokens=1000)
    state = EpisodeState(query_id="q1", user_prompt="question", context_threshold_tokens=10)
    state.rounds.append(
        ToolRound(
            assistant_message=Message(role="assistant", content='{"tool_name": "search", "arguments": {"query": "q"}}'),
            tool_call=ToolCallRecord(tool_name="search", arguments={"query": "q"}, raw_output="{}"),
            tool_result=Message(role="tool", content='[{"docid": "d1"}]'),
        )
    )

    prompt = manager.build_summary_context(state, remaining_tool_calls=3)

    assert "### TOOL_BUDGET" in prompt
    assert "Remaining search/get_document calls: 3" in prompt
    assert "context summarization AI agent" in prompt
```

Create `tests/test_rewards.py`:

```python
from self_summarization_agent.rewards import answer_reward, incorrect_reward


def test_answer_reward_is_terminal_only() -> None:
    assert answer_reward("correct_answer") == 1.0
    assert answer_reward("wrong_answer") == -1.0
    assert incorrect_reward() == -1.0
```

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest tests/test_context.py tests/test_rewards.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/self_summarization_agent tests/test_context.py tests/test_rewards.py
git commit -m "feat: port core runtime support modules"
```

### Task 3: Implement Runtime With Budget, Forced Answer, And Step Tracing

**Files:**
- Modify: `src/self_summarization_agent/models.py`
- Modify: `src/self_summarization_agent/runtime.py`
- Modify: `src/self_summarization_agent/launcher_utils.py`
- Test: `tests/test_runtime.py`

- [ ] **Step 1: Write parser and runtime tests**

Create `tests/test_runtime.py`:

```python
from self_summarization_agent.backend import FakeBackend
from self_summarization_agent.runtime import EpisodeRuntime, ScriptedModel, extract_summary_output, parse_model_tool_call


def tool_output(json_text: str, thinking: str = "thinking") -> str:
    return f"<think>{thinking}</think>\n{json_text}"


def test_parse_model_tool_call_requires_completed_thinking() -> None:
    assert parse_model_tool_call('{"tool_name": "search", "arguments": {"query": "q"}}') is None


def test_parse_model_tool_call_ignores_json_inside_thinking() -> None:
    raw = '<think>{"tool_name": "finish", "arguments": {"answer": "bad"}}</think>\n{"tool_name": "search", "arguments": {"query": "good"}}'
    parsed = parse_model_tool_call(raw)
    assert parsed is not None
    payload, _ = parsed
    assert payload == {"tool_name": "search", "arguments": {"query": "good"}}


def test_action_prompt_includes_remaining_budget() -> None:
    model = ScriptedModel([tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}')])
    runtime = EpisodeRuntime(
        model=model,
        backend=FakeBackend(search_index={}, documents={}),
        context_threshold_tokens=1000,
        max_context_tokens=4096,
        max_tool_calls=5,
    )

    result = runtime.run("q1", "question")

    assert result.status == "completed"
    assert "Remaining search/get_document calls: 5" in result.generation_steps[0].prompt
    assert result.generation_steps[0].kind == "action"


def test_search_decrements_budget_and_finish_does_not() -> None:
    model = ScriptedModel(
        [
            tool_output('{"tool_name": "search", "arguments": {"query": "q"}}'),
            tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}'),
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=FakeBackend(search_index={"q": ["d1"]}, documents={"d1": "fact"}),
        context_threshold_tokens=1000,
        max_context_tokens=4096,
        max_tool_calls=2,
    )

    result = runtime.run("q1", "question")

    assert result.status == "completed"
    assert result.tool_call_counts == {"search": 1, "get_document": 0}
    assert "Remaining search/get_document calls: 2" in result.generation_steps[0].prompt
    assert "Remaining search/get_document calls: 1" in result.generation_steps[1].prompt


def test_budget_exhaustion_adds_forced_answer_step() -> None:
    model = ScriptedModel(
        [
            tool_output('{"tool_name": "search", "arguments": {"query": "q"}}'),
            tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}'),
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=FakeBackend(search_index={"q": ["d1"]}, documents={"d1": "fact"}),
        context_threshold_tokens=1000,
        max_context_tokens=4096,
        max_tool_calls=1,
    )

    result = runtime.run("q1", "question")

    assert result.status == "completed"
    assert [step.kind for step in result.generation_steps] == ["action", "forced_answer"]
    assert "You must now submit the final answer" in result.generation_steps[-1].prompt


def test_forced_answer_rejects_more_tool_calls() -> None:
    model = ScriptedModel(
        [
            tool_output('{"tool_name": "search", "arguments": {"query": "q"}}'),
            tool_output('{"tool_name": "search", "arguments": {"query": "again"}}'),
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=FakeBackend(search_index={"q": ["d1"], "again": []}, documents={"d1": "fact"}),
        context_threshold_tokens=1000,
        max_context_tokens=4096,
        max_tool_calls=1,
    )

    result = runtime.run("q1", "question")

    assert result.status == "malformed_forced_answer"
    assert result.final_answer is None
    assert result.generation_steps[-1].kind == "forced_answer"


def test_summary_is_trainable_step_but_uses_summary_state() -> None:
    model = ScriptedModel(
        [
            tool_output('{"tool_name": "search", "arguments": {"query": "first"}}'),
            tool_output('{"tool_name": "search", "arguments": {"query": "second"}}'),
            "<think>summary reasoning</think>\nsummary body",
            tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}'),
        ]
    )
    runtime = EpisodeRuntime(
        model=model,
        backend=FakeBackend(search_index={"first": ["old"], "second": ["new"]}, documents={}),
        context_threshold_tokens=1,
        max_context_tokens=4096,
        max_tool_calls=5,
        token_counter=lambda text: text.count("new"),
    )

    result = runtime.run("q1", "question")

    assert [step.kind for step in result.generation_steps] == ["action", "action", "summary", "action"]
    assert result.generation_steps[2].is_trainable is True
    assert "summary reasoning" not in result.generation_steps[3].prompt
    assert "### SUMMARY\nsummary body" in result.generation_steps[3].prompt
```

- [ ] **Step 2: Run runtime tests to verify failures**

Run: `python -m pytest tests/test_runtime.py -q`

Expected: FAIL because the temporary runtime does not implement parsing, budget, summary, or forced answer behavior.

- [ ] **Step 3: Implement runtime parser and result helpers**

Replace `src/self_summarization_agent/runtime.py` with the reference runtime plus these required changes:

```python
def _remaining_tool_calls(self, state: EpisodeState) -> int:
    if self.max_tool_calls is None:
        return 999999
    used = sum(1 for round_record in state.rounds if round_record.tool_call.tool_name in {"search", "get_document"})
    return max(0, self.max_tool_calls - used)


def _record_generation_step(
    self,
    active: _ActiveEpisode,
    *,
    kind: StepKind,
    prompt: str,
    completion: str,
    parsed_tool_name: str | None = None,
) -> None:
    active.generation_steps.append(
        GenerationStep(
            step_id=f"step-{len(active.generation_steps) + 1}",
            kind=kind,
            prompt=prompt,
            completion=completion,
            parsed_tool_name=parsed_tool_name,
            is_trainable=True,
        )
    )
```

Also update `_ActiveEpisode` to include:

```python
generation_steps: list[GenerationStep] = field(default_factory=list)
```

Update `_build_runtime_prompt` to include:

```python
pieces = [
    self._build_transcript_block("SYSTEM", build_system_prompt()),
    build_budget_block(self._remaining_tool_calls(state)),
    self._build_transcript_block("USER", state.user_prompt),
]
```

Add:

```python
def _build_forced_answer_runtime_prompt(self, state: EpisodeState) -> str:
    pieces = [
        self._build_transcript_block("SYSTEM", build_system_prompt()),
        build_budget_block(0),
        self._build_transcript_block("USER", state.user_prompt),
    ]
    if state.latest_summary:
        pieces.append(self._build_transcript_block("SUMMARY", state.latest_summary))
    for round_record in self._raw_tail_rounds(state):
        pieces.extend(
            [
                self._build_transcript_block("ASSISTANT_TOOL_CALL", round_record.assistant_message.content),
                self._build_transcript_block("TOOL_RESULT", round_record.tool_result.content),
            ]
        )
    pieces.append(build_forced_answer_prompt())
    return "\n".join(pieces)
```

Change `_apply_action_output` signature:

```python
def _apply_action_output(self, active: _ActiveEpisode, raw_output: str, *, prompt: str, step_kind: StepKind) -> None:
```

Inside it, record a generation step for every action/forced-answer output before mutating state. If `step_kind == "forced_answer"` and the parsed tool is not `finish`, set result status `malformed_forced_answer`.

Add completed and malformed results that include `generation_steps=list(active.generation_steps)`.

Update summary prompt building:

```python
prompt = active.context_manager.build_summary_context(
    summary_state,
    remaining_tool_calls=self._remaining_tool_calls(active.state),
)
```

Record summary generations with `kind="summary"` and `parsed_tool_name=None`.

In `run_many`, replace immediate budget failure with forced-answer generation:

```python
if self.max_tool_calls is not None and self._remaining_tool_calls(active.state) <= 0:
    forced_prompt = self._build_forced_answer_runtime_prompt(active.state)
    active.context_manager.assert_fits(forced_prompt)
    forced_items.append((active, forced_prompt))
    continue
```

Generate forced items and apply them with `step_kind="forced_answer"`.

- [ ] **Step 4: Update launcher serialization**

In `src/self_summarization_agent/launcher_utils.py`, ensure `serialize_runtime_result` includes:

```python
"generation_steps": [
    {
        "step_id": step.step_id,
        "kind": step.kind,
        "prompt": step.prompt,
        "completion": step.completion,
        "parsed_tool_name": step.parsed_tool_name,
        "is_trainable": step.is_trainable,
    }
    for step in result.generation_steps
],
```

- [ ] **Step 5: Run runtime tests**

Run: `python -m pytest tests/test_runtime.py -q`

Expected: PASS.

- [ ] **Step 6: Run core tests**

Run: `python -m pytest tests/test_cli.py tests/test_context.py tests/test_rewards.py tests/test_runtime.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/self_summarization_agent/runtime.py src/self_summarization_agent/models.py src/self_summarization_agent/launcher_utils.py tests/test_runtime.py
git commit -m "feat: add full-trajectory runtime steps"
```

### Task 4: Add rLLM Dataset And Evaluator

**Files:**
- Create: `src/self_summarization_agent/rllm_dataset.py`
- Create: `src/self_summarization_agent/rllm_evaluator.py`
- Test: `tests/test_rllm_dataset.py`
- Test: `tests/test_rllm_evaluator.py`

- [ ] **Step 1: Write rLLM dataset tests**

Create `tests/test_rllm_dataset.py`:

```python
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
```

- [ ] **Step 2: Implement rLLM dataset builder**

Create `src/self_summarization_agent/rllm_dataset.py`:

```python
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
        {
            "query_id": example.query_id,
            "query": example.query,
            "answer": example.answer,
        }
        for example in train_examples
    ]
```

- [ ] **Step 3: Write rLLM evaluator tests**

Create `tests/test_rllm_evaluator.py`:

```python
from dataclasses import dataclass

from self_summarization_agent.judge import JudgeDecision
from self_summarization_agent.rllm_evaluator import evaluate_episode_artifacts


@dataclass(slots=True)
class FakeJudge:
    outcome: str

    def evaluate(self, example, status: str, response: str) -> JudgeDecision:
        return JudgeDecision(
            outcome=self.outcome,
            judge_prompt="judge prompt",
            judge_response="correct: yes" if self.outcome == "correct_answer" else "correct: no",
            parse_error=False,
        )


def test_evaluator_returns_positive_reward_for_correct_answer() -> None:
    output = evaluate_episode_artifacts(
        task={"query_id": "q1", "query": "question", "answer": "answer"},
        artifacts={"status": "completed", "final_answer": "answer"},
        judge=FakeJudge("correct_answer"),
    )

    assert output["reward"] == 1.0
    assert output["is_correct"] is True
    assert output["metrics"]["status"] == "completed"


def test_evaluator_returns_negative_reward_for_malformed_action_without_judge() -> None:
    output = evaluate_episode_artifacts(
        task={"query_id": "q1", "query": "question", "answer": "answer"},
        artifacts={"status": "malformed_tool_call", "final_answer": None},
        judge=FakeJudge("correct_answer"),
    )

    assert output["reward"] == -1.0
    assert output["is_correct"] is False
    assert output["metrics"]["judge_called"] is False
```

- [ ] **Step 4: Implement evaluator helper and optional rLLM decorator**

Create `src/self_summarization_agent/rllm_evaluator.py`:

```python
from __future__ import annotations

from typing import Any

from self_summarization_agent.dataset import QueryExample
from self_summarization_agent.judge import RewardJudge
from self_summarization_agent.rewards import answer_reward, incorrect_reward


def evaluate_episode_artifacts(
    *,
    task: dict[str, Any],
    artifacts: dict[str, Any],
    judge: RewardJudge,
) -> dict[str, Any]:
    status = str(artifacts.get("status") or "")
    final_answer = artifacts.get("final_answer")
    if status != "completed" or not isinstance(final_answer, str):
        return {
            "reward": incorrect_reward(),
            "is_correct": False,
            "metrics": {"status": status, "judge_called": False},
        }

    example = QueryExample(
        query_id=str(task["query_id"]),
        query=str(task["query"]),
        answer=str(task["answer"]) if task.get("answer") is not None else None,
    )
    decision = judge.evaluate(example, status, final_answer)
    is_correct = decision.outcome == "correct_answer"
    return {
        "reward": answer_reward("correct_answer" if is_correct else "wrong_answer"),
        "is_correct": is_correct,
        "metrics": {
            "status": status,
            "judge_called": True,
            "judge_parse_error": decision.parse_error,
            "judge_outcome": decision.outcome,
        },
    }


def build_rllm_evaluator(judge: RewardJudge):
    try:
        import rllm
        from rllm.experimental.eval.types import EvalOutput, Signal
    except ImportError as exc:
        raise ImportError("rLLM is required to build the decorated evaluator") from exc

    @rllm.evaluator
    def score(task: dict[str, Any], episode: Any) -> Any:
        artifacts = dict(getattr(episode, "artifacts", {}) or {})
        result = evaluate_episode_artifacts(task=task, artifacts=artifacts, judge=judge)
        return EvalOutput(
            reward=result["reward"],
            is_correct=result["is_correct"],
            signals=[
                Signal(name="accuracy", value=1.0 if result["is_correct"] else 0.0),
                Signal(name="reward", value=result["reward"]),
            ],
        )

    return score
```

- [ ] **Step 5: Run rLLM dataset/evaluator tests**

Run: `python -m pytest tests/test_rllm_dataset.py tests/test_rllm_evaluator.py -q`

Expected: PASS without installing rLLM.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/self_summarization_agent/rllm_dataset.py src/self_summarization_agent/rllm_evaluator.py tests/test_rllm_dataset.py tests/test_rllm_evaluator.py
git commit -m "feat: add rllm dataset and evaluator"
```

### Task 5: Add rLLM AgentFlow Wrapper

**Files:**
- Create: `src/self_summarization_agent/rllm_agent.py`
- Test: `tests/test_rllm_agent.py`

- [ ] **Step 1: Write fake-client AgentFlow tests**

Create `tests/test_rllm_agent.py`:

```python
from self_summarization_agent.backend import FakeBackend
from self_summarization_agent.config import RuntimeConfig
from self_summarization_agent.rllm_agent import OpenAICompatibleGenerator, run_self_summarization_episode


class FakeChoiceMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeChoiceMessage(content)


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [FakeChoice(content)]


class FakeChatCompletions:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.prompts: list[str] = []

    def create(self, *, model, messages, temperature, top_p, max_tokens):
        del model, temperature, top_p, max_tokens
        self.prompts.append(messages[-1]["content"])
        return FakeResponse(self.outputs.pop(0))


class FakeChat:
    def __init__(self, outputs: list[str]) -> None:
        self.completions = FakeChatCompletions(outputs)


class FakeClient:
    def __init__(self, outputs: list[str]) -> None:
        self.chat = FakeChat(outputs)


def tool_output(json_text: str) -> str:
    return f"<think>reason</think>\n{json_text}"


def test_run_self_summarization_episode_records_all_generation_steps() -> None:
    client = FakeClient(
        [
            tool_output('{"tool_name": "search", "arguments": {"query": "q"}}'),
            tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}'),
        ]
    )
    generator = OpenAICompatibleGenerator(
        client=client,
        model="policy",
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.95,
    )

    result = run_self_summarization_episode(
        task={"query_id": "q1", "query": "question", "answer": "done"},
        generator=generator,
        backend=FakeBackend(search_index={"q": ["d1"]}, documents={"d1": "fact"}),
        runtime_config=RuntimeConfig(context_threshold_tokens=1000, max_context_tokens=4096, tool_budget=3),
    )

    assert result["artifacts"]["status"] == "completed"
    assert result["artifacts"]["final_answer"] == "done"
    assert [step["kind"] for step in result["artifacts"]["generation_steps"]] == ["action", "action"]
    assert all(step["is_trainable"] for step in result["artifacts"]["generation_steps"])
```

- [ ] **Step 2: Implement OpenAI-compatible generator and episode helper**

Create `src/self_summarization_agent/rllm_agent.py`:

```python
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


def _step_to_dict(step: Any) -> dict[str, Any]:
    return {
        "step_id": step.step_id,
        "kind": step.kind,
        "prompt": step.prompt,
        "completion": step.completion,
        "parsed_tool_name": step.parsed_tool_name,
        "is_trainable": step.is_trainable,
    }


def run_self_summarization_episode(
    *,
    task: dict[str, Any],
    generator: Any,
    backend: Any,
    runtime_config: Any,
) -> dict[str, Any]:
    runtime = build_runtime(generator, backend, runtime_config)
    result = runtime.run(query_id=str(task["query_id"]), user_prompt=str(task["query"]))
    artifacts = {
        "query_id": result.query_id,
        "query": str(task["query"]),
        "answer": task.get("answer"),
        "status": result.status,
        "final_answer": result.final_answer,
        "retrieved_docids": result.retrieved_docids,
        "tool_call_counts": result.tool_call_counts,
        "summary_turn_count": len(result.summary_turns),
        "generation_steps": [_step_to_dict(step) for step in result.generation_steps],
    }
    return {"artifacts": artifacts}


def build_rllm_rollout(*, config: Any, backend: Any):
    try:
        import rllm
        from openai import OpenAI
        from rllm.types import Episode, Trajectory
    except ImportError as exc:
        raise ImportError("rLLM and openai are required to build the decorated rollout") from exc

    @rllm.rollout
    def solve(task: Any, agent_config: Any) -> Any:
        task_data = getattr(task, "data", task)
        client = OpenAI(base_url=agent_config.base_url, api_key="EMPTY")
        generator = OpenAICompatibleGenerator(
            client=client,
            model=agent_config.model,
            max_new_tokens=config.model.max_new_tokens,
            temperature=config.model.temperature,
            top_p=config.model.top_p,
        )
        payload = run_self_summarization_episode(
            task=task_data,
            generator=generator,
            backend=backend,
            runtime_config=config.runtime,
        )
        episode = Episode(
            trajectories=[Trajectory(name="self_summarization_agent", steps=[])],
            artifacts=payload["artifacts"],
        )
        episode.id = str(task_data.get("query_id", ""))
        episode.task = task_data
        return episode

    return solve
```

- [ ] **Step 3: Run rLLM agent tests**

Run: `python -m pytest tests/test_rllm_agent.py -q`

Expected: PASS without installing rLLM.

- [ ] **Step 4: Commit**

Run:

```bash
git add src/self_summarization_agent/rllm_agent.py tests/test_rllm_agent.py
git commit -m "feat: add rllm rollout wrapper"
```

### Task 6: Add rLLM Training Config And Launcher

**Files:**
- Modify: `src/self_summarization_agent/config.py`
- Create: `src/self_summarization_agent/train_rllm.py`
- Create: `configs/train/rllm_verl.yaml`
- Create: `configs/run/default.yaml`
- Test: `tests/test_train_rllm_config.py`

- [ ] **Step 1: Write config tests**

Create `tests/test_train_rllm_config.py`:

```python
from pathlib import Path

from self_summarization_agent.config import load_train_config


def test_load_rllm_train_config() -> None:
    config = load_train_config(Path("configs/train/rllm_verl.yaml"))

    assert config.training.backend == "rllm_verl"
    assert config.rllm.backend == "verl"
    assert config.runtime.tool_budget > 0
```

- [ ] **Step 2: Extend config dataclasses**

In `src/self_summarization_agent/config.py`, add:

```python
@dataclass(slots=True)
class RLLMConfig:
    backend: str = "verl"
    algorithm: str = "grpo"
    train_batch_size: int = 8
    rollout_group_size: int = 4
    max_prompt_length: int = 65536
    max_response_length: int = 8192
    total_training_steps: int = 100
    save_freq: int = 10
    eval_freq: int = 0
    project_name: str = "self-summarization-agent"
    experiment_name: str = "rllm-verl"
```

Add field to `TrainConfig`:

```python
rllm: RLLMConfig = field(default_factory=RLLMConfig)
```

Update `load_train_config`:

```python
rllm=RLLMConfig(**_require_section(raw, "rllm")),
```

Set `TrainingConfig.backend` default to `"rllm_verl"` or ensure the YAML sets it explicitly.

- [ ] **Step 3: Add configs**

Create `configs/train/rllm_verl.yaml`:

```yaml
experiment:
  name: qwen-bcplus-rllm
  seed: 7
  output_root: artifacts
  bc_plus_root: bc-plus

dataset:
  decrypted_path:
  queries_tsv_path:
  offset: 0
  limit:
  shuffle: false
  train_limit: 200
  eval_limit: 10

retrieval:
  backend: faiss
  top_k: 5
  snippet_max_tokens: 512
  document_max_tokens: 8192
  snippet_tokenizer_path:
  index_path: bc-plus/indexes/qwen3-embedding-8b/corpus.shard*.pkl
  model_name: Qwen/Qwen3-Embedding-8B
  normalize: true
  pooling: eos
  torch_dtype: bfloat16
  dataset_name: Tevatron/browsecomp-plus-corpus
  max_length: 8192

model:
  backend: rllm_gateway
  model_path: Qwen/Qwen3.5-9B
  dtype: bfloat16
  device_map: auto
  max_new_tokens: 8192
  temperature: 0.7
  top_p: 0.95
  do_sample: true
  trust_remote_code: false
  enable_thinking: true

rollout:
  backend: rllm_gateway
  gpu_ids: []
  tensor_parallel_size: 1
  attention_backend:
  max_model_len: 65536
  max_concurrent_episodes: 32
  max_new_tokens: 8192
  temperature: 0.7
  top_p: 0.95
  do_sample: true

runtime:
  context_threshold_tokens: 24000
  max_context_tokens: 65536
  tool_budget: 20

judge:
  enabled: true
  max_new_tokens: 4096
  temperature: 0.0
  top_p: 1.0
  do_sample: false

training:
  backend: rllm_verl
  gpu_ids: []
  group_size: 4
  learning_rate: 1.0e-6
  steps: 100

rllm:
  backend: verl
  algorithm: grpo
  train_batch_size: 8
  rollout_group_size: 4
  max_prompt_length: 65536
  max_response_length: 8192
  total_training_steps: 100
  save_freq: 10
  eval_freq: 0
  project_name: self-summarization-agent
  experiment_name: qwen-bcplus-rllm
```

Create `configs/run/default.yaml` by copying the same experiment, dataset, retrieval, model, rollout, and runtime sections, without `judge`, `training`, or `rllm`.

- [ ] **Step 4: Implement `train_rllm.py`**

Create `src/self_summarization_agent/train_rllm.py`:

```python
from __future__ import annotations

import argparse
from typing import Any

from self_summarization_agent.bcplus_backend import build_backend
from self_summarization_agent.config import load_train_config, parse_cli_overrides
from self_summarization_agent.generation import build_generator
from self_summarization_agent.judge import RewardJudge
from self_summarization_agent.rllm_agent import build_rllm_rollout
from self_summarization_agent.rllm_dataset import build_rllm_tasks
from self_summarization_agent.rllm_evaluator import build_rllm_evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the self-summarization agent with rLLM/verl.")
    parser.add_argument("--config", required=True, help="Path to configs/train/rllm_verl.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def build_trainer(config: Any) -> Any:
    try:
        from rllm.experimental.unified_trainer import AgentTrainer
    except ImportError as exc:
        raise ImportError("rLLM is required for train_rllm.py. Install the rllm extra in the training environment.") from exc

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
```

- [ ] **Step 5: Run config test**

Run: `python -m pytest tests/test_train_rllm_config.py -q`

Expected: PASS.

- [ ] **Step 6: Run import check**

Run: `python -m py_compile src/self_summarization_agent/train_rllm.py src/self_summarization_agent/rllm_agent.py src/self_summarization_agent/rllm_evaluator.py`

Expected: exits with code 0. It should not require rLLM because imports are inside builder functions.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/self_summarization_agent/config.py src/self_summarization_agent/train_rllm.py configs tests/test_train_rllm_config.py
git commit -m "feat: add rllm verl training launcher"
```

### Task 7: Restore Eval Launcher And README

**Files:**
- Create: `src/self_summarization_agent/run_launcher.py`
- Modify: `README.md`
- Test: `tests/test_run_launcher.py`

- [ ] **Step 1: Write run launcher smoke test**

Create `tests/test_run_launcher.py`:

```python
from pathlib import Path

from self_summarization_agent.backend import FakeBackend
from self_summarization_agent.config import DatasetConfig, ExperimentConfig, ModelConfig, RetrievalConfig, RolloutConfig, RunConfig, RuntimeConfig
from self_summarization_agent.run_launcher import run_experiment
from self_summarization_agent.runtime import ScriptedModel


def tool_output(json_text: str) -> str:
    return f"<think>reason</think>\n{json_text}"


def test_run_experiment_writes_trajectory_jsonl(tmp_path: Path) -> None:
    config = RunConfig(
        experiment=ExperimentConfig(name="demo", seed=1, output_root=str(tmp_path), bc_plus_root=str(tmp_path)),
        dataset=DatasetConfig(limit=1),
        retrieval=RetrievalConfig(backend="faiss", index_path="unused"),
        model=ModelConfig(model_path="unused"),
        runtime=RuntimeConfig(context_threshold_tokens=1000, max_context_tokens=4096, tool_budget=2),
        rollout=RolloutConfig(),
    )
    run_dir = run_experiment(
        config,
        examples=[],
        backend=FakeBackend(search_index={}, documents={}),
        generator=ScriptedModel([tool_output('{"tool_name": "finish", "arguments": {"answer": "done"}}')]),
        explicit_examples=[("q1", "question")],
    )

    assert (run_dir / "trajectories.jsonl").exists()
```

- [ ] **Step 2: Implement run launcher**

Create `src/self_summarization_agent/run_launcher.py` from the reference implementation, with this small test seam:

```python
def run_experiment(
    config,
    *,
    examples: list[QueryExample] | None = None,
    backend: Any | None = None,
    generator: Any | None = None,
    explicit_examples: list[tuple[str, str]] | None = None,
) -> Path:
    seed_everything(config.experiment.seed)
    if explicit_examples is not None:
        loaded_examples = [QueryExample(query_id=query_id, query=query) for query_id, query in explicit_examples]
    else:
        loaded_examples = examples or load_query_examples(
            config.experiment.bc_plus_root,
            config.dataset,
            require_answers=False,
            seed=config.experiment.seed,
        )
    backend = backend or build_backend(config.experiment.bc_plus_root, config.retrieval)
    generator = generator or build_generator(config.model)
    runtime = build_runtime(generator, backend, config.runtime)
```

Keep the reference artifact writing behavior and include `generation_steps` through `serialize_runtime_result`.

- [ ] **Step 3: Update README**

Replace `README.md` with concise run instructions:

```markdown
# rLLM Self-Summarization Agent

This repo implements the rLLM/verl training version of the BrowseComp-Plus self-summarization agent.

The agent uses `search`, `get_document`, and `finish` tool calls. Runtime-triggered summarization is treated as a normal trainable rLLM step. The evaluator assigns one terminal answer reward: `+1` for a judge-correct final answer and `-1` otherwise.

## Smoke

```powershell
python main.py
```

## Evaluation Run

```powershell
python -m self_summarization_agent.run_launcher --config configs/run/default.yaml
```

## rLLM/verl Training

```powershell
python -m self_summarization_agent.train_rllm --config configs/train/rllm_verl.yaml
```

The training environment needs rLLM, verl, BrowseComp-Plus assets, retrieval indexes, and the configured model gateway.
```

- [ ] **Step 4: Run run launcher test**

Run: `python -m pytest tests/test_run_launcher.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/self_summarization_agent/run_launcher.py README.md tests/test_run_launcher.py
git commit -m "feat: add evaluation launcher and docs"
```

### Task 8: Final Validation And Cleanup

**Files:**
- Modify: any touched files with lint, import, or typing errors found by validation.

- [ ] **Step 1: Run the full local test suite**

Run: `python -m pytest -q`

Expected: all local tests pass. If rLLM is not installed, tests must still pass because rLLM imports are optional and delayed.

- [ ] **Step 2: Run syntax check**

Run:

```bash
python -m py_compile main.py src/self_summarization_agent/*.py
```

Expected: exits with code 0.

- [ ] **Step 3: Inspect git diff**

Run: `git status --short`

Expected: only intended implementation files are modified or untracked. `bc-plus/` remains untouched unless it was already tracked in this repo.

- [ ] **Step 4: Commit validation fixes if needed**

If Step 1 or Step 2 required fixes, commit them:

```bash
git add src tests README.md configs pyproject.toml main.py
git commit -m "test: validate rllm self-summarization agent"
```

If no fixes were needed, do not create an empty commit.

## Self-Review Notes

Spec coverage:

- Full-trajectory training is covered by Tasks 3 and 5.
- Summary-as-step behavior is covered by Task 3.
- Answer-only terminal reward is covered by Task 4.
- Malformed action and forced-answer failure are covered by Task 3 and Task 4.
- Remaining budget in every generation prompt is covered by Tasks 2 and 3.
- Forced answer after budget exhaustion is covered by Task 3.
- rLLM/verl launcher is covered by Task 6.
- README and runnable commands are covered by Task 7.

Known validation limit:

- Local tests intentionally avoid requiring rLLM, verl, vLLM, FAISS, or GPU execution. Full distributed training must be validated in the remote GPU environment after local unit tests pass.

