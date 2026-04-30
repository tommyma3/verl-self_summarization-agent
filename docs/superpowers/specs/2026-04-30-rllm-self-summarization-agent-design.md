# rLLM Self-Summarization Agent Design

Date: 2026-04-30

## Purpose

This project builds the rLLM/verl version of the reference self-summarization
agent. The reference project already defines the BrowseComp-Plus runtime loop,
retrieval tools, context summarization behavior, and LLM judge. This design
keeps those agent semantics and replaces the training path with rLLM using its
verl backend.

The active training objective is trajectory-level RL. Every model generation in
the agent episode is a trainable rLLM step, including tool calls, summarization
calls, and the final answer call. The evaluator assigns one terminal answer
reward to the full trajectory. rLLM/verl is responsible for trajectory-level
advantage computation and applying that advantage to each step.

## Reference Behavior To Preserve

The reference agent solves one BrowseComp-Plus query by repeatedly building a
runtime prompt, asking the model for one action, executing environment tools,
and compacting old context when needed.

The preserved runtime rules are:

- The model can use `search`, `get_document`, and `finish`.
- Action outputs must parse as one JSON object after a completed `</think>`.
- JSON inside the thinking section is ignored.
- Only normalized tool-call JSON and tool results enter later action prompts.
- Tool-call thinking is not copied into future context.
- Summarization is runtime-triggered after completed tool rounds.
- Only the post-think summary body is inserted into future context.
- BrowseComp retrieval is handled through the local `bc-plus` backend adapter.
- Final answers are judged by the existing LLM judge.

The revised rLLM design changes the training boundary from the reference
summary/final-answer-only extraction to full-trajectory training.

## rLLM Episode Model

One BrowseComp-Plus query maps to one rLLM episode.

Each episode contains one primary trajectory named `self_summarization_agent`.
Every LLM generation inside the trajectory is a step:

- Action step: action prompt to `search`, `get_document`, or `finish` JSON.
- Summary step: summary prompt to free-form summary text.
- Forced-answer step: final-answer-only prompt after the retrieval budget is
  exhausted.

Example episode shape:

```text
Episode(query_id=q1)
  Trajectory(self_summarization_agent)
    Step 1: action prompt -> search JSON
    Step 2: action prompt -> get_document JSON
    Step 3: summary prompt -> summary text
    Step 4: action prompt -> search JSON
    Step 5: forced answer prompt -> finish JSON
```

All steps are trainable. The design does not mask tool calls or summaries out
of the active rLLM training path.

## Runtime State Modes

The runtime has two generation modes.

### Action Mode

Action mode asks the policy for one JSON tool call. Valid actions are:

- `search`
- `get_document`
- `finish`

The action prompt contains the system prompt, user query, latest summary if any,
unsummarized tool-call history, tool results, remaining retrieval budget, and
next-action instructions. The remaining-budget section is part of the prompt
state for every model-generation round.

Action mode mutates state as follows:

- `search` executes the retrieval backend and appends a tool round.
- `get_document` fetches a document body and appends a tool round.
- `finish` stores the final answer and ends the episode.
- Malformed output ends the episode as an incorrect answer.

### Summary Mode

Summary mode is entered only when the context manager decides the raw context
crosses the configured threshold and there is enough old raw history to retire.

The summary prompt is not a tool prompt. It asks for compact research context as
plain summary text. It still includes the remaining-budget section in its system
state for consistency with every round, but the budget does not change during
summarization. The generated summary is still an rLLM step, but its state
transition is different from a tool action:

- Extract thinking and post-think body if `</think>` is present.
- Store only the post-think summary body as `latest_summary`.
- Retire older raw tool rounds from future action prompts.
- Keep the newest raw round visible so the agent has recent local context.

An empty summary does not terminate the episode. It simply fails to compact and
leaves existing raw context in place.

## Tool Budget Semantics

The runtime tracks remaining retrieval calls. `search` and `get_document` count
against the tool budget. `finish` does not count against it because it
terminates the episode.

Every model-generation prompt includes an explicit budget section:

```text
### TOOL_BUDGET
Remaining search/get_document calls: N
```

In action mode, when the remaining budget is greater than zero, the agent may
call `search`, `get_document`, or `finish`.

When the remaining budget reaches zero, the runtime runs one additional forced
answer step. That prompt must instruct the agent to submit a final answer:

```text
### TOOL_BUDGET
Remaining search/get_document calls: 0

### NEXT_ACTION
You must now submit the final answer.
Return exactly one JSON object:
{"tool_name": "finish", "arguments": {"answer": "..."}}
Do not call search or get_document.
```

If the forced-answer step produces valid `finish` JSON, the judge scores the
answer. If it produces a malformed output or another tool call, the episode is
treated as incorrect and receives reward `-1`.

## Reward Design

The reward is answer-level only:

- Correct final answer: `+1`
- Incorrect final answer: `-1`
- Malformed tool/action output: `-1`
- Malformed forced-answer output: `-1`
- Runtime error that prevents a valid answer: `-1`

The evaluator reuses the reference `RewardJudge` for completed final answers.
Malformed action outputs do not need judge calls because they are directly
incorrect.

The evaluator returns one terminal reward for the episode. rLLM/verl computes
trajectory-level advantages from grouped rollouts and applies the advantage to
all steps in the trajectory.

## Components

### Core Runtime Package

The target repo should include the reference runtime modules with focused
changes for the new rLLM semantics:

```text
src/self_summarization_agent/
  backend.py
  bcplus_backend.py
  config.py
  context.py
  dataset.py
  export.py
  generation.py
  judge.py
  models.py
  prompts.py
  rewards.py
  runtime.py
```

The core changes from the reference runtime are:

- Add remaining budget to action prompts.
- Count `search` and `get_document` calls against retrieval budget.
- Add forced final-answer generation after budget exhaustion.
- Record every generation step in a trajectory-friendly structure.
- Keep summary compaction behavior and sanitized visible context intact.

### rLLM Integration Package

The new rLLM-facing modules are:

```text
src/self_summarization_agent/rllm_agent.py
src/self_summarization_agent/rllm_dataset.py
src/self_summarization_agent/rllm_evaluator.py
src/self_summarization_agent/train_rllm.py
```

`rllm_agent.py` defines the rollout function or AgentFlow wrapper. It runs the
agent loop for one task, calls the model through rLLM's tracked client or
gateway, records every generation as a step, executes tools, triggers summaries,
and returns an episode with artifacts for evaluation and debugging.

`rllm_dataset.py` loads BrowseComp-Plus examples and exposes task dictionaries
containing `query_id`, `query`, and `answer`.

`rllm_evaluator.py` scores each episode using the final answer in the episode
artifacts. It emits reward `+1` or `-1` plus metrics such as answer correctness,
status, tool-call count, and summary count.

`train_rllm.py` loads config, builds datasets, wires the AgentFlow and
Evaluator, constructs rLLM `AgentTrainer` with `backend="verl"`, and starts
training.

### Configs

The repo should include:

```text
configs/run/default.yaml
configs/train/rllm_verl.yaml
```

The training config should separate:

- experiment paths and seed
- BrowseComp dataset paths
- retrieval backend settings
- model/generation settings
- runtime context and tool-budget settings
- judge settings
- rLLM/verl training settings

The old reference custom trainer config is not the active path.

## Data Flow

The end-to-end training data flow is:

```text
BrowseComp examples
  -> rLLM dataset task rows
  -> rLLM AgentTrainer
  -> SelfSummarization AgentFlow
  -> tracked model calls for every action/summary/forced answer step
  -> BrowseComp backend tool execution
  -> Episode artifacts with final answer and trace metadata
  -> SelfSummarization Evaluator
  -> terminal reward
  -> rLLM/verl trajectory advantage and policy update
```

The runtime trace should remain human-readable. Episode artifacts should include
at least:

- `query_id`
- `query`
- `status`
- `final_answer`
- `retrieved_docids`
- `tool_call_counts`
- `summary_turn_count`
- `turn_records` or equivalent step metadata
- judge outcome and parse status when applicable

## Error Handling

Malformed action output terminates the episode with reward `-1`.

Unknown tool names, missing arguments, non-string search queries, non-string
document IDs, and invalid `finish` answers are malformed action outputs.

If the retrieval backend raises for a missing document or other tool failure,
the episode should terminate with status reflecting the tool error and reward
`-1`.

If the context manager detects that a packed prompt exceeds the safe context
limit, the episode should terminate with reward `-1` and a clear status rather
than crashing the trainer process.

Judge parse failures on completed answers are treated as incorrect answers and
receive reward `-1`.

## Testing Plan

The implementation should preserve or add tests for:

- Tool-call parsing only after completed `</think>`.
- JSON inside thinking is ignored.
- Normalized tool calls, not raw thinking, enter future prompts.
- Summary prompts are separate from action prompts.
- Post-think summary body enters context; summary thinking does not.
- Empty summary does not retire raw context.
- Remaining tool budget appears in every action prompt.
- `search` and `get_document` decrement the remaining budget.
- `finish` does not decrement the retrieval budget.
- Budget exhaustion triggers one forced final-answer step.
- Malformed forced-answer output receives reward `-1`.
- Every action and summary generation is represented as a trainable step.
- Evaluator assigns `+1` only for judge-correct final answers and `-1`
  otherwise.
- rLLM modules can be imported without requiring GPU execution in unit tests,
  using fake clients and fake episodes.

Local verification may be limited to syntax and unit tests in this Windows
workspace. Full rLLM/verl training should be validated in the GPU environment.

## Non-Goals

This design does not keep the reference custom FSDP2/context-parallel trainer as
the active training backend.

This design does not preserve summary/final-answer-only training in the active
rLLM path.

This design does not redesign BrowseComp retrieval or the LLM judge.

This design does not require local end-to-end verl execution on the Windows
workspace.

## Implementation Order

1. Port the reference runtime, data, retrieval, judge, and export modules.
2. Update prompt/runtime behavior for remaining tool budget and forced final
   answer.
3. Add trajectory-friendly step recording for every model generation.
4. Add rLLM dataset, AgentFlow, evaluator, and training launcher modules.
5. Add configs and README commands.
6. Port and update tests around runtime, budget, rewards, and rLLM wrappers.
7. Run local syntax/unit validation where possible.
