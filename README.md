# rLLM Self-Summarization Agent

This project is an rLLM/verl self-summarization agent for BrowseComp-Plus. The
runtime records each trainable agent step, including tool calls, summaries, and
final answers, so rLLM can assign trajectory-level advantages back to the full
episode.

## Installing packages

Use the existing rLLM virtual environment at `./rllm/.venv`. The environment
already contains rLLM with verl support, so install only the packages needed by
the self-summarization training code from the project root:

```bash
uv pip install --python rllm/.venv/bin/python 
  "faiss-cpu>=1.13.2" \
  "pyserini>=1.2.0" \
  "tevatron @ git+https://github.com/texttron/tevatron.git@main" \
  "qwen-omni-utils>=0.0.8" \
  "numpy>=1.26,<2.0"
```

Make the local `src/self_summarization_agent` package importable from that
virtual environment:

```bash
printf '%s\n' "$PWD/src" > rllm/.venv/lib/python3.11/site-packages/verl_self_summarization_agent_src.pth
```

Verify the environment:

```bash
rllm/.venv/bin/python -c "import rllm, verl, torch, transformers, ray, vllm; print('core imports ok')"
rllm/.venv/bin/python -c "import faiss, pyserini, tevatron, qwen_omni_utils; print('retrieval imports ok')"
rllm/.venv/bin/python -m self_summarization_agent.train_rllm --help
```

Smoke check:

```bash
python main.py
```

Run evaluation rollouts:

```bash
python -m self_summarization_agent.run_launcher --config configs/run/default.yaml
```

Train with rLLM/verl:

```bash
python -m self_summarization_agent.train_rllm --config configs/train/rllm_verl.yaml
```

The `retrieval`, `serving`, and `rllm` extras are only needed in real retrieval,
model-serving, or rLLM training environments.
