# rLLM Self-Summarization Agent

This project is an rLLM/verl self-summarization agent for BrowseComp-Plus. The
runtime records each trainable agent step, including tool calls, summaries, and
final answers, so rLLM can assign trajectory-level advantages back to the full
episode.

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
