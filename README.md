# LayerSkip Evaluation Framework

A **lm-evaluation-harness** style framework for comparing the performance of
different layer-skipping strategies across standard NLP benchmarks.

## Supported Layer-Skipping Strategies

| Strategy | Description | Key Paper |
|----------|-------------|-----------|
| **none** | Full model (baseline, no skipping) | – |
| **layerskip** | Static early exit at a fixed fraction of layers | [Elhoushi et al., 2024](https://arxiv.org/abs/2404.16710) |
| **caml** | Confidence-Adaptive Multi-Layer: exits when model confidence exceeds a threshold | [Schuster et al., 2022](https://arxiv.org/abs/2207.07061) |
| **gateskip** | Gate-based skipping: skips layers where the relative hidden-state change is below a threshold | [Laitenberger et al., 2024](https://arxiv.org/abs/2510.13876) |

## Supported Benchmarks

| Task | Type | Metric | Default shots |
|------|------|--------|---------------|
| **MMLU** | Multiple-choice QA | Accuracy | 5-shot |
| **HellaSwag** | Commonsense reasoning | Accuracy | 0-shot |
| **WinoGrande** | Pronoun resolution | Accuracy | 0-shot |
| **GSM8K** | Math word problems | Exact match | 8-shot |
| **HumanEval** | Python code generation | pass@1 | 0-shot |

## Supported Backbone Models

- `meta-llama/Meta-Llama-3-8B-Instruct`
- `meta-llama/Llama-3.2-1B-Instruct`

Any HuggingFace causal language model can also be used.

---

## Installation

```bash
git clone https://github.com/Artessay/LayerSkip
cd LayerSkip
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Evaluate with no layer skipping (baseline)

```bash
python eval.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --strategy none \
  --tasks mmlu hellaswag \
  --max_samples 200
```

### Evaluate with LayerSkip (75% of layers)

```bash
python eval.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --strategy layerskip \
  --layerskip_exit_ratio 0.75 \
  --tasks mmlu hellaswag winogrande gsm8k humaneval
```

### Compare all three strategies simultaneously

```bash
python eval.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --strategy none layerskip caml gateskip \
  --tasks mmlu hellaswag winogrande \
  --batch_size 4 \
  --output comparison_results.json
```

### CAML with custom confidence threshold

```bash
python eval.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --strategy caml \
  --caml_confidence_threshold 0.85 \
  --tasks mmlu
```

### GateSkip with custom budget

```bash
python eval.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --strategy gateskip \
  --gateskip_skip_budget 0.3 \
  --gateskip_gate_threshold 0.01 \
  --tasks mmlu hellaswag
```

---

## Command-Line Reference

```
python eval.py --help
```

### Model arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | *required* | HuggingFace model ID or local path |
| `--dtype` | `auto` | `auto`, `float16`, `bfloat16`, `float32` |
| `--device` | `auto` | `cuda`, `cuda:0`, `cpu`, etc. |
| `--batch_size` | `1` | Batch size for loglikelihood evaluation |
| `--max_length` | `2048` | Maximum token sequence length |
| `--trust_remote_code` | `False` | Allow remote code execution |

### Strategy arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--strategy` | `none` | One or more of `none layerskip caml gateskip` |
| `--layerskip_exit_ratio` | `0.75` | Fraction of layers to execute (LayerSkip) |
| `--layerskip_min_layers` | `4` | Minimum layers always executed (LayerSkip) |
| `--caml_confidence_threshold` | `0.9` | Exit threshold (CAML) |
| `--caml_min_layers` | `4` | Minimum layers before checking (CAML) |
| `--caml_check_every` | `1` | Check confidence every N layers (CAML) |
| `--gateskip_gate_threshold` | `0.01` | Relative-change threshold (GateSkip) |
| `--gateskip_skip_budget` | `0.3` | Max fraction of layers to skip (GateSkip) |
| `--gateskip_min_layers` | `4` | Minimum layers before skipping (GateSkip) |

### Task arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--tasks` | `mmlu` | One or more of `mmlu hellaswag winogrande gsm8k humaneval` |
| `--max_samples` | all | Per-task example cap |
| `--num_fewshot` | task default | Override few-shot count for all tasks |
| `--seed` | `42` | Random seed |

### Output arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output` | none | Save results JSON to this path |
| `--verbosity` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Programmatic API

```python
from evaluation.evaluator import Evaluator

# Single strategy
evaluator = Evaluator(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    strategy_name="layerskip",
    strategy_kwargs={"exit_ratio": 0.75},
    tasks=["mmlu", "hellaswag"],
    task_kwargs={"mmlu": {"max_samples": 100}},
    batch_size=4,
)
results = evaluator.run()
Evaluator.print_results(results)

# Compare multiple strategies
all_results = []
for strategy in ["none", "layerskip", "caml", "gateskip"]:
    ev = Evaluator(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        strategy_name=strategy,
        tasks=["mmlu"],
    )
    all_results.append(ev.run())

comparison = Evaluator.compare_results(all_results)
Evaluator.print_comparison(comparison)
```

### Strategy API

```python
from evaluation.strategies import get_strategy

# LayerSkip: use first 75% of layers
strategy = get_strategy("layerskip", exit_ratio=0.75)

# CAML: exit when confidence > 90%
strategy = get_strategy("caml", confidence_threshold=0.9)

# GateSkip: skip up to 30% of low-change layers
strategy = get_strategy("gateskip", skip_budget=0.3)
```

### Task API

```python
from evaluation.tasks import get_task

mmlu = get_task("mmlu", num_fewshot=5, max_samples=100)
hellaswag = get_task("hellaswag", num_fewshot=0)
gsm8k = get_task("gsm8k", num_fewshot=8)
```

---

## Project Structure

```
LayerSkip/
├── eval.py                    # CLI entry point
├── requirements.txt
├── setup.py
├── evaluation/
│   ├── evaluator.py           # Evaluation orchestrator
│   ├── models/
│   │   ├── base_model.py      # Abstract LM interface
│   │   └── hf_model.py        # HuggingFace model wrapper
│   ├── strategies/
│   │   ├── base_strategy.py   # Abstract strategy base class
│   │   ├── layerskip.py       # Static early-exit strategy
│   │   ├── caml.py            # Confidence-adaptive strategy
│   │   └── gateskip.py        # Gate/change-based strategy
│   ├── tasks/
│   │   ├── base_task.py       # Abstract task base class
│   │   ├── mmlu.py            # MMLU (57 subjects)
│   │   ├── hellaswag.py       # HellaSwag
│   │   ├── winogrande.py      # WinoGrande
│   │   ├── gsm8k.py           # GSM8K math
│   │   └── humaneval.py       # HumanEval code generation
│   └── utils/
│       └── metrics.py         # Shared metric helpers
└── tests/
    ├── test_strategies.py
    ├── test_tasks.py
    └── test_evaluator.py
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## How the Strategies Work

### LayerSkip

Executes only the first `exit_ratio × N` transformer layers, then applies the
model's layer norm and LM head to those intermediate representations. This is
the fastest strategy to reason about: it always uses the same set of layers
regardless of the input.

### CAML (Confidence-Adaptive Multi-Layer)

At each candidate exit layer (starting from `min_layers`), the strategy
computes the mean maximum softmax probability over the batch and sequence. The
first layer that exceeds `confidence_threshold` is used as the exit point.
"Easy" inputs (high-confidence after few layers) exit early; "hard" inputs
use more layers.

### GateSkip

Computes the *relative change* in hidden-state norms between consecutive
layers: `||h_l − h_{l−1}|| / ||h_{l−1}||`. Layers where this change is below
`gate_threshold` are considered low-importance and counted as skipped, up to
`skip_budget` fraction of the total. The strategy returns the last
high-importance layer as the exit point.

---

## Citation

If you use this evaluation framework, please cite the relevant papers:

```bibtex
@article{elhoushi2024layerskip,
  title   = {LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding},
  author  = {Elhoushi, Mostafa and Shrivastava, Akshat and Liskovich, Diana and
             Hosmer, Basil and Wasti, Bram and Lai, Liangzhen and Mahmoud, Anas
             and Acun, Bilge and Agarwal, Saurabh and Roman, Ahmed and others},
  journal = {arXiv preprint arXiv:2404.16710},
  year    = {2024}
}

@article{schuster2022confident,
  title   = {Confident Adaptive Language Modeling},
  author  = {Schuster, Tal and Fisch, Adam and Gupta, Jai and Dehghani, Mostafa
             and Bahri, Dara and Tran, Vinh and Tay, Yi and Metzler, Donald},
  journal = {arXiv preprint arXiv:2207.07061},
  year    = {2022}
}

@article{laitenberger2024gateskip,
  title   = {What Layers When: Learning to Skip Compute in LLMs with Residual Gates},
  author  = {Laitenberger, Felix and others},
  journal = {arXiv preprint arXiv:2510.13876},
  year    = {2024}
}
```