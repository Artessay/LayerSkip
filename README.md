# LayerSkip

An evaluation framework for comparing the performance of
different layer-skipping strategies across standard NLP benchmarks.

## Supported Layer-Skipping Strategies

| Strategy | Description | Key Paper |
|----------|-------------|-----------|
| **none** | Full model (baseline, no skipping) | – |
| **layerskip** | Static suffix pruning (generic early-exit baseline) | Inspired by [Elhoushi et al., 2024](https://arxiv.org/abs/2404.16710) |
| **caml** | Token-wise CALM early exit (historical misspelled CLI name) | [Schuster et al., 2022](https://arxiv.org/abs/2207.07061) |
| **gateskip** | Learned sigmoid residual gates with quantile token budgets | [Laitenberger et al., 2025](https://arxiv.org/abs/2510.13876) |
| **calibratedskip** | Compute and save calibration-set layer importance metrics | – |
| **manualskip** | Bypass user-selected transformer layers | – |
| **shortgpt** | One-shot global block removal using Block Influence | [Men et al., 2024](https://arxiv.org/abs/2403.03853) |
| **sleb** | Greedy global block elimination using calibration token NLL | [Song et al., 2024](https://arxiv.org/abs/2402.09025) |
| **tale** | Task-aware greedy layer elimination on a labeled search split | [Naim et al., 2026](https://aclanthology.org/2026.findings-acl.1136/) |

> [!IMPORTANT]
> CALM requires intermediate-head training; ordinary checkpoints require the
> explicit `--caml_allow_untrained_exits` ablation flag. GateSkip requires a
> jointly fine-tuned gate state via `--gateskip_gate_state_path`.
> The generic `layerskip` option does not reproduce the paper's specialized
> training recipe. ShortGPT and SLEB search once per model and reuse the same
> selected layers for every downstream task; TALE searches separately for each
> task. TALE is disabled for HumanEval because it has no labeled search split
> disjoint from test. See the
> [baseline audit and roadmap](docs/BASELINE_AUDIT.md) before interpreting or
> extending results.

## Supported Benchmarks

| Task | Type | Metric | Default shots |
|------|------|--------|---------------|
| **MMLU** | Multiple-choice QA | Accuracy | 5-shot |
| **HellaSwag** | Commonsense reasoning | Accuracy | 0-shot |
| **WinoGrande** | Pronoun resolution | Accuracy | 0-shot |
| **GSM8K** | Math word problems | Exact match | 8-shot |
| **HumanEval** | Python code generation | pass@1 | 0-shot |

Datasets can be downloaded ahead of time with ModelScope. Set `HF_HOME` and
pass `--local` to read datasets from `$HF_HOME/datasets/source`; local models
continue to use `/data/<model_id>`.

```bash
modelscope download --dataset cais/mmlu --local_dir "$HF_HOME/datasets/source/cais/mmlu"
modelscope download --dataset evalscope/hellaswag --local_dir "$HF_HOME/datasets/source/Rowan/hellaswag"
modelscope download --dataset allenai/winogrande --local_dir "$HF_HOME/datasets/source/allenai/winogrande"
modelscope download --dataset openai-mirror/gsm8k --local_dir "$HF_HOME/datasets/source/openai/gsm8k"
modelscope download --dataset openai-mirror/openai_humaneval --local_dir "$HF_HOME/datasets/source/openai/openai_humaneval"
```

ShortGPT additionally requires PG19 (`emozilla/pg19` by default), and SLEB
requires WikiText (`wikitext`, configuration `wikitext-2-raw-v1`). With
`--local`, these calibration corpora resolve to
`$HF_HOME/datasets/source/emozilla/pg19` and
`$HF_HOME/datasets/source/Salesforce/wikitext`. This keeps Hugging Face source
data in one location; set `HF_HOME` before running, or pass
`--shortgpt_dataset` / `--sleb_dataset` with another local path.

```bash
hf download emozilla/pg19 --repo-type dataset --local-dir "$HF_HOME/datasets/source/emozilla/pg19"
hf download Salesforce/wikitext --repo-type dataset --local-dir "$HF_HOME/datasets/source/Salesforce/wikitext"
```

## Supported Backbone Models

- `meta-llama/Meta-Llama-3-8B-Instruct`
- `meta-llama/Llama-3.2-1B-Instruct`

Any HuggingFace causal language model can also be used.

```bash
modelscope download --model LLM-Research/Meta-Llama-3-8B-Instruct --local_dir /data/meta-llama/Meta-Llama-3-8B-Instruct
modelscope download --model LLM-Research/Llama-3.2-1B-Instruct --local_dir /data/meta-llama/Llama-3.2-1B-Instruct
```

---

## Installation

```bash
git clone https://github.com/Artessay/LayerSkip
cd LayerSkip

conda create -n layer python=3.12 -y
conda activate layer
```

```bash
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
  --tasks mmlu hellaswag winogrande gsm8k humaneval \
  --local \
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

### Compare all strategies simultaneously

```bash
python eval.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --strategy none layerskip caml gateskip manualskip \
  --manualskip_layers 2 4 8 \
  --tasks mmlu hellaswag winogrande \
  --batch_size 4 \
  --output results
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
  --gateskip_gate_state_path checkpoints/gates.pt \
  --tasks mmlu hellaswag
```

### ManualSkip with explicit layers

```bash
python eval.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --strategy manualskip \
  --manualskip_layers 2 4 8 \
  --tasks mmlu hellaswag
```

### Calibrate layer importance metrics

Use `calibratedskip` when you only want to score layer importance. This mode
writes calibration metrics and exits; it does not run task evaluation and does
not choose any layers to skip automatically.

```bash
python eval.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --strategy calibratedskip \
  --calibratedskip_metrics activation_ratio gradient_value gradient_trace shapley_value \
  --calibration_max_samples 4096 \
  --local \
  --tasks mmlu hellaswag winogrande gsm8k
```

```bash
python eval.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --strategy calibratedskip \
  --calibratedskip_metrics activation_ratio gradient_value gradient_trace shapley_value \
  --calibration_max_samples 4096 \
  --local \
  --tasks mmlu hellaswag winogrande gsm8k
```

For each task, CalibratedSkip saves a JSON file under
`results/<model>/<task>/calibration/` containing every layer's metrics. Inspect
those files to choose layers, then run `manualskip` with the selected layer
numbers.

### Run ShortGPT

ShortGPT scores Block Influence once on PG19, removes the globally lowest
scoring blocks, and uses that same compact architecture for every requested
task. `--shortgpt_num_remove` takes precedence over the ratio.

```bash
python eval.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --strategy shortgpt \
  --shortgpt_num_remove 4 \
  --shortgpt_dataset emozilla/pg19 \
  --shortgpt_split validation \
  --tasks mmlu hellaswag winogrande gsm8k
```

### Run SLEB

SLEB greedily tries every currently eligible block and permanently selects the
candidate with the lowest WikiText next-token NLL at each round. The paper
searches all blocks (`0/0` barriers); the flags allow reproducing the official
code's optional first/last-layer protection.

```bash
python eval.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --strategy sleb \
  --sleb_num_remove 4 \
  --sleb_dataset wikitext \
  --sleb_dataset_name wikitext-2-raw-v1 \
  --sleb_early_barrier 0 \
  --sleb_latter_barrier 0 \
  --tasks mmlu hellaswag winogrande gsm8k
```

### Run TALE

TALE performs a separate greedy search for each task. At every round it removes
the single layer giving the highest search-split accuracy. The default final
variant is the deepest accepted configuration no more than `0.08` below the
dense baseline (eight percentage points).

```bash
python eval.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --strategy tale \
  --tale_threshold 0.08 \
  --tale_variant threshold_final \
  --tale_search_max_samples 128 \
  --tasks mmlu hellaswag winogrande gsm8k
```

The command above explicitly caps each search at 128 examples to control
cost. Omit `--tale_search_max_samples` to use the full labeled search split,
which matches the authors' single-GPU default.

Search traces are written atomically below
`results/<model>/pruning/<method>/` and reused only when the complete search
configuration, model/runtime identity, task scope, and search inputs match.
For TALE this includes the constructed prompts (and few-shot examples); for
ShortGPT and SLEB it includes a SHA-256 digest of the exact calibration token
tensors. ShortGPT computes this digest during its first streaming scoring pass
and reconstructs it on CPU before reusing a completed trace. Local model paths
also include a content digest of recognized model, tokenizer, and custom-code
artifacts; cache directories, VCS metadata, optimizer state, and unrelated
files are excluded.

Final task evaluation physically shortens the transformer `ModuleList`; trace
layer IDs are original-model 0-based IDs, while result metadata also includes
a 1-based rendering for humans. The reversible evaluator retains removed
modules for restoration, so use exported compact checkpoints—not this context
manager—to measure parameter memory or checkpoint size.

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
| `--trust_remote_code` | `False` | Allow remote code execution |

### Strategy arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--strategy` | `none` | One or more of `none layerskip caml gateskip calibratedskip manualskip shortgpt sleb tale` |
| `--layerskip_exit_ratio` | `0.75` | Fraction of layers to execute (LayerSkip) |
| `--layerskip_min_layers` | `4` | Minimum layers always executed (LayerSkip) |
| `--caml_confidence_threshold` | `0.9` | Exit threshold (CAML) |
| `--caml_min_layers` | `4` | Minimum layers before checking (CAML) |
| `--caml_check_every` | `1` | Check confidence every N layers (CAML) |
| `--gateskip_skip_budget` | `0.3` | Target fraction of tokens skipped per gated module (GateSkip) |
| `--gateskip_gate_state_path` | required | Fine-tuned GateSkip vector-gate state dict |
| `--gateskip_min_layers` | `1` | Number of initial transformer layers left ungated (GateSkip evaluator) |
| `--calibratedskip_metrics` | `activation_ratio gradient_trace` | Calibration-only metrics to compute and save for every layer: `activation_ratio`, `gradient_value`, `gradient_trace`, `shapley_value` |
| `--calibration_max_samples` | all | Cap calibration examples per task |
| `--manualskip_layers` | required for `manualskip` | 1-based layer numbers to bypass, e.g. `2 4 8` or `2,4,8` |
| `--shortgpt_prune_ratio` | `0.25` | Fraction of blocks to remove when an exact count is not supplied |
| `--shortgpt_num_remove` | unset | Exact global removal count; overrides the ratio |
| `--shortgpt_dataset` | `emozilla/pg19` | PG19 identifier or local path |
| `--shortgpt_split` | `validation` | PG19 calibration split |
| `--shortgpt_max_samples` | all | Cap PG19 source documents |
| `--shortgpt_sequence_length` | `256` | Non-overlapping token chunk length |
| `--shortgpt_search_batch_size` | `1` | Block-influence scoring batch size |
| `--sleb_prune_ratio` | `0.2` | Fraction of blocks to remove when an exact count is not supplied |
| `--sleb_num_remove` | unset | Exact global removal count; overrides the ratio |
| `--sleb_dataset` | `wikitext` | Calibration identifier or local path |
| `--sleb_dataset_name` | `wikitext-2-raw-v1` | Dataset configuration |
| `--sleb_split` | `train` | Calibration corpus split |
| `--sleb_max_samples` | `128` | Number of source documents sampled |
| `--sleb_sequence_length` | `2048` | Candidate-scoring token sequence length |
| `--sleb_search_batch_size` | `1` | Candidate-scoring batch size |
| `--sleb_seed` | `0` | Reference WikiText document-shuffle seed |
| `--sleb_early_barrier` | `0` | Number of initial blocks protected from removal |
| `--sleb_latter_barrier` | `0` | Number of final blocks protected from removal |
| `--tale_threshold` | `0.08` | Absolute accuracy tolerance below the dense baseline |
| `--tale_search_max_samples` | all | Optional cap on labeled search examples per task |
| `--tale_max_remove` | all but one | Maximum greedy search depth |
| `--tale_target_remove` | unset | Exact `budget` depth; also caps search when no maximum is supplied |
| `--tale_variant` | `threshold_final` | Evaluate `threshold_final`, `best`, `bsba`, or `budget` |
| `--tale_continue_below_threshold` | `False` | Continue the trace after first crossing the fixed threshold |

### Task arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--tasks` | `mmlu` | One or more of `mmlu hellaswag winogrande gsm8k humaneval` |
| `--max_samples` | all | Per-task example cap |
| `--num_fewshot` | task default | Override few-shot count for all tasks |
| `--seed` | `42` | Random seed |
| `--local` | `False` | Use `/data/<model_id>` for models and `$HF_HOME/datasets/source/<dataset_id>` for all datasets |

### Output arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output` | `results` | Directory for per-task JSON files. Each model/task/strategy/config setting is saved separately by default |
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

ShortGPT, SLEB, and TALE use the same `Evaluator` entry point but are search
pipelines rather than entries in the layer-skipping strategy registry:

```python
shortgpt = Evaluator(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    strategy_name="shortgpt",
    strategy_kwargs={"num_remove": 4, "seed": 42},
    tasks=["mmlu", "hellaswag"],
)
results = shortgpt.run()
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

# CalibratedSkip: metadata holder for saved calibration scores
strategy = get_strategy("calibratedskip")

# ManualSkip: bypass layers 2, 4, and 8
strategy = get_strategy("manualskip", skip_layers=[2, 4, 8])
```

### Task API

```python
from evaluation.tasks import get_task

mmlu = get_task("mmlu", num_fewshot=5, max_samples=100)
hellaswag = get_task("hellaswag", num_fewshot=0)
winogrande = get_task("winogrande", num_fewshot=5)
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
│   ├── calibration.py          # Layer-importance calibration and metric saving
│   ├── models/
│   │   ├── base_model.py      # Abstract LM interface
│   │   └── hf_model.py        # HuggingFace model wrapper
│   ├── pruning/
│   │   ├── runner.py          # Search orchestration and resumable traces
│   │   ├── shortgpt.py        # Block Influence scoring and selection
│   │   ├── sleb.py            # Iterative NLL-based elimination
│   │   ├── tale.py            # Task-aware greedy elimination
│   │   ├── corpora.py         # PG19/WikiText calibration batches
│   │   ├── structural.py      # Reversible physical ModuleList pruning
│   │   └── io.py              # Versioned atomic search-trace persistence
│   ├── strategies/
│   │   ├── base_strategy.py   # Abstract strategy base class
│   │   ├── layerskip.py       # Static early-exit strategy
│   │   ├── caml.py            # Confidence-adaptive strategy
│   │   ├── gateskip.py        # Checkpoint-backed vector-gate evaluator
│   │   ├── calibratedskip.py  # Calibration metadata strategy
│   │   └── manualskip.py      # User-selected layer bypass strategy
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
model's layer norm and LM head to that representation. The implementation
bypasses all suffix blocks after the exit, so it reduces executed block FLOPs.
It does not include the LayerSkip paper's training or self-speculative decoder.

### `caml` (CALM)

At each candidate exit layer (starting from `min_layers`), the strategy
uses a per-token top-two softmax probability gap (or hidden-state saturation).
The first layer exceeding the shared threshold supplies that token's state.
Generation optionally uses CALM Eq. (5)'s decaying threshold.

### GateSkip

Loads jointly trained vector gates `sigmoid(W_l h_l + b_l)`, averages each
gate over its hidden dimension, and uses a per-layer linearly interpolated
quantile threshold. Low-ranked tokens copy their residual state; retained
tokens receive the gated module output, matching the scoring and masking
equations in Algorithms 2–3. This repository currently reconstructs those
outputs after a normal full forward pass. It is therefore a checkpoint-quality
evaluator, not the paper's attention/MLP wrappers or fused kernel, and must not
be used to report latency or realized FLOP savings.

### ShortGPT

For transformer block `i`, ShortGPT computes Block Influence

```text
BI_i = 1 - mean_valid_tokens cosine(block_input_i, block_output_i).
```

The implementation captures each block's true input and pre-final-norm output
with hooks, masks padding tokens, scores all blocks in one PG19 pass, and
removes the globally lowest `BI` values. This is a model-global, one-shot
ranking: downstream task labels never affect selection. The optional recovery
tuning studied by the paper is not implemented, so report this baseline as
one-shot ShortGPT.

### SLEB

SLEB maintains a set of already removed blocks. At every round it evaluates
all remaining eligible candidates after adding that candidate to the set,
computes mean next-token NLL on the same fixed WikiText batches, and commits the
candidate with the lowest NLL. It then recomputes all candidate scores for the
new shortened model. Search therefore costs approximately
`N + (N - 1) + ...` calibration evaluations for successive removals, making it
substantially more expensive than ShortGPT. Both barrier defaults are `0` to
match the paper; setting them to `1/1` matches the protection used by the
released reference code.

For `N` layers, `R` removals, and protected early/late counts `be`/`bl`, the
search performs `R(N-be-bl) - R(R-1)/2` full calibration evaluations. For
example, 32 layers with 7 removals and `1/1` barriers requires 189 evaluations;
budget this search separately from the final benchmark run.

### TALE

TALE is task-specific. It first measures dense accuracy `A0` on a labeled
search split. At every round it evaluates deletion of each remaining layer and
commits the candidate with the highest search accuracy. The acceptance floor
is always `A0 - 0.08` by default; it is not recomputed from the previous round.
The trace records four selectable views:

- `best`: highest-accuracy configuration on the accepted greedy trajectory;
- `bsba`: deepest configuration whose score is at least the dense baseline;
- `threshold_final`: deepest configuration at or above `A0 - threshold`;
- `budget`: configuration at exactly `--tale_target_remove`, when reached.

Search and final evaluation are disjoint: MMLU uses validation/test,
HellaSwag and WinoGrande use train/validation, and GSM8K uses train/test.
Few-shot demonstrations are removed from a same-split search/calibration set
before subsampling, so a query cannot contain its own gold answer in context.
HumanEval has no legal labeled search split and is rejected. The TALE search
semantics were aligned against the
[authors' repository at a fixed revision](https://github.com/omyokun/tale/tree/d10cec53295ab4ce544e553509935ffcf0ac3e0d)
in an independent implementation; the project user reports having author
approval for source reuse. Source provenance is retained here and in the
module documentation.

### CalibratedSkip

CalibratedSkip builds teacher-forcing calibration requests from labeled
examples on the same explicitly disjoint splits listed above. It never falls
back to the final evaluation split, and calibration on HumanEval is disabled.
It supports four layer-level metrics:

- `activation_ratio`: fraction of positive values in each layer's output hidden
  states over non-padding tokens.
- `gradient_value`: layer-level sum of `abs(loss_gradient)` over the layer's
  parameters, using the calibration labels.
- `gradient_trace`: layer-level sum of `abs(weight * loss_gradient)` over the
  layer's parameters, using the calibration labels.
- `shapley_value`: layer-level sum of row-wise Shapley values over each 2D
  parameter, using a Fisher-information approximation to the Hessian.

CalibratedSkip does not run benchmark evaluation and does not automatically
bypass any layers. It only writes the per-layer metrics so you can inspect them
offline. After choosing layers from the saved metrics, run ManualSkip with those
1-based layer numbers.

### ManualSkip

Bypasses the exact 1-based transformer layer numbers provided by the user. For
each skipped layer, the model does not execute that transformer block; the
previous layer's hidden state is passed directly to the following layer. The
model still runs to the final layer after applying those bypasses, and uses the
same final-logit and generation path as the full-model baseline.

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

@article{men2024shortgpt,
  title   = {{ShortGPT}: Layers in Large Language Models are More Redundant
             Than You Expect},
  author  = {Men, Xin and Xu, Mingyu and Zhang, Qingyu and Wang, Bingning and
             Lin, Hongyu and Lu, Yaojie and Han, Xianpei and Chen, Weipeng},
  journal = {arXiv preprint arXiv:2403.03853},
  year    = {2024}
}

@inproceedings{song2024sleb,
  title     = {{SLEB}: Streamlining {LLM}s through Redundancy Verification and
               Elimination of Transformer Blocks},
  author    = {Song, Jiwon and Oh, Kyungseok and Kim, Taesu and Kim, Hyungjun
               and Kim, Yulhwa and Kim, Jae-Joon},
  booktitle = {Proceedings of the 41st International Conference on Machine
               Learning},
  year      = {2024}
}

@inproceedings{naim2026tell,
  title     = {{TELL-TALE}: Task Efficient {LLM}s with Task Aware Layer
               Elimination},
  author    = {Naim, Omar and Sharma, Krish and Barman, Niyar R and
               Asher, Nicholas},
  booktitle = {Findings of the Association for Computational Linguistics:
               {ACL} 2026},
  pages     = {22616--22638},
  year      = {2026},
  doi       = {10.18653/v1/2026.findings-acl.1136}
}

```
