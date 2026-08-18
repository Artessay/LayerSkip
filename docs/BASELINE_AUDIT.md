# Baseline audit and roadmap

Last reviewed: 2026-08-15. The literature pass used OpenAlex title search and
primary arXiv/ACL/AAAI/NeurIPS records. Search indexes lag preprints, so this is
a prioritized engineering roadmap rather than an exhaustive 2026 survey.

## What the current strategies actually measure

| CLI name | Status | Execution | Interpretation |
|---|---|---|---|
| `none` | sound reference path | full model | unpruned quality reference |
| `manualskip` | sound structural baseline | selected blocks are bypassed | arbitrary block pruning; no recovery tuning |
| `layerskip` | static early-exit baseline | suffix blocks are bypassed | generic early exit, **not** a reproduction of LayerSkip training |
| `caml` | CALM inference | token-wise intermediate exit | requires intermediate-head training for faithful results |
| `gateskip` | learned residual gating | token-wise quantile budget | requires jointly fine-tuned gate weights |
| `calibratedskip` | metric collector | full forward/backward | produces scores but does not prune automatically |
| `shortgpt` | paper-aligned one-shot search | physical block removal for final evaluation | global PG19 Block Influence ranking; no recovery tuning |
| `sleb` | paper-aligned greedy search | physical block removal for final evaluation | global iterative WikiText NLL minimization |
| `tale` | paper/author-code-aligned greedy search | physical block removal for final evaluation | task-specific accuracy search on a labeled non-test split |

`execution_mode` is exposed in each run summary. Only `structural_pruning` runs
physically shorten the block list during final evaluation. Wall-clock,
peak memory, parameter count and FLOPs still need to be measured separately.

## Correctness findings

1. Earlier revisions ran every layer for `layerskip`, then selected an already
   computed hidden state. This saved no compute. The implementation now bypasses
   the suffix and uses the native final norm/head.
2. The Meta LayerSkip paper trains checkpoints with layer dropout and an
   early-exit loss. Applying an exit head to an ordinary Hugging Face checkpoint
   tests generic truncation, not the paper's method. A faithful reproduction
   must add the released LayerSkip checkpoints and self-speculative decoding.
3. The CLI keeps the historical `caml` spelling for compatibility, but now
   implements CALM's token-wise softmax-response/hidden-state confidence and
   decaying threshold. Untrained intermediate exits require explicit opt-in.
4. GateSkip now rejects missing trained gates and implements sigmoid vector
   gates plus per-layer quantile token budgets; the old hidden-change proxy was
   removed.
5. Earlier MMLU few-shot examples were sampled across concatenated subjects.
   The implementation now uses subject-specific development examples and the
   standard subject heading.
6. HellaSwag previously divided by whitespace word count. It now reports both
   raw accuracy and tokenizer-token-normalized `accuracy_norm`.
7. HumanEval executes generated code with only a timeout. It has no real
   filesystem, process, syscall, or network sandbox. Calibration and TALE
   search are now disabled for HumanEval because it has no labeled split
   disjoint from test.
8. Chat templates are now opt-in (`--apply_chat_template`) and recorded in the
   run configuration; canonical benchmark prompts are the default.
9. Calibration scores are mostly size-dependent sums. Comparing raw gradient
   sums across differently parameterized blocks needs normalization. The
   implemented `shapley_value` is a gradient outer-product approximation, not
   exact or sampled Shapley attribution.
10. HellaSwag and WinoGrande now use train for calibration/search and validation
    for final evaluation; GSM8K uses train/test; MMLU uses validation/test.
    Search evaluation receives an explicit document list and cannot silently
    load the final split.
11. Existing result files do not report latency, memory, effective parameter
    count, executed blocks, or FLOPs, so they support quality comparisons only.

## Implemented block-pruning baselines

### ShortGPT

For block `i`, the implementation computes the paper's Block Influence over
non-padding calibration tokens:

```text
BI_i = 1 - E_t[cos(h_i,t, h_(i+1),t)].
```

Forward pre/post hooks capture the block's actual input and raw output, so the
last block is not accidentally compared after the model's final norm. All
blocks are scored together in one pass over non-overlapping PG19 validation
chunks. The globally lowest scores are removed; ties retain original layer
order. This search is model-global and its selected layer set is reused across
all requested tasks.

Fidelity boundary: the repository implements direct one-shot block removal,
not ShortGPT's optional recovery fine-tuning. Search costs one dense pass over
the selected PG19 corpus, followed by ordinary final benchmark evaluation.

### SLEB

Let `S_r` be blocks already removed at round `r`. SLEB evaluates every eligible
remaining block on the same fixed WikiText token batches and selects

```text
l_r = argmin_l mean_t[-log p_(model without S_r union {l})(x_t | x_<t)].
```

Candidate scores are recomputed after every committed removal; an isolated
per-block score is not reused. Search is therefore roughly quadratic in depth:
removing `k` blocks evaluates `N + (N-1) + ... + (N-k+1)` candidates before
barriers. The paper-faithful early/latter barriers are `0/0`; `1/1` is available
to reproduce the released code's default protection. The selected set is
model-global and shared across tasks.

More exactly, `R` removals with early/latter barriers `be`/`bl` require
`R(N-be-bl) - R(R-1)/2` full calibration evaluations.

### TALE

TALE evaluates the dense model and every single-layer deletion on a labeled
task search split. At round `r` it commits the remaining layer maximizing task
accuracy. Its acceptance floor is fixed once:

```text
A_floor = A_dense - 0.08
```

The tolerance is eight percentage points and is never subtracted from the
previous greedy configuration. Each task has an independent trace and selected
architecture. The exposed variants are:

- `best`: highest search accuracy, with deeper configurations winning ties;
- `bsba`: deepest configuration at least as accurate as the dense baseline;
- `threshold_final`: deepest configuration at or above the fixed floor;
- `budget`: the exact requested removal depth, or an explicit unavailable
  status when search stops before reaching it.

MMLU searches validation and evaluates test; HellaSwag and WinoGrande search
train and evaluate validation; GSM8K searches train and evaluates test.
When demonstrations and search data share a training split, exact few-shot
documents are excluded before search subsampling; this prevents a query from
carrying its own gold answer in context. HumanEval is rejected because it
offers no labeled non-test search split.
The CLI uses the full labeled search split by default, matching the authors'
single-GPU default; `--tale_search_max_samples` is an explicit cost-control
option and should be reported whenever it is used.

The TALE semantics were checked against the
[official source at commit `d10cec5`](https://github.com/omyokun/tale/tree/d10cec53295ab4ce544e553509935ffcf0ac3e0d)
in an independent implementation; the project user reports having author
approval for source reuse. The local implementation adapts the greedy search
to this repository's task and model interfaces while retaining paper,
repository, and fixed-revision provenance.

### Shared execution and trace policy

All three methods store versioned, atomic JSON search traces under
`results/<model>/pruning/<method>/`. A trace records the complete candidate
scores or Block Influence ranking, original 0-based layer IDs, corpus/search
configuration, seed, exact SLEB token fingerprint or TALE request fingerprint,
model/runtime identity, and library provenance. ShortGPT likewise validates an
exact token-stream fingerprint before reusing its one-shot ranking. Local
checkpoints are identified by model/tokenizer artifact contents rather than
their path alone. Only an exact matching trace is resumed.

Final benchmark evaluation uses a physically shortened transformer
`ModuleList`, updates common model depth fields, and reindexes attention cache
layer IDs. The context is reversible, so removed modules remain referenced for
restoration; it measures the shortened forward/decode path but does not by
itself demonstrate a smaller checkpoint or lower resident parameter memory.

## Further baselines worth adding

Recommended first, because they fit this repository's block-pruning focus:

| Priority | Work | What to implement |
|---|---|---|
| P0 | LaCo (Yang et al., 2024) | Layer collapse/merging rather than identity bypass |
| P0 | BlockPruner (ACL Findings, 2025) | Fine-grained removal of attention/MLP sub-blocks using perplexity increase |
| P0 | Shortened LLaMA (Kim et al., 2024) | Depth-pruning criteria plus retraining-method comparison |
| P0 | LLM-Pruner (Ma et al., 2023) | Dependency-aware structured pruning plus LoRA recovery |
| P1 | SliceGPT (Ashkboos et al., 2024) | PCA/orthogonal-transform width pruning |
| P1 | FLAP (An et al., 2024) | fluctuation-based structured channel pruning and bias compensation |
| P1 | SlimGPT / DISP-LLM (NeurIPS 2024) | Layer-wise and dimension-independent structured pruning |
| P1 | Sheared LLaMA (Xia et al., 2023) | learned targeted structured pruning with dynamic batch loading |
| P1 | Compresso (Guo et al., 2023) | collaborative structured pruning with LoRA and L0 gates |
| P2 | Wanda (Sun et al., 2023) | activation-aware weight pruning |
| P2 | SparseGPT (Frantar & Alistarh, 2023) | second-order one-shot unstructured/semi-structured pruning |
| P2 | Pruner-Zero (Dong et al., 2024) | search-discovered pruning metric |
| P2 | Wanda++ (ACL Findings, 2025) | regional-gradient extension of Wanda |
| P2 | KVPruner (ICASSP 2025) | structural pruning aimed at decode latency and KV-cache memory |
| P2 | NVIDIA Minitron (2024) | depth/width/head/neuron pruning with distillation |
| P2 | DarwinLM (2025) | evolutionary search over structured pruning and lightweight recovery |

Weight-sparsity methods (Wanda/SparseGPT) answer a different systems question
from block removal: speedups require sparse kernels. Report them in a separate
family rather than ranking all methods only by nominal sparsity.

## Evaluation protocol needed for credible comparisons

- Use a disjoint external corpus (for example PG19/WikiText) for model-global
  criteria. Task-aware methods may use labeled train/validation data, but never
  the downstream final-evaluation split.
- Evaluate with lm-evaluation-harness-compatible prompts and metrics; include
  both `acc` and token-normalized `acc_norm` where standard.
- Match removed parameters or measured FLOPs, not merely “number of layers”.
- Report dense parameter count, effective parameters, executed blocks/token,
  prefill/decode latency, peak memory, throughput, and quality.
- Run base and instruct checkpoints separately and record tokenizer, chat
  template, revision, dtype, device, seed and library versions.
- Add recovery-tuning as a separate experimental axis so one-shot and tuned
  pruning are not conflated.

## Primary references

- [LayerSkip](https://arxiv.org/abs/2404.16710)
- [CALM](https://arxiv.org/abs/2207.07061)
- [SparseGPT](https://arxiv.org/abs/2301.00774)
- [Wanda](https://arxiv.org/abs/2306.11695)
- [LLM-Pruner](https://arxiv.org/abs/2305.11627)
- [Sheared LLaMA](https://arxiv.org/abs/2310.06694)
- [Compresso](https://arxiv.org/abs/2310.05015)
- [SliceGPT](https://arxiv.org/abs/2401.15024)
- [ShortGPT](https://arxiv.org/abs/2403.03853)
- [SLEB](https://arxiv.org/abs/2402.09025)
- [TELL-TALE](https://aclanthology.org/2026.findings-acl.1136/)
- [ShortGPT reference code](https://github.com/icip-cas/ShortGPT)
- [SLEB reference code](https://github.com/jiwonsong-dev/SLEB)
- [TALE fixed source revision](https://github.com/omyokun/tale/tree/d10cec53295ab4ce544e553509935ffcf0ac3e0d)
- [FLAP](https://arxiv.org/abs/2312.11983)
- [Shortened LLaMA](https://arxiv.org/abs/2402.02834)
- [BlockPruner](https://aclanthology.org/2025.findings-acl.262/)
- [Wanda++](https://aclanthology.org/2025.findings-acl.224/)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
