# Baseline audit and roadmap

Last reviewed: 2026-08-13. The literature pass used OpenAlex title search and
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

`execution_mode` is now exposed in each run summary. Only `structural` runs can
claim savings in executed transformer blocks. Wall-clock,
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
5. MMLU few-shot examples are sampled across concatenated subjects. Standard
   MMLU uses subject-specific development examples and a subject heading.
6. HellaSwag previously divided by whitespace word count. It now reports both
   raw accuracy and tokenizer-token-normalized `accuracy_norm`.
7. HumanEval executes generated code with only a timeout. It has no real
   filesystem, process, syscall, or network sandbox. It also uses the test set
   for calibration, which leaks benchmark information into pruning decisions.
8. Chat templates are now opt-in (`--apply_chat_template`) and recorded in the
   run configuration; canonical benchmark prompts are the default.
9. Calibration scores are mostly size-dependent sums. Comparing raw gradient
   sums across differently parameterized blocks needs normalization. The
   implemented `shapley_value` is a gradient outer-product approximation, not
   exact or sampled Shapley attribution.
10. Existing result files do not report latency, memory, effective parameter
    count, executed blocks, or FLOPs, so they support quality comparisons only.

## Baselines worth adding

Recommended first, because they fit this repository's block-pruning focus:

| Priority | Work | What to implement |
|---|---|---|
| P0 | ShortGPT (Men et al., 2024) | Block Influence score, one-shot global block removal, optional recovery tuning |
| P0 | SLEB (Song et al., 2024) | Iterative transformer-block elimination measured by calibration loss |
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

- Use a disjoint calibration corpus (for example C4/WikiText) and never the
  downstream test split.
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
- [FLAP](https://arxiv.org/abs/2312.11983)
- [Shortened LLaMA](https://arxiv.org/abs/2402.02834)
- [BlockPruner](https://aclanthology.org/2025.findings-acl.262/)
- [Wanda++](https://aclanthology.org/2025.findings-acl.224/)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
