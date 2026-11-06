# OPSD degradation on Qwen3-1.7B (thinking): reproduction results and next steps

Status: **Phase 2a (reproduction) complete, 2026-11-06 03:48.** Fork-aware variants have not been started.
Paper: Kaur et al., arXiv 2607.05184, "Rethinking OPSD for Thinking Models".

## 1. Headline: the OPSD degradation reproduces

avg@16 at a 38,912-token cap, T=0.6, top-p 0.95, 30 problems × 16 samples per benchmark.

| Benchmark | Base (ours) | OPSD (ours) | Δ ours | Paper base → OPSD (Δ) |
|---|---|---|---|---|
| AIME24 | 0.477 | 0.444 | −0.033 | 0.502 → 0.435 (−0.067) |
| AIME25 | 0.360 | 0.319 | −0.042 | 0.398 → 0.302 (−0.096) |
| HMMT25 | 0.248 | 0.194 | −0.054 | 0.215 → 0.185 (−0.030) |
| **Average** | **0.362** | **0.319** | **−0.043** | **0.372 → 0.308 (−0.064)** |

- **The base model matches the paper** (0.362 vs 0.372, within sampling noise), which validates the eval setup.
- **OPSD lowers accuracy on all three benchmarks.** A paired bootstrap over the 90 problems gives Δ = −0.043, 95% CI [−0.067, −0.019], with P(Δ ≥ 0) = 0.0002. 36 problems got worse, 10 got better, and 44 were unchanged.
- **The drop is about two-thirds of the paper's** (−0.043 vs −0.064), and both 95% intervals overlap it. The per-benchmark pattern differs: our largest drop is on HMMT25, the paper's on AIME25. With 30 problems per benchmark, per-benchmark differences are noisy.

## 2. Behavioural changes match the paper's mechanism

Long-budget eval responses (38,912 cap):

| Benchmark | Mean length, base → OPSD | Self-correction markers per 1k tokens, base → OPSD |
|---|---|---|
| AIME24 | 17,790 → 14,014 (−21%) | 8.43 → 7.82 |
| AIME25 | 18,826 → 13,638 (−28%) | 9.32 → 8.30 |
| HMMT25 | 19,565 → 14,615 (−25%) | 8.56 → 8.30 |

The marker count is a word-boundary proxy over wait/hmm/maybe/however/verify/check/reconsider/backtrack, not the paper's full taxonomy. Cap hits were rare in both models (≤3%), so the shorter responses come from earlier termination, not truncation. Over training, rollout length at the 4096 cap fell from 3,426 to 3,133 tokens, and the share hitting the cap fell from 56% to 39%.

## 3. Fork suppression diagnostic (paper §4.1; the paper ran it on Qwen3-4B, not 1.7B)

There are 32 OpenThoughts training prompts, with one 4096-token rollout each. The teacher is the base model, given no context, the answer only, or the full solution. Values are medians over rollouts.

| Trajectories | Teacher context | Fork % | Lock % |
|---|---|---|---|
| Base student | none | 14.2 | 70.3 |
| Base student | answer only | 13.1 | 72.4 |
| Base student | full solution | 13.7 | 71.4 |
| OPSD student | none | 12.9 | 71.6 |
| OPSD student | answer only | 12.1 | 73.6 |
| OPSD student | full solution | 12.5 | 72.1 |

*Paper (4B, none → full solution): fork 8.3 → 3.5, lock 69 → 77.*

- **Definition matters.** With the main-text rule (top-1 probability < 0.25 means fork), 1.7B has only about 0.07% fork positions, too few to compare. The table uses the paper's Appendix D rule (top-3 normalized entropy: ≥ 0.60 fork, ≤ 0.20 lock). [research/rescore_topk.py](research/rescore_topk.py) reports both definitions.
- **Direction reproduces:** privileged context lowers fork rates and raises lock rates. Fork is lower with answer-only context in 30 of 32 rollouts, and lock is higher in 32 of 32.
- **Magnitude doesn't:** the change is about 1 point here, against 4.8 in the paper for 4B. Unlike the paper's monotonic pattern, the answer-only context changes it more than the full solution.
- **After OPSD training, the student's own rollouts have fewer forks** (14.2 → 12.9% under the unprivileged teacher) **and more locks** (70.3 → 71.6%), consistent with the student absorbing the teacher's fork suppression.

## 4. Deviations from the paper's setup (all deliberate and recorded)

| Item | Paper | Here | Why |
|---|---|---|---|
| Hardware | 8× H100 80 GB | 8× RTX 3090 24 GB | Available hardware |
| JSD loss computation | Full-vocab tensors | Chunked (512 positions) | The full-vocab version runs out of memory on 24 GB. Same math: loss identical, gradients within the bf16 noise floor ([bench/check_chunked_loss.py](bench/check_chunked_loss.py)). |
| vLLM memory share | 0.15 (repo config) | 0.40 | 0.15 leaves about 0.4 GB of KV cache on 24 GB |
| Everything else | Full FT, LR 5e-6, batch 64, 15k prompts, 4k cap, α=0.5, frozen teacher, 1 epoch, seed 42 | Same | Config: [configs/research/vanilla_4096_3090.json](configs/research/vanilla_4096_3090.json) |
| Answer-only diagnostic context | `Answer` column | Last `\boxed{}` of the solution | 27% of `Answer` values in the 15k set are truncated (e.g. `\dfrac{85`) |
| Diagnostic model | Qwen3-4B | Qwen3-1.7B | Matches the trained model |

Single seed (42) for training. Eval and diagnostic sampling seeds were not fixed.

## 5. Proposed next steps (need your decision)

1. **Fork-aware variants**, now justified because the degradation reproduces: Fork Mask, Random Mask (same number of masked tokens, as the control) and positive-log-ratio-at-forks (D4). Each needs about 6.5 h of training plus 2 h of eval. The fork definition (main text vs Appendix D) has to be chosen first, because the main-text rule marks almost nothing on 1.7B.
2. **Optional robustness checks:** a second training seed (the measured drop is −0.043 with CI [−0.067, −0.019]), and a mid-budget eval (8k cap) to see whether OPSD helps at short budgets, as the paper reports.

## 6. Where everything is

- Runs: `/mnt/SSD3/mssuk/opsd/runs/` (`base/eval`, `vanilla-3090-s42/{checkpoint,eval}`, `probe-{base,opsd}-4096-s42`). The repo `runs/` folder is a symlink to it.
- Comparison numbers: `/mnt/SSD3/mssuk/opsd/runs/comparison_base_vs_opsd.txt`
- Environment: `/mnt/SSD3/mssuk/opsd/venv`. Scripts: `/mnt/SSD3/mssuk/opsd/scripts/` (pipeline, work-queue eval, progress).
