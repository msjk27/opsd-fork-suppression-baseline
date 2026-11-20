# E5 and E6: pre-registered design

Written 2026-11-19, before any E5 or E6 model was trained. `research/e5e6_report.py` implements these endpoints.

## Common protocol

- All runs start from `configs/research/vanilla_4096_3090_paper.json` (paper Table 9, seed 42) and change only the factor under test. `tests/test_e5e6.py` enforces this.
- Every run takes 64 rollouts per optimizer step, the same prompts in the same order, and 234 steps.
- Evaluation is T 0.6, top-p 0.95, 38,912-token cap, on AIME24, AIME25 and HMMT25 (90 problems).
- Every arm, including `base` and `vanilla-4x16-s42`, is evaluated with **48 rollouts per problem**: the paper's avg@16 batch plus two more batches of 16 with different eval seeds (`<run>-top1`, `<run>-top2`). Batches of 16 keep the memory profile of the paper's eval.
- **How and where runs train.** Every E5 and E6 run, and their vanilla control `vanilla-4x16-s42`, trains as 4 GPUs × 16 accumulation: 64 rollouts per step, the same prompt batches, the setup the paper's code uses for its answer-only runs. Both servers have RTX 3090s, the same driver and the same copied Python environment. E5 and the vanilla control train on one server, so every E5 contrast stays on it. The three E6 arms train on the other, so the primary E6 contrast and its controls stay on it; only the secondary E6 − vanilla contrasts cross servers. Run B (8 GPUs × 8) is now only a replication check. Training runs keep only their final model.

Why 48: measured on our evals, the smallest difference detectable between two OPSD models (80% power, paired over problems) is 0.029 at 16 rollouts and 0.017 at 48. The full vanilla gap to base is 0.024 to 0.043, so 16 rollouts could miss a complete recovery.

## E5: which part of the teacher's context drives the change?

| arm | teacher context in the training template |
|---|---|
| `vanilla-4x16-s42` (control) | the gold solution |
| `e5-answer` | only the final answer |
| `e5-shuffled` | the solution of a different problem with a different final answer |
| `e5-cot` | the R1 thinking trace for the same problem |

E5 is an estimation experiment, with no directional claim. For each arm the report gives accuracy − base, accuracy − vanilla (95% CIs), and the change in p(commit) and p(Wait/Hmm/Alternatively) at the paragraph starts of base's rollouts.

Deviation from the paper's code: its answer-only runs use the `Answer` column, where 27% of values are truncated. We use the last non-empty `\boxed{}` of the solution. Its `max_prompt_length` of 8,192 for those runs is kept at 25,000 here so all arms match; no teacher prompt exceeds 6,350 tokens, so nothing is truncated either way.

## E6: do the decision positions carry the damage?

A **decision position** is a paragraph start inside the thinking: the token after a blank line, before `</think>`. E1-C and E4 measure the policy shift at these positions.

| arm | tokens removed from the loss |
|---|---|
| `e6-decision` | all decision positions |
| `e6-matched` (primary control) | none of them; instead the same number of other thinking tokens from the same rollout, matched one to one on their current loss |
| `e6-random` (secondary control) | none of them; the same number of other thinking tokens, at random |

The mask only changes which tokens enter the loss. Rollouts, attention and importance weights are unchanged. Decision positions hold about 2% of tokens but 6–9% of the loss on real rollouts, so a random control alone would confound "these positions matter" with "less training signal".

**Primary endpoint:** accuracy(`e6-decision`) − accuracy(`e6-matched`) > 0, 95% paired bootstrap CI over problems.

It can be read only if both checks pass:

1. **Balance:** in the training logs, the matched control drops the same token count and loss share as the decision mask, within 10%.
2. **Manipulation:** on base's rollouts, `e6-decision` has lower p(commit) and higher p(Wait/Hmm/Alternatively) than `e6-matched` (95% CIs exclude 0).

Reading:

- **Both checks pass, endpoint > 0:** the learning at decision positions contributes causally to the accuracy loss.
- **Both checks pass, endpoint ≈ 0 with a CI that excludes the vanilla gap:** the shift at these positions is not the main cause.
- **Both checks pass, CI still includes both:** inconclusive; add seeds.
- **Manipulation fails:** the shift is learned through other positions, and the experiment says nothing about its effect on accuracy.

Secondary: `e6-decision` − base asks whether the method actually beats base, which is the project goal. `e6-decision` − `vanilla-4x16-s42` asks how much of the vanilla loss the mask removes. `vanilla-4x16-s42` − Run B is a replication check on the training setup.

## Limits

One training seed per arm unless more are added (an arm can pool several; seed 43 is the planned replicate). Bootstrap CIs cover problems and rollouts, not training-seed variation. The paper's three vanilla seeds span 0.308 to 0.324, so a seed-42 result near the detection limit needs a replicate before it is claimed.
