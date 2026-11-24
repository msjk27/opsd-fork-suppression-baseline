# E6 results: does masking the decision positions prevent the OPSD degradation?

Final results 2026-11-20. The design was pre-registered on 2026-11-19 in [e5e6_preregistration.md](e5e6_preregistration.md), before any E6 model was trained. Model: Qwen3-1.7B, thinking ON student and teacher, one training seed (42) per arm.

## Summary

- **The mask did not prevent the accuracy loss.** The decision-masked model scores 0.324 against base's 0.364 (−0.040), the same as plain OPSD.
- **Primary endpoint:** decision − matched = +0.002 [−0.010, +0.016].
- **The mask barely changed what the model learned.** The decision-masked model still learned 88–99% of the decision shift (less Wait, earlier commit). The shift is learned through the other ~98% of tokens, not locally at the decision positions.
- **Conclusion:** position masks are not a fix. Because the shift was not actually blocked, E6 cannot tell whether the shift causes the accuracy loss.

## Why this experiment

- The paper's proposed mechanism, fork suppression, measured weak on our setup. With the solution in context, the teacher's fork rate moves about −0.6 points from 14.7% (paper, 4B: about −58% relative), and a shuffled context lowers it as much.
- A much clearer change was measured at **paragraph starts in the thinking**. There the model picks how the next paragraph opens: go on, self-correct (Wait/Hmm/Alternatively), contrast (But/However/Actually), conclude (Therefore/Thus), or commit (`**Final Answer**`).
- **E4, step 0, solution context:** the teacher lowers p(Wait/Hmm/Alt.) by 17% and raises p(commit) by 27%, with no change in top-3 entropy. That is a specific redirection, not general sharpening. Source: `runs/e1e4/report.md`.
- **E1-C, pre-registered:** on base's rollouts, Run A − base is p(commit) +0.0042 (about 2×) and p(Wait/Hmm/Alt.) −0.0216. Both survive Bonferroni. Most of the shift is already present at step 100.
- **E1-D, policy swap:** the final stopping decision explains only ~8% of the gap. Most is carried by the reasoning before the stop (P2 +0.037 [+0.014, +0.063]). So E6 targets every paragraph start, not just the final commit point.

### Why paragraph starts and not self-correction tokens

| | self-correction token mask | paragraph-start mask (E6) |
|---|---|---|
| positions | where the student sampled Wait/Hmm/Alternatively | every position after a blank line inside the thinking |
| share of tokens | 0.59% | 1.75% |
| share of loss | ~3% (3.2% with the solution context) | 6.4% |
| covers the commit choice | no | yes |

Shares come from E4 at step 0 and from the E6 training logs. About 25% of self-correction words sit at paragraph starts; the rest are mid-paragraph.

1. **The loss is a full-vocabulary JSD.** p(Wait) is pushed down at every paragraph start, not only where Wait was sampled (about 15% of paragraph starts).
2. **The shift goes both ways: less Wait and more commit.** Both are options at the same position, and a self-correction token mask leaves commit untouched.
3. **Treatment and measurement sit at the same positions.** This makes the manipulation check direct.

A self-correction-only mask was not trained. Several results suggest it would not help:
- E6's wider mask blocked only 1–12% of the shift.
- The paper's App. F epistemic-only mask (8B-teacher OPD) scored 0.385, against 0.386 for a random mask.
- AD-OPSD's masking diagnostic restored thinking density but not accuracy.

## Design

| arm | tokens removed from the loss | role |
|---|---|---|
| `e6-decision-s42` | all decision positions | treatment |
| `e6-matched-s42` | none of those; the same number of other thinking tokens from the same rollout, matched one to one on their current loss | primary control |
| `e6-random-s42` | none of those; the same number of other thinking tokens, at random | secondary control |
| `vanilla-4x16-s42` | nothing | reference, trained on the other server |

- All arms use Run B's config (paper Table 9) and change only `research_mask`. Each trains 4 GPUs × 16 accumulation for 234 steps.
- Evaluation: AIME24, AIME25 and HMMT25 (90 problems), T 0.6, top-p 0.95, 38,912-token cap, 48 rollouts per problem.

## Results

### Accuracy

| arm | rollouts / problem | acc (all) | avg@16 | mean thinking tokens | hit cap | − base [95% CI] |
|---|---|---|---|---|---|---|
| base | 48 | 0.364 | 0.362 | 17,701 | 2.5% | — |
| e6-decision | 48 | 0.324 | 0.324 | 12,925 | 0.7% | −0.040 [−0.060, −0.020] |
| e6-matched | 48 | 0.322 | 0.317 | 12,973 | 0.8% | −0.042 [−0.062, −0.024] |
| e6-random | 48 | 0.318 | 0.322 | 13,262 | 1.0% | −0.047 [−0.066, −0.028] |
| vanilla-4x16 | 32 | 0.318 | 0.319 | 13,225 | 1.5% | −0.046 [−0.067, −0.028] |
| Run B (8×8) | 16 | 0.338 | 0.338 | 12,952 | 0.9% | −0.026 [−0.047, −0.005] |

### Check 1: balance (passed)

| run | tokens dropped | loss dropped |
|---|---|---|
| e6-decision | 1.75% | 6.39% |
| e6-matched | 1.74% | 6.30% |
| e6-random | 1.75% | 1.91% (count-matched only, by design) |

### Check 2: manipulation (passed nominally only)

These are probabilities at the paragraph starts of base's rollouts; every model scores the same states.

| model | p(commit) | shift | p(Wait/Hmm/Alt.) | shift |
|---|---|---|---|---|
| base | 0.0041 | — | 0.1710 | — |
| e6-decision | 0.0077 | +0.0036 | 0.1512 | −0.0199 |
| e6-matched | 0.0081 | +0.0041 | 0.1504 | −0.0206 |
| e6-random | 0.0079 | +0.0038 | 0.1512 | −0.0199 |
| Run B | 0.0082 | +0.0041 | 0.1510 | −0.0201 |

decision − matched is −0.0004 [−0.0005, −0.0003] for p(commit) and +0.0008 [+0.0005, +0.0011] for p(Wait). Both are in the predicted direction with CIs excluding 0, so the check passes as pre-registered. But the decision arm still learned 88% of the commit shift and 97–99% of the Wait shift. The pre-registration set no minimum size for this check, which is a design flaw.

### Contrasts (accuracy, all pooled rollouts)

| contrast | difference [95% CI] |
|---|---|
| **decision − matched (primary)** | **+0.002 [−0.010, +0.016]** |
| decision − random | +0.007 [−0.006, +0.019] |
| decision − vanilla-4x16 (cross-server) | +0.007 [−0.008, +0.023] |
| matched − vanilla-4x16 (cross-server) | +0.004 [−0.011, +0.021] |
| random − vanilla-4x16 (cross-server) | −0.000 [−0.015, +0.015] |
| decision − base | −0.040 [−0.060, −0.020] |
| vanilla-4x16 − Run B (replication check) | −0.020 [−0.039, −0.003] |

## Interpretation

**Pre-registered reading.** Both checks passed, and the primary CI excludes the ~0.04 vanilla gap. By the pre-registered rule this reads as "learning at the decision positions is not the main cause".

**What the data support.** The manipulation check passed only nominally, so the situation is effectively the "manipulation fails" case. E6 then says nothing about whether the decision shift causes the accuracy loss. What it does show:

1. **The decision habit is not learned locally.** It spreads from the remaining ~98% of tokens. In E4, ordinary text tokens carry ~78% of the loss, and the teacher lowers the sampled token's probability there too.
2. **Position masks cannot block it.** This likely applies to self-correction-token masks and fork masks as well.
3. **Treatments should change the teacher signal broadly.** Candidates are the teacher context (E5) and target adjustment over many tokens (AD-OPSD acts on ~42% of the loss).

## Limits and open items

- **One training seed per arm.** The paper's three vanilla seeds span 0.308–0.324. The bootstrap CIs cover problems and rollouts, not seed variation.
- **`vanilla-4x16-s42` is incomplete.** It has 32 of 48 rollouts, and its decision probe has not run. Both are held in altair's eval queue, and neither affects the primary contrast.
- **vanilla-4x16 trained on the other server**, so the E6 − vanilla contrasts cross servers.
- **The training setup itself matters:** 4×16 vanilla scored 0.020 below 8×8 Run B under the same config.
- **A self-correction-token mask was not trained.** The section on it above is inference.

## Note for E5

E4 already measured the teacher under all four E5 contexts, at step-0 paragraph starts:

| context | Δp(Wait) | Δp(commit) |
|---|---|---|
| answer | −13% | +26% |
| solution | −17% | +27% |
| shuffled | −10% | 0% |
| R1 thinking trace (`e5-cot`) | −30% | +53% |

Every tested context lowers the teacher's p(Wait). `e5-cot` is therefore expected to strengthen the shift, not to act as a "more deliberative teacher". Testing that idea needs a context that raises the teacher's p(Wait), which the E4 probe can screen for before any training.

## Next steps (proposed)

1. **Resume E5.** Two trainings and their evals remain.
2. **Find a context that raises the teacher's p(Wait),** screening candidates with the E4 probe before training.
3. **E6 follow-up for causality:** anchor the decision positions to base's distribution instead of masking them, so the shift is actually blocked.

## Reproduce

- Configs: `configs/research/e6_{decision,matched,random}_3090.json`. Mask code: `research/masking.py`. The code is identical to what trained on altair; file hashes were checked on 2026-11-24.
- Report: `python -m research.e5e6_report`. The full output is in `runs/e5e6_report_e6final_20261120.md`, which is server-local and not in git.
