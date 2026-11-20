"""Report for E5 (teacher-context controls) and E6 (decision-position loss mask); works on partial results.
The endpoints and reading rules are fixed in docs/e5e6_preregistration.md.

Inputs under --runs-root:
  <run>/eval/<bench>_38912/*.jsonl         graded eval rollouts (run_generate.py --research_metrics)
  <run>/checkpoint/checkpoint-*/           trainer_state.json with the mask/* training metrics (E6)
  e1-dec/<run>-on-base/shard*.jsonl        research.self_entropy --label <run> --run base (decisions on base's rollouts)
An arm is a list of runs whose rollouts are pooled: extra eval batches of one model (<run>-top1, -top2) or extra
training seeds. --arms FILE (JSON {arm: [runs]}) replaces the default mapping. CIs: paired bootstrap over the 90 problems.
Usage: python -m research.e5e6_report [--runs-root runs] [--arms arms.json] > report.md
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from research.e1e4_report import Data, acc, dec_rows, paired_dq, pboot, per_problem, train_log
from research.evalio import load_records

# Every E5/E6 arm trains as 4 GPUs x 16 and is compared with `vanilla-4x16`, trained the same way. `run-b` is the earlier
# 8 GPUs x 8 vanilla run: only a replication check, with its original 16 eval rollouts per problem.
TRAINED = {'vanilla-4x16': 'vanilla-4x16-s42', 'e5-answer': 'e5-answer-s42', 'e5-shuffled': 'e5-shuffled-s42',
           'e5-cot': 'e5-cot-s42', 'e6-decision': 'e6-decision-s42', 'e6-matched': 'e6-matched-s42',
           'e6-random': 'e6-random-s42'}
# <run>-top1 and -top2 hold 16 more eval rollouts each of the same model, generated with another eval seed.
ARMS = {arm: [run, f'{run}-top1', f'{run}-top2'] for arm, run in {'base': 'base', **TRAINED}.items()}
ARMS['run-b'] = ['vanilla-3090-paper-s42']
CONTROL = 'vanilla-4x16'
E6 = ('e6-decision', 'e6-matched', 'e6-random')
PAPER_N = 16           # rollouts per problem in the paper's avg@16 protocol
BALANCE_TOLERANCE = .10


def is_extra_eval(run):
    return re.search(r'-top\d+$', run) is not None


def control_of(arm):
    """The vanilla run this arm is compared with: the one trained the same way, or Run B for the control itself."""
    return 'run-b' if arm == CONTROL else CONTROL


def records(root, run):
    try:
        return load_records(root, run)
    except (FileNotFoundError, KeyError):   # not generated yet, or merged but not graded yet
        return []


def accuracy(rows):
    return per_problem(((r['bench'], r['qid']), r['correct']) for r in rows)


def contrast(a, b):
    """(mean, 95% CI) of b - a over common problems, or None."""
    d = paired_dq(a, b)
    if len(d) < 10:
        return None
    m, c95, _, _, _ = pboot(d)
    return m, c95


def fmt(c, f='+.3f'):
    return 'pending' if c is None else f'{c[0]:{f}} [{c[1][0]:{f}}, {c[1][1]:{f}}]'


def verdict(c, sign):
    if c is None:
        return 'pending'
    lo, hi = c[1]
    if (sign > 0 and lo > 0) or (sign < 0 and hi < 0):
        return '**as predicted**'
    if (sign > 0 and hi < 0) or (sign < 0 and lo > 0):
        return '**opposite**'
    return 'not resolved'


class Results:
    def __init__(self, root, arms):
        self.root, self.arms = root, arms
        self.by_run = {r: records(root, r) for runs in arms.values() for r in runs}
        self.decisions = defaultdict(dict)
        for r in dec_rows(Data(root)):
            if r['run'] == 'base':
                self.decisions[r['model']][(r['bench'], r['qid'], r['sid'])] = r

    def rollouts(self, arm):
        return [r for run in self.arms[arm] for r in self.by_run[run]]

    def pooled(self, arm):
        return accuracy(self.rollouts(arm))

    def paper_protocol(self, arm):
        """The first available run of the arm, first 16 rollouts per problem."""
        run = next((r for r in self.arms[arm] if self.by_run[r]), None)
        return accuracy(x for x in self.by_run[run] if x['sid'] < PAPER_N) if run else {}

    def decision_value(self, arm, key):
        """{problem: mean over rollouts and over the arm's runs} of a decision metric on base's rollouts."""
        vals = [((k[0], k[1]), x[key]) for run in self.arms[arm] for k, x in self.decisions.get(run, {}).items()
                if x.get(key) is not None]
        return per_problem(vals)

    def mask_metrics(self, run):
        final = Path(self.root) / run / 'checkpoint' / 'trainer_state.json'     # runs without checkpoints keep it here
        history = json.loads(final.read_text())['log_history'] if final.is_file() else train_log(self.root, run)
        log = [h for h in history if 'mask/dropped_frac' in h]
        keys = ('dropped_frac', 'dropped_loss_share', 'decision_frac', 'decision_loss_share')
        return {k: mean(h[f'mask/{k}'] for h in log) for k in keys} if log else None


def section_accuracy(res, out):
    out.append('## Accuracy\n')
    out.append('`all` pools every eval rollout of the arm; `avg@16` is the paper protocol (first 16 rollouts per problem '
               'of the first run). Differences are paired over problems, using each pair\'s `all` rollouts.\n')
    out.append('| arm | runs with results | rollouts / problem | acc (all) | acc (avg@16) | mean thinking tokens | hit cap | '
               '− base | − its vanilla control |')
    out.append('|---|---|---|---|---|---|---|---|---|')
    base = res.pooled('base')
    for arm in res.arms:
        rows = res.rollouts(arm)
        if not rows:
            out.append(f'| {arm} | 0 | — | pending | | | | | |')
            continue
        p = accuracy(rows)
        found = sum(bool(res.by_run[r]) for r in res.arms[arm])
        out.append(f'| {arm} | {found} | {len(rows) / len(p):.0f} | {acc(p)} | {acc(res.paper_protocol(arm))} | '
                   f'{mean(r["think_tokens"] for r in rows):,.0f} | {mean(r["capped"] for r in rows):.1%} | '
                   f'{"—" if arm == "base" else fmt(contrast(base, p))} | '
                   f'{"—" if arm in ("base", "run-b") else fmt(contrast(res.pooled(control_of(arm)), p))} |')
    out.append('')


def section_decisions(res, out):
    out.append('## Learned decision policy on base\'s rollouts (same states for every model)\n')
    out.append('`shift` = model − base at the paragraph starts of base\'s rollouts. `of vanilla` = shift relative to the arm\'s '
               'vanilla control: 100% means the arm learned the full vanilla shift, 0% means none.\n')
    out.append('| arm | p(commit) | shift | of vanilla | p(Wait/Hmm/Alt.) | shift | of vanilla |')
    out.append('|---|---|---|---|---|---|---|')
    keys = ('para_p_fa', 'para_p_sc')
    base = {k: res.decision_value('base', k) for k in keys}
    van = {(a, k): contrast(base[k], res.decision_value(a, k)) for a in ('run-b', CONTROL) for k in keys}
    for arm in res.arms:
        cells = []
        for k in keys:
            v = res.decision_value(arm, k)
            c = contrast(base[k], v) if arm != 'base' else None
            ref = van[(control_of(arm), k)]
            rel = f'{c[0] / ref[0]:.0%}' if c and ref and ref[0] else '—'
            cells += [f'{mean(v.values()):.4f}' if v else 'pending', fmt(c, '+.4f') if arm != 'base' else '—', rel]
        out.append(f'| {arm} | ' + ' | '.join(cells) + ' |')
    out.append('')


def section_e6(res, out):
    out.append('## E6. Does removing the decision positions from the loss prevent the damage?\n')
    runs = {a: [r for r in res.arms[a] if not is_extra_eval(r)] for a in E6}
    stats = {r: res.mask_metrics(r) for a in E6 for r in runs[a]}
    out.append('**Check 1: training balance** (mean over training steps; the matched control must remove the same token count and '
               f'loss mass as the decision mask, within {BALANCE_TOLERANCE:.0%})\n')
    out.append('| run | tokens dropped | loss dropped | decision positions: tokens | decision positions: loss |')
    out.append('|---|---|---|---|---|')
    for r, s in stats.items():
        out.append(f'| {r} | pending | | | |' if s is None else
                   f'| {r} | {s["dropped_frac"]:.2%} | {s["dropped_loss_share"]:.2%} | {s["decision_frac"]:.2%} | '
                   f'{s["decision_loss_share"]:.2%} |')
    balanced = None
    pairs = list(zip(runs['e6-decision'], runs['e6-matched']))
    if pairs and all(stats[d] and stats[m] for d, m in pairs):
        balanced = all(abs(stats[m][k] / stats[d][k] - 1) <= BALANCE_TOLERANCE for d, m in pairs
                       for k in ('dropped_frac', 'dropped_loss_share'))
    out.append(f'\nBalanced: **{"pending" if balanced is None else "yes" if balanced else "NO"}**\n')

    out.append('**Check 2: did the mask change what was learned?** (decision mask − matched control, on base\'s rollouts)\n')
    out.append('| measure | difference | prediction | result |')
    out.append('|---|---|---|---|')
    results = []
    for key, label, sign in (('para_p_fa', 'p(commit)', -1), ('para_p_sc', 'p(Wait/Hmm/Alt.)', +1)):
        c = contrast(res.decision_value('e6-matched', key), res.decision_value('e6-decision', key))
        results.append(verdict(c, sign))
        out.append(f'| {label} | {fmt(c, "+.4f")} | {"< 0" if sign < 0 else "> 0"} | {results[-1]} |')
    worked = None if 'pending' in results else all(x == '**as predicted**' for x in results)
    out.append(f'\nMask changed the policy: **{"pending" if worked is None else "yes" if worked else "NO"}**\n')

    out.append('**Primary endpoint and secondary contrasts** (accuracy, all pooled rollouts)\n')
    out.append('| contrast | difference | prediction | result |')
    out.append('|---|---|---|---|')
    primary = contrast(res.pooled('e6-matched'), res.pooled('e6-decision'))
    out.append(f'| **decision − matched control (primary)** | {fmt(primary)} | > 0 | {verdict(primary, +1)} |')
    for text, a, b in (('decision − random control', 'e6-random', 'e6-decision'),
                       ('decision − vanilla (trained on the other server)', CONTROL, 'e6-decision'),
                       ('matched control − vanilla (other server)', CONTROL, 'e6-matched'),
                       ('random control − vanilla (other server)', CONTROL, 'e6-random'),
                       ('decision − base', 'base', 'e6-decision'),
                       ('vanilla 4x16 − Run B (replication check)', 'run-b', CONTROL)):
        out.append(f'| {text} | {fmt(contrast(res.pooled(a), res.pooled(b)))} | two-sided | |')
    readable = balanced and worked
    out.append(f'\nPrimary endpoint interpretable (both checks passed): **{"pending" if balanced is None or worked is None else "yes" if readable else "NO"}**\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root', default='runs')
    p.add_argument('--arms', help='JSON file {arm: [runs]} replacing the default mapping')
    a = p.parse_args()
    arms = dict(ARMS)
    if a.arms:
        with open(a.arms) as f:
            arms = json.load(f)
    missing = set(ARMS) - set(arms)
    if missing:
        raise ValueError(f'--arms must define every arm; missing {sorted(missing)}')
    res = Results(a.runs_root, arms)
    out = ['# E5 and E6 (Qwen3-1.7B)\n']
    section_accuracy(res, out)
    section_decisions(res, out)
    section_e6(res, out)
    print('\n'.join(out))


if __name__ == '__main__':
    main()
