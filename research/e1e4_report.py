"""Report for the degradation-mechanism experiments (works on partial results).

The sections follow one causal chain, from where accuracy is lost to where the training signal causes it:
  E1-A' belief curves     runs/e1-belief/<run>/   accuracy of the answer held after t thinking tokens, and a
                                                  length-matched decomposition of the gap (forcing belief)
  E1-A  forced exit        runs/e1-exit/<run>/     budget curve with a real answer section (forcing exit)
  E1-C  decisions          runs/e1-dec/<m>-on-<r>/ commit / self-correct propensities of model m on the
                                                  identical states of run r's rollouts (self_entropy)
  E1-D  policy swap        runs/e1-swap/<r>-to-<m>/ run r's rollouts cut where they commit, continued by m
                                                  (forcing swap; m = r is the control)
  E2    training dynamics  trainer_state.json of both runs + E1-C on checkpoints
  E3    failure analysis   eval rollouts + belief probes
  E4    training signal    runs/e4-signal/<student>/ (teacher_signal)
Accuracy CIs: paired bootstrap over the 90 problems. Per-position CIs: bootstrap over prompts / problems.
Usage: python -m research.e1e4_report [--runs-root runs] > report.md
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict
from statistics import mean

import numpy as np

from research.compare_evals import bootstrap
from research.evalio import (CLOSE, COMMIT_WORDS, CONCLUSION_WORDS, CONTRAST_WORDS, SELF_CORRECTION_WORDS,
                             load_records, single_token_ids)

BASE, RUN_A, RUN_B = 'base', 'vanilla-3090-s42', 'vanilla-3090-paper-s42'
NAMES = {BASE: 'base', RUN_A: 'Run A', RUN_B: 'Run B', 'ck100': 'Run A @100', 'ck200': 'Run A @200',
         'ckB100': 'Run B @100', 'ckB200': 'Run B @200'}
EXIT_GRID = (2048, 4096, 6144, 8192, 12288, 16384, 20480, 24576, 32768)
BELIEF_SHOW = (2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 20480, 24576, 28672, 32768)
N_ROLLOUTS = 1440


def name(x):
    return NAMES.get(x, x)


def read_rows(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    return rows


def per_problem(items):
    """items: iterable of ((bench, qid), value) -> {problem: mean value}."""
    d = defaultdict(list)
    for q, v in items:
        d[q].append(float(v))
    return {q: mean(v) for q, v in d.items()}


def pp_rows(rows, key='correct'):
    return per_problem(((r['bench'], r['qid']), r[key]) for r in rows)


def diff(a, b):
    """b - a, paired bootstrap over common problems."""
    common = sorted(set(a) & set(b))
    if len(common) < 10:
        return 'n/a'
    m, lo, hi, _, _, _ = bootstrap({q: a[q] for q in common}, {q: b[q] for q in common}, 0, n=4000)
    return f'{m:+.3f} [{lo:+.3f}, {hi:+.3f}]'


def acc(pp):
    return f'{mean(pp.values()):.3f}' if pp else 'n/a'


def ci(x, groups, f, n=2000, w=None):
    """(Weighted) mean with a bootstrap CI over groups (prompts or problems)."""
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return '—'
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=np.float64)
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    k = inv.max() + 1
    num, den = np.bincount(inv, x * w, k), np.bincount(inv, w, k)
    idx = np.random.default_rng(0).integers(0, k, (n, k))
    b = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return f'{(x * w).sum() / w.sum():{f}} [{np.percentile(b, 2.5):{f}}, {np.percentile(b, 97.5):{f}}]'


class Data:
    def __init__(self, root, belief_key='correct_s'):
        self.root = root
        self.belief_key = belief_key   # correct_s: expected accuracy of 4 sampled answers; correct: greedy
        self._evals, self._belief = {}, None

    def evals(self, run):
        if run not in self._evals:
            try:
                self._evals[run] = load_records(self.root, run)
            except (FileNotFoundError, KeyError):
                self._evals[run] = None
        return self._evals[run]

    def belief(self):
        """{run: {rollout: {'end': correct, 'end_pos': int, ('grid', t): correct, ('pair:R', t): correct}}}"""
        if self._belief is None:
            b = defaultdict(lambda: defaultdict(dict))
            for r in read_rows(f'{self.root}/e1-belief/*/shard*.jsonl'):
                k = (r['bench'], r['qid'], r['sid'])
                if r['kind'] == 'end':
                    b[r['run']][k]['end'] = r[self.belief_key]
                    b[r['run']][k]['natural'] = r['natural_correct']
                    b[r['run']][k]['end_pos'] = r['end']
                elif r['probed']:
                    b[r['run']][k][(r['kind'], r['cut'])] = r[self.belief_key]
                else:
                    b[r['run']][k][(r['kind'], r['cut'])] = None   # past the end: end belief applies
            self._belief = b
        return self._belief

    def belief_at(self, run, kind, t):
        """Per rollout belief correctness at (kind, t); rollouts done reasoning by t keep their end belief."""
        out = {}
        for k, v in self.belief().get(run, {}).items():
            if 'end' not in v:
                continue
            if kind == 'grid' and t >= v['end_pos']:
                out[k] = v['end']
            elif (kind, t) in v:
                out[k] = v['end'] if v[(kind, t)] is None else v[(kind, t)]
        return out


def complete(belief_run):
    return sum('end' in v for v in belief_run.values()) == N_ROLLOUTS


def section_belief(d, out):
    out.append("## E1-A'. Belief curves: the answer each model holds after t thinking tokens\n")
    b = d.belief()
    runs = [r for r in (BASE, RUN_A, RUN_B) if r in b]
    if not runs:
        out.append('_pending_\n')
        return
    out.append('Probe: at t (moved to the nearest paragraph start), append "**Final Answer**\\n\\boxed{" and sample '
               f'the answer at the eval settings (accuracy = {"fraction of 4 samples correct" if d.belief_key == "correct_s" else "greedy answer correct"}). '
               'Rollouts that already finished reasoning keep the answer they ended with. Same probe for every '
               'model, so equal t means equal thinking.\n')
    out.append('**Probe validity**: the probe at the end of the reasoning should reproduce the rollout\'s own result.\n')
    out.append('| run | rollouts probed | natural acc | probe-at-end acc | agreement |')
    out.append('|---|---|---|---|---|')
    for r in runs:
        v = [x for x in b[r].values() if 'end' in x]
        out.append(f'| {name(r)} | {len(v)} | {mean(x["natural"] for x in v):.3f} | {mean(x["end"] for x in v):.3f} | '
                   f'{mean(1 - abs(x["natural"] - x["end"]) for x in v):.1%} |')
    out.append('\n| t | ' + ' | '.join(f'{name(r)} acc (still reasoning)' for r in runs) +
               ' | Run A − base | Run B − base |')
    out.append('|' + '---|' * (len(runs) + 3))
    for t in BELIEF_SHOW + (None,):
        cells, pp = [], {}
        for r in runs:
            if t is None:
                bt = {k: v['end'] for k, v in b[r].items() if 'end' in v}
                still = 0.0
            else:
                bt = d.belief_at(r, 'grid', t)
                still = mean(v['end_pos'] > t for v in b[r].values() if 'end' in v)
            pp[r] = per_problem(((k[0], k[1]), c) for k, c in bt.items())
            cells.append(f'{acc(pp[r])} ({still:.0%})' if complete(b[r]) else f'partial ({len(bt)})')
        dd = [diff(pp[BASE], pp[x]) if BASE in pp and x in pp else 'n/a' for x in (RUN_A, RUN_B)]
        out.append(f'| {"end" if t is None else f"{t:,}"} | ' + ' | '.join(cells) + ' | ' + ' | '.join(dd) + ' |')

    out.append('\n### Length-matched decomposition (all three numbers are probe beliefs)\n')
    out.append('base@len(M): base rollout k of problem q probed where rollout k of M on q ended its reasoning '
               '(base given exactly M\'s amount of thinking, rollout by rollout).\n')
    out.append('- **length part** = base − base@len(M): accuracy base gains by thinking past the point where M stops.\n'
               '- **quality part** = base@len(M) − M: the gap left at equal thinking.\n')
    out.append('| M | base | base@len(M) | M | gap | length part | quality part |')
    out.append('|---|---|---|---|---|---|---|')
    for m in (RUN_A, RUN_B):
        dec = belief_decomposition(d, m)
        if dec is None:
            continue
        p_base, p_at, p_m = dec
        out.append(f'| {name(m)} | {acc(p_base)} | {acc(p_at)} | {acc(p_m)} | {diff(p_m, p_base)} | '
                   f'{diff(p_at, p_base)} | {diff(p_m, p_at)} |')
    out.append('')


def belief_decomposition(d, m):
    """Per-problem probe beliefs (base at its end, base at M's end length, M at its end), or None until complete."""
    b = d.belief()
    if BASE not in b or m not in b or not complete(b[BASE]) or not complete(b[m]):
        return None
    at = {}
    for k, v in b[BASE].items():
        keys = [x for x in v if isinstance(x, tuple) and x[0] == f'pair:{m}']
        if keys and 'end' in v:
            c = v[keys[0]]
            at[k] = v['end'] if c is None else c
    if len(at) < N_ROLLOUTS:
        return None
    return (per_problem(((k[0], k[1]), v['end']) for k, v in b[BASE].items()),
            per_problem(((k[0], k[1]), c) for k, c in at.items()),
            per_problem(((k[0], k[1]), v['end']) for k, v in b[m].items()))


def pboot(dq, n=10000, seed=0):
    """Mean of per-problem paired differences with 95% and 99% (Bonferroni over the 5 directional tests)
    bootstrap CIs."""
    qs = sorted(dq)
    dv = np.array([dq[q] for q in qs])
    idx = np.random.default_rng(seed).integers(0, len(dv), (n, len(dv)))
    bm = dv[idx].mean(1)
    pct = lambda lv: (np.percentile(bm, (100 - lv) / 2), np.percentile(bm, 100 - (100 - lv) / 2))
    return dv.mean(), pct(95), pct(99), qs, idx


def verdict(c95, c99, sign):
    if sign is None:
        return 'estimate' if c95[0] > 0 or c95[1] < 0 else 'estimate (CI includes 0)'
    if (sign > 0 and c99[0] > 0) or (sign < 0 and c99[1] < 0):
        return '**supported** (survives Bonferroni)'
    if (sign > 0 and c95[0] > 0) or (sign < 0 and c95[1] < 0):
        return 'supported at 95% only'
    if (sign > 0 and c95[1] < 0) or (sign < 0 and c95[0] > 0):
        return '**contradicted**'
    return 'not resolved'


def paired_dq(a, b):
    """b - a per problem over the problems both have."""
    return {q: b[q] - a[q] for q in sorted(set(a) & set(b))}


def section_primary(d, out):
    """Pre-registered primary endpoints (runs/e1e4/preregistration.md, written before these results existed)."""
    out.append('## Pre-registered primary endpoints\n')
    out.append('Defined in `runs/e1e4/preregistration.md` before the swap, decision and belief-comparison results '
               'existed. Each endpoint is a difference of per-problem means with a paired bootstrap over problems. '
               'The directional family (P1-A, P1-A\', P1-B, P3a, P3b) must exclude 0 at 99% (Bonferroni over 5 '
               'tests). E1-C values are per-rollout means over positions, averaged per problem.\n')
    sw = defaultdict(dict)
    for r in read_rows(f'{d.root}/e1-swap/*/shard*.jsonl'):
        sw[(r['run'], r.get('cut_at', 'end'), r['label'])][(r['bench'], r['qid'], r['sid'])] = r
    arm_keys = {'A→A': (RUN_A, 'end', RUN_A), 'A→base': (RUN_A, 'end', BASE),
                'base@lenA→base': (BASE, f'pair:{RUN_A}', BASE), 'base@lenA→A': (BASE, f'pair:{RUN_A}', RUN_A),
                'B→B': (RUN_B, 'end', RUN_B), 'B→base': (RUN_B, 'end', BASE)}
    arm, nat, status = {}, {}, {}
    for nm, key in arm_keys.items():
        rs = sw.get(key, {})
        status[nm] = f'{len(rs)}/{N_ROLLOUTS}'
        if len(rs) == N_ROLLOUTS:
            arm[nm] = per_problem(((k[0], k[1]), x['correct']) for k, x in rs.items())
            nat[nm] = per_problem(((k[0], k[1]), x['natural_correct']) for k, x in rs.items())

    out.append('**E1-D arms** (accuracy; rollouts done / total)\n')
    out.append('| prefix ↓ / continued by → | base | own model |')
    out.append('|---|---|---|')
    cell = lambda nm: f'{acc(arm[nm])}' if nm in arm else f'pending ({status[nm]})'
    out.append(f'| Run A, cut at its last commit | A→base {cell("A→base")} | A→A {cell("A→A")} (control) |')
    out.append(f'| base, cut at Run A\'s stopping length | base@lenA→base {cell("base@lenA→base")} (control) | '
               f'base@lenA→A {cell("base@lenA→A")} |')
    out.append(f'| Run B, cut at its last commit | B→base {cell("B→base")} | B→B {cell("B→B")} (control) |\n')

    out.append('**Validity checks** (control arm − the same rollouts\' natural result; must include 0)\n')
    out.append('| control | natural acc | control acc | Δ [95% CI] | passes |')
    out.append('|---|---|---|---|---|')
    valid = {}
    for nm in ('A→A', 'B→B', 'base@lenA→base'):
        if nm not in arm:
            out.append(f'| {nm} | — | — | pending ({status[nm]}) | — |')
            continue
        m, c95, _, _, _ = pboot(paired_dq(nat[nm], arm[nm]))
        valid[nm] = c95[0] <= 0 <= c95[1]
        out.append(f'| {nm} | {acc(nat[nm])} | {acc(arm[nm])} | {m:+.3f} [{c95[0]:+.3f}, {c95[1]:+.3f}] | '
                   f'{"yes" if valid[nm] else "**NO**"} |')
    out.append('')

    rows = []   # (id, endpoint, prediction sign, per-problem differences or a pending note)

    def swap_ep(eid, text, a_nm, b_nm, sign):
        rows.append((eid, text, sign, paired_dq(arm[a_nm], arm[b_nm]) if a_nm in arm and b_nm in arm else
                     f'pending ({a_nm} {status[a_nm]}, {b_nm} {status[b_nm]})'))

    swap_ep('P1-A', 'acc(A→base) − acc(A→A): continuing from Run A\'s commit state with base\'s policy',
            'A→A', 'A→base', +1)
    swap_ep('P1-A\'', 'acc(base@lenA→base) − acc(base@lenA→A): the same policy contrast on base\'s own reasoning '
            'at Run A\'s stopping length (no selection on Run A\'s tokens)', 'base@lenA→A', 'base@lenA→base', +1)
    swap_ep('P1-B', 'acc(B→base) − acc(B→B): same as P1-A for Run B', 'B→B', 'B→base', +1)

    dec = defaultdict(dict)
    for r in dec_rows(d):
        dec[(r['model'], r['run'])][(r['bench'], r['qid'], r['sid'])] = r

    def dec_ep(eid, text, run, m1, m0, key, sign):   # model m1 - model m0 on run's rollouts
        x, y = dec.get((m0, run), {}), dec.get((m1, run), {})
        n_need = 4 * 90
        if len(x) < n_need or len(y) < n_need:
            rows.append((eid, text, sign, f'pending ({name(m0)} {len(x)}/{n_need}, {name(m1)} {len(y)}/{n_need})'))
            return
        per = [((k[0], k[1]), y[k][key] - x[k][key]) for k in x
               if k in y and x[k].get(key) is not None and y[k].get(key) is not None]
        rows.append((eid, text + f' ({len(per)} rollouts)', sign, per_problem(per)))

    dec_ep('P3a', 'p(commit \'**\') at paragraph starts of base\'s rollouts, Run A − base', BASE, RUN_A, BASE,
           'para_p_fa', +1)
    dec_ep('P3b', 'p(Wait/Hmm/Alt.) at paragraph starts of base\'s rollouts, Run A − base', BASE, RUN_A, BASE,
           'para_p_sc', -1)
    swap_ep('P2', 'acc(base@lenA→base) − acc(A→base): damage carried in Run A\'s reasoning before the stop',
            'A→base', 'base@lenA→base', None)
    swap_ep('P2\'', 'acc(base@lenA→A) − acc(A→A): the same prefix effect under Run A\'s continuation',
            'A→A', 'base@lenA→A', None)
    bd = belief_decomposition(d, RUN_A)
    rows.append(('P4', 'base@len(A) − A (belief probes; biased in Run A\'s favour)', None,
                 paired_dq(bd[2], bd[1]) if bd else 'pending (belief probes)'))

    out.append('| id | endpoint | problems | estimate | 95% CI | 99% CI | prediction | verdict |')
    out.append('|---|---|---|---|---|---|---|---|')
    res = {}
    for eid, text, sign, dq in rows:
        pred = {1: '> 0', -1: '< 0', None: 'two-sided'}[sign]
        if isinstance(dq, str):
            out.append(f'| {eid} | {text} | — | — | — | — | {pred} | {dq} |')
            continue
        f = '.4f' if eid.startswith('P3') else '+.3f'
        m, c95, c99, qs, idx = pboot(dq)
        res[eid] = (dq, qs, idx)
        out.append(f'| {eid} | {text} | {len(dq)} | {m:{f}} | [{c95[0]:{f}}, {c95[1]:{f}}] | '
                   f'[{c99[0]:{f}}, {c99[1]:{f}}] | {pred} | {verdict(c95, c99, sign)} |')
    for p1n, p2n in (('P1-A', 'P2'), ('P1-A\'', 'P2\'')):
        if p1n not in res or p2n not in res:
            continue
        p1, p2 = res[p1n][0], res[p2n][0]
        qs = sorted(set(p1) & set(p2))
        v1, v2 = np.array([p1[q] for q in qs]), np.array([p2[q] for q in qs])
        idx = np.random.default_rng(1).integers(0, len(qs), (10000, len(qs)))
        b1, bg = v1[idx].mean(1), (v1 + v2)[idx].mean(1)
        g = (v1 + v2).mean()
        lo, hi = np.percentile(bg, 2.5), np.percentile(bg, 97.5)
        if lo > 0:
            sh = b1 / bg
            out.append(f'\n**Share of the gap explained by the policy from the stopping point on** = {p1n} / ({p1n} + {p2n}) = '
                       f'{v1.mean() / g:.0%} [{np.percentile(sh, 2.5):.0%}, {np.percentile(sh, 97.5):.0%}] '
                       f'(2×2 gap base@lenA→base − A→A = {g:+.3f} [{lo:+.3f}, {hi:+.3f}])')
        else:
            out.append(f'\nShare of the gap ({p1n}): not computed, the 2×2 gap {g:+.3f} [{lo:+.3f}, {hi:+.3f}] includes 0.')
    if valid and not all(valid.values()):
        out.append('\n**Warning: a validity check failed; P1 and P2 are not interpretable until it is explained.**')
    out.append('')


def section_exit(d, out):
    out.append('## E1-A. Budget curve with a real answer section (forced exit)\n')
    rows = list({(r['run'], r['bench'], r['qid'], r['sid'], r['kind'], r['cut']): r
                 for r in read_rows(f'{d.root}/e1-exit/*/shard*.jsonl')}.values())
    if not rows:
        out.append('_pending_\n')
        return
    out.append('At t the thinking is closed with Qwen3\'s thinking-budget text and the model writes its answer '
               '(≤ 4,096 tokens, which can include fresh non-thinking work). This is the practical "what if the '
               'budget were t" curve; for the mechanism, the belief curves above are the cleaner measure.\n')
    runs = [r for r in (BASE, RUN_A, RUN_B) if any(x['run'] == r for x in rows)]
    grid = defaultdict(list)
    for r in rows:
        if r['kind'] == 'grid':
            grid[(r['run'], r['cut'])].append(r)
    out.append('| t | ' + ' | '.join(name(r) for r in runs) + ' | Run A − base | Run B − base |')
    out.append('|' + '---|' * (len(runs) + 3))
    for t in EXIT_GRID:
        pp, cells = {}, []
        for r in runs:
            rs = grid.get((r, t), [])
            if len(rs) == N_ROLLOUTS:
                pp[r] = pp_rows(rs)
                cells.append(acc(pp[r]))
            else:
                cells.append(f'partial ({len(rs)})' if rs else 'pending')
        dd = [diff(pp[BASE], pp[x]) if BASE in pp and x in pp else 'n/a' for x in (RUN_A, RUN_B)]
        out.append(f'| {t:,} | ' + ' | '.join(cells) + ' | ' + ' | '.join(dd) + ' |')
    out.append('')


def dec_rows(d):
    rows = read_rows(f'{d.root}/e1-dec/*/shard*.jsonl')
    return list({(r['model'], r['run'], r['bench'], r['qid'], r['sid']): r for r in rows}.values())


def section_decisions(d, out):
    out.append('## E1-C. Decisions on identical states\n')
    rows = dec_rows(d)
    if not rows:
        out.append('_pending_\n')
        return
    out.append('Every model scores the same fixed rollouts (4 per problem). Paragraph starts are where the model '
               'chooses between going on, self-correcting (Wait / Hmm / Alternatively) or committing to an answer '
               '("**Final Answer**"). A commit point is a paragraph start where the rollout actually committed. '
               'Commit-point columns are selected on the token the states\' own model sampled there, which inflates '
               'that model\'s probability (regression to the mean): compare models on the paragraph-start columns, '
               'or read the commit-point columns on both state sets.\n')
    by = defaultdict(list)
    for r in rows:
        by[(r['model'], r['run'])].append(r)
    states = [s for s in (BASE, RUN_A, RUN_B) if any(k[1] == s for k in by)]
    models = [m for m in (BASE, 'ck100', 'ck200', RUN_A, 'ckB100', 'ckB200', RUN_B) if any(k[0] == m for k in by)]
    val = lambda rs, key: [x[key] for x in rs if x.get(key) is not None]
    for s in states:
        out.append(f'**States: {name(s)} rollouts**\n')
        out.append('| scoring model | rollouts | p(Wait/Hmm/Alt.) @para | p(commit) @para | p(commit) @first commit | '
                   'p(Wait/Hmm/Alt.) @first commit | p(commit) @last commit | fork (App. D) | lock |')
        out.append('|---|---|---|---|---|---|---|---|---|')
        for m in models:
            rs = by.get((m, s))
            if not rs:
                continue
            g = lambda key: ci(val(rs, key), [f"{x['bench']}|{x['qid']}" for x in rs if x.get(key) is not None], '.4f')
            out.append(f'| {name(m)} | {len(rs)} | {g("para_p_sc")} | {g("para_p_fa")} | {g("commit_p_fa_first")} | '
                       f'{g("commit_p_sc_first")} | {g("commit_p_fa_last")} | {g("fork")} | {g("lock")} |')
        out.append('')
        out.append(f'Per tenth of the thinking ({name(s)} states): p(commit) @para / p(Wait/Hmm/Alt.) @para\n')
        out.append('| scoring model | ' + ' | '.join(f'{i}0–{i + 1}0%' for i in range(10)) + ' |')
        out.append('|' + '---|' * 11)
        for m in models:
            rs = by.get((m, s))
            if not rs:
                continue
            cells = []
            for i in range(10):
                fa = [x['para_fa_dec'][i] for x in rs if x['para_fa_dec'][i] is not None]
                sc = [x['para_sc_dec'][i] for x in rs if x['para_sc_dec'][i] is not None]
                cells.append(f'{mean(fa):.3f} / {mean(sc):.3f}' if fa and sc else '—')
            out.append(f'| {name(m)} | ' + ' | '.join(cells) + ' |')
        out.append('')


def section_swap(d, out):
    out.append('## E1-D. Policy swap at the commit point\n')
    rows = read_rows(f'{d.root}/e1-swap/*/shard*.jsonl')
    if not rows:
        out.append('_pending_\n')
        return
    out.append('Each rollout of run r is cut where its reasoning ended (its last "**Final Answer**" commit), or at '
               'the length where the paired rollout of another run ended ("cut at"), and continued from that exact '
               'state by model m with the eval sampling. m = r is the control. Δ vs control is paired by rollout '
               'and bootstrapped over problems. "wrong → right" compares with the rollout\'s own final result.\n')
    arms = defaultdict(dict)
    for r in rows:
        arms[(r['run'], r.get('cut_at', 'end'), r['label'])][(r['bench'], r['qid'], r['sid'])] = r
    out.append('| states | cut at | continued by | rollouts | acc | Δ vs control | commits at once | extra thinking '
               '(mean) | Wait/Hmm/Alt. added | wrong → right | right → wrong |')
    out.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for (run, cut_at, lab), rs in sorted(arms.items()):
        ctrl = arms.get((run, cut_at, run))
        v = list(rs.values())
        p = per_problem(((k[0], k[1]), x['correct']) for k, x in rs.items())
        dv = 'control' if lab == run else (
            diff(per_problem(((k[0], k[1]), x['correct']) for k, x in ctrl.items() if k in rs),
                 per_problem(((k[0], k[1]), x['correct']) for k, x in rs.items() if k in ctrl)) if ctrl else 'n/a')
        fix = mean((not x['natural_correct']) and x['correct'] for x in v)
        brk = mean(x['natural_correct'] and not x['correct'] for x in v)
        out.append(f'| {name(run)} | {"own end" if cut_at == "end" else "end of " + name(cut_at[5:])} | {name(lab)} | {len(v)} | {acc(p)} | {dv} | {mean(x["commits_first"] for x in v):.0%} | '
                   f'{mean(x["added_think"] for x in v):,.0f} | {mean(x["n_sc_added"] for x in v):.2f} | '
                   f'{fix:.1%} | {brk:.1%} |')
    out.append('')


def train_log(root, run):
    f = sorted(glob.glob(f'{root}/{run}/checkpoint/checkpoint-*/trainer_state.json'),
               key=lambda p: int(re.search(r'checkpoint-(\d+)', p).group(1)))
    return json.load(open(f[-1]))['log_history'] if f else []


def section_training(d, out):
    out.append('## E2. Training dynamics (training rollouts, 4,096 cap; mean over 20-step bins)\n')
    for run in (RUN_A, RUN_B):
        log = [h for h in train_log(d.root, run) if 'completions/mean_length' in h]
        if not log:
            continue
        out.append(f'**{name(run)}**\n')
        out.append('| steps | rollout length | hit cap | student entropy | loss (JSD) |')
        out.append('|---|---|---|---|---|')
        for lo in range(0, len(log), 20):
            b = log[lo:lo + 20]
            out.append(f'| {b[0]["step"]}–{b[-1]["step"]} | {mean(h["completions/mean_length"] for h in b):,.0f} | '
                       f'{mean(h["completions/clipped_ratio"] for h in b):.0%} | {mean(h["entropy"] for h in b):.3f} | '
                       f'{mean(h["loss"] for h in b):.4f} |')
        out.append('')


def section_failures(d, out, tok):
    out.append('## E3. Failure analysis\n')
    base, runs = d.evals(BASE), [r for r in (RUN_A, RUN_B) if d.evals(r)]
    if not base or not runs:
        out.append('_pending_\n')
        return
    pb = pp_rows(base)
    for m in runs:
        pm = pp_rows(d.evals(m))
        out.append(f'- **{name(m)}**: {sum(pb[q] - pm[q] >= .25 for q in pb)} problems lose ≥ 4/16 correct rollouts, '
                   f'{sum(pm[q] - pb[q] >= .25 for q in pb)} gain ≥ 4/16.')
    sc_ids = set(single_token_ids(tok, SELF_CORRECTION_WORDS))
    close_id = tok.convert_tokens_to_ids(CLOSE)
    bins = [(i * 2048, (i + 1) * 2048) for i in range(12)]
    out.append('\n### Self-correction markers by absolute position in the thinking\n')
    out.append('Wait / Hmm / Alternatively per 1k thinking tokens, over rollouts still thinking there (n). Equal rates '
               'at equal positions would mean OPSD reasons like base, only stopping sooner.\n')
    out.append('| tokens | ' + ' | '.join(f'{name(r)} (n)' for r in [BASE] + runs) + ' |')
    out.append('|' + '---|' * (len(runs) + 2))
    dens = {}
    for r in [BASE] + runs:
        cnt, tot, n = np.zeros(len(bins)), np.zeros(len(bins)), np.zeros(len(bins))
        for x in d.evals(r):
            ids = tok(x['response'], add_special_tokens=False).input_ids
            think = ids.index(close_id) if close_id in ids else len(ids)
            pos = np.array([i for i, t in enumerate(ids[:think]) if t in sc_ids])
            for k, (lo, hi) in enumerate(bins):
                if think > lo:
                    tot[k] += min(think, hi) - lo
                    n[k] += 1
                    cnt[k] += ((pos >= lo) & (pos < hi)).sum() if len(pos) else 0
        dens[r] = (cnt, tot, n)
    for k, (lo, hi) in enumerate(bins):
        out.append(f'| {lo // 1024}k–{hi // 1024}k | ' + ' | '.join(
            f'{1000 * dens[r][0][k] / dens[r][1][k]:.2f} ({int(dens[r][2][k])})' if dens[r][1][k] else '—'
            for r in [BASE] + runs) + ' |')
    b = d.belief()
    if BASE in b and complete(b[BASE]):
        out.append('\n### What base\'s thinking beyond the OPSD length does (belief probes)\n')
        out.append('Base rollouts that think longer than the paired OPSD rollout: belief accuracy at the OPSD length vs '
                   'at their own end (expected accuracy of the sampled answers). Gains and losses are averaged '
                   'per rollout as max(end − at, 0) and max(at − end, 0).\n')
        out.append('| M | base rollouts longer than M\'s | gained later | lost later | net |')
        out.append('|---|---|---|---|---|')
        for m in runs:
            changes = []
            for k, v in b[BASE].items():
                keys = [x for x in v if isinstance(x, tuple) and x[0] == f'pair:{m}' and v[x] is not None]
                if keys and 'end' in v:
                    changes.append((v[keys[0]], v['end']))
            if changes:
                res = mean(max(float(e) - float(a), 0.0) for a, e in changes)
                harm = mean(max(float(a) - float(e), 0.0) for a, e in changes)
                out.append(f'| {name(m)} | {len(changes)} | {res:.1%} | {harm:.1%} | {res - harm:+.1%} |')
    out.append('')


def e4_outcomes(root, st, tok, df):
    """Per rollout outcome of the E4 training rollouts: correct / wrong (closed) or truncated (cap)."""
    from evaluation.grader import math_equal
    from evaluation.utils import extract_answer_math
    close_id = tok.convert_tokens_to_ids(CLOSE)
    out = {}
    for l in open(f'{root}/e4-signal/{st}/rollouts.jsonl'):
        r = json.loads(l)
        if close_id not in r['ids']:
            out[(r['row'], r['sid'])] = 'truncated'
            continue
        text = tok.decode(r['ids'], skip_special_tokens=True)
        ok = math_equal(extract_answer_math(text), str(df.answer_only[r['row']]), timeout=True)
        out[(r['row'], r['sid'])] = 'correct' if ok else 'wrong'
    return out


def section_signal(d, out, tok):
    out.append('## E4. Anatomy of the training signal\n')
    students = sorted(os.path.basename(p) for p in glob.glob(f'{d.root}/e4-signal/*') if
                      glob.glob(f'{p}/signal_shard*.npz'))
    if not students:
        out.append('_pending_\n')
        return
    import pandas as pd
    df = pd.read_parquet('data/raw/openthoughts_math_filtered_15k_ctx.parquet', columns=['answer_only'])
    close_id = tok.convert_tokens_to_ids(CLOSE)
    sets = {'close': {close_id}, 'self-corr': set(single_token_ids(tok, SELF_CORRECTION_WORDS)),
            'contrast': set(single_token_ids(tok, CONTRAST_WORDS)),
            'conclusion': set(single_token_ids(tok, CONCLUSION_WORDS)), 'commit': set(single_token_ids(tok, COMMIT_WORDS))}
    vocab_text = [tok.decode([i]) for i in range(len(tok))]
    para_tok = np.array([t.endswith('\n\n') for t in vocab_text])
    math_tok = np.array([bool(re.search(r'[0-9=+\-*/^_{}\\()\[\]<>|]', t)) for t in vocab_text])
    out.append('On-policy rollouts on 128 training prompts × 2 (training sampling, 4,096 cap); the teacher is base '
               'with the context in the training template. Δ = teacher − student: Δp(Wait) < 0 means the teacher '
               'wants fewer self-corrections, Δp(commit) > 0 that it wants an earlier "**Final Answer**". JSD is the '
               'per-token training loss. **All averages use the trainer\'s weighting** (each rollout\'s loss is a '
               'per-token mean, so a position weighs 1/length of its rollout). CIs: bootstrap over prompts. '
               'Validity check: the step-0 per-rollout JSD with the solution context reproduces the training loss '
               'logged at steps 1–5 (see text).\n')
    for st in students:
        parts = [dict(np.load(f)) for f in sorted(glob.glob(f'{d.root}/e4-signal/{st}/signal_shard*.npz'))]
        z = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k not in ('contexts', 'skip')}
        contexts = [str(c) for c in parts[0]['contexts']]
        f32 = lambda k: z[k].astype(np.float32)
        tokid, rows, pos = z['token'], z['row'], z['pos']
        valid = pos >= int(parts[0]['skip'])
        rkey = rows.astype(np.int64) * 4 + z['sid']
        _, rinv = np.unique(rkey, return_inverse=True)
        nvalid = np.bincount(rinv[valid], minlength=rinv.max() + 1)
        w = np.where(valid, 1.0 / np.maximum(nvalid[rinv], 1), 0.0)          # trainer weighting
        wm = lambda x, m: float((x[m] * w[m]).sum() / w[m].sum()) if m.any() else float('nan')
        prev = np.where(pos > 0, np.roll(tokid, 1), -1)
        para = (prev >= 0) & para_tok[np.clip(prev, 0, None)]
        start = np.flatnonzero(pos == 0)
        thinking = np.ones(len(tokid), bool)
        for s0, s1 in zip(start, np.r_[start[1:], len(tokid)]):
            c = np.flatnonzero(tokid[s0:s1] == close_id)
            if len(c):
                thinking[s0 + c[0] + 1:s1] = False
        outcome = e4_outcomes(d.root, st, tok, df)
        oc = np.array([outcome.get((r_, s_), 'truncated') for r_, s_ in zip(rows, z['sid'])])
        cat = np.full(len(tokid), 'text', dtype=object)
        cat[math_tok[tokid]] = 'math'
        cat[para] = 'para-start other'
        for k in ('conclusion', 'contrast', 'commit', 'self-corr', 'close'):
            cat[np.isin(tokid, list(sets[k]))] = k
        cats = ['close', 'self-corr', 'commit', 'contrast', 'conclusion', 'para-start other', 'math', 'text']
        n_out = {o: len({k for k, v in outcome.items() if v == o}) for o in ('correct', 'wrong', 'truncated')}
        out.append(f'### Student: {name(st)} ({len(np.unique(rows))} prompts; rollouts {n_out}; '
                   f'{valid.sum():,} trained positions, {(valid & ~thinking).mean():.1%} after the thinking)\n')
        if st == BASE and 'solution' in contexts:
            jj = f32('solution_jsd')
            per = np.bincount(rinv[valid], jj[valid], rinv.max() + 1) / np.maximum(nvalid, 1)
            out.append(f'Step-0 training loss reproduced: mean per-rollout JSD (solution context) = {per.mean():.4f}.\n')
        out.append('**Where the loss is (share of the weighted JSD) and how the teacher re-weights the sampled token '
                   '(weighted mean log-ratio teacher − student)**\n')
        out.append('| token category | share of positions (weighted) | ' + ' | '.join(f'{c}: JSD share / log-ratio' for c in contexts) + ' |')
        out.append('|' + '---|' * (len(contexts) + 2))
        for c in cats + ['(after the thinking)']:
            m = valid & ~thinking if c.startswith('(') else valid & thinking & (cat == c)
            cells = []
            for ctx in contexts:
                jw = z[f'{ctx}_jsd'] * w
                lr = f32(f'{ctx}_lp') - f32('s_lp')
                cells.append(f'{jw[m].sum() / jw[valid].sum():.1%} / {wm(lr, m):+.3f}' if m.any() else '—')
            out.append(f'| {c} | {w[m].sum() / w[valid].sum():.2%} | ' + ' | '.join(cells) + ' |')
        m = valid & thinking & para
        base_mass = {k: wm(f32(f's_{k}'), m) for k in ('p_sc', 'p_ct', 'p_cc', 'p_fa', 'p_close')}
        out.append(f'\n**Paragraph starts in the thinking** ({m.sum():,} positions). Student: '
                   f'p(Wait/Hmm/Alt.) {base_mass["p_sc"]:.4f}, p(But/However/Actually) {base_mass["p_ct"]:.4f}, '
                   f'p(Therefore/Thus) {base_mass["p_cc"]:.4f}, p(commit) {base_mass["p_fa"]:.4f}, '
                   f'top-3 entropy {wm(f32("s_h3"), m):.3f}.\n')
        out.append('| context | mean JSD | Δp(Wait/Hmm/Alt.) | Δp(But/However/Actually) | Δp(Therefore/Thus) | Δp(commit) |')
        out.append('|---|---|---|---|---|---|')
        for ctx in contexts:
            cells = [ci(z[f'{ctx}_jsd'][m], rows[m], '.4f', w=w[m])]
            for k in ('p_sc', 'p_ct', 'p_cc', 'p_fa'):
                cells.append(ci(f32(f'{ctx}_{k}')[m] - f32(f's_{k}')[m], rows[m], '+.4f', w=w[m]))
            out.append(f'| {ctx} | ' + ' | '.join(cells) + ' |')
        out.append('\n**Specific shift or general sharpening?** Relative change of each option\'s mass at paragraph '
                   'starts, and the change in top-3 entropy there. Sharpening alone would shrink every minority option '
                   'by a similar fraction and could not raise one.\n')
        out.append('| context | Wait/Hmm/Alt. | But/However/Actually | Therefore/Thus | commit | close | Δ top-3 entropy |')
        out.append('|---|---|---|---|---|---|---|')
        for ctx in contexts:
            rel = [f'{wm(f32(f"{ctx}_{k}"), m) / base_mass[k] - 1:+.0%}' if base_mass[k] > 1e-6 else '—'
                   for k in ('p_sc', 'p_ct', 'p_cc', 'p_fa', 'p_close')]
            out.append(f'| {ctx} | ' + ' | '.join(rel) + f' | {ci(f32(f"{ctx}_h3")[m] - f32("s_h3")[m], rows[m], "+.3f", w=w[m])} |')
        out.append('\n**Is the push conditional on the student being right?** Paragraph starts split by how the '
                   'rollout ended (correct / wrong final answer, or cut at 4,096 before answering): '
                   'Δp(Wait/Hmm/Alt.) / Δp(commit)\n')
        out.append('| context | correct | wrong | truncated |')
        out.append('|---|---|---|---|')
        for ctx in contexts:
            cells = []
            for o in ('correct', 'wrong', 'truncated'):
                mo = m & (oc == o)
                cells.append(f'{wm(f32(f"{ctx}_p_sc") - f32("s_p_sc"), mo):+.4f} / '
                             f'{wm(f32(f"{ctx}_p_fa") - f32("s_p_fa"), mo):+.4f} (n={mo.sum()})' if mo.any() else '—')
            out.append(f'| {ctx} | ' + ' | '.join(cells) + ' |')
        out.append('\n**Fork / lock rates on the same trajectories (App. D rule, all thinking positions)**: the '
                   'paper\'s comparison. Unlike the table below, no position is selected by either model.\n')
        mt = valid & thinking
        sf, sl = (f32('s_h3') >= .6).astype(float), (f32('s_h3') <= .2).astype(float)
        out.append(f'Student: fork {wm(sf, mt):.2%}, lock {wm(sl, mt):.2%}.\n')
        out.append('| context | teacher fork | teacher lock | fork, teacher − student | lock, teacher − student |')
        out.append('|---|---|---|---|---|')
        for ctx in contexts:
            tf, tl = (f32(f'{ctx}_h3') >= .6).astype(float), (f32(f'{ctx}_h3') <= .2).astype(float)
            out.append(f'| {ctx} | {wm(tf, mt):.2%} | {wm(tl, mt):.2%} | '
                       f'{ci((tf - sf)[mt], rows[mt], "+.4f", w=w[mt])} | {ci((tl - sl)[mt], rows[mt], "+.4f", w=w[mt])} |')
        out.append('\n**Fork positions in the thinking (student top-3 entropy ≥ .60)**. Selecting on the student\'s '
                   'entropy biases the teacher − student difference downward (regression to the mean); with a trained '
                   'student, read each context against the "none" row.\n')
        out.append('| context | share of positions | share of JSD | teacher h3 − student h3 | teacher locks (h3 ≤ .20) | '
                   'of which at paragraph starts |')
        out.append('|---|---|---|---|---|---|')
        f = valid & thinking & (f32('s_h3') >= .6)
        for ctx in contexts:
            th = f32(f'{ctx}_h3')
            jw = z[f'{ctx}_jsd'] * w
            out.append(f'| {ctx} | {w[f].sum() / w[valid].sum():.1%} | {jw[f].sum() / jw[valid].sum():.1%} | '
                       f'{ci(th[f] - f32("s_h3")[f], rows[f], "+.3f", w=w[f])} | {wm((th <= .2).astype(float), f):.1%} | '
                       f'{(f & para).sum() / max(f.sum(), 1):.1%} |')
        out.append('\n**By position in the rollout (tenths), paragraph starts: Δp(Wait/Hmm/Alt.) / Δp(commit)**\n')
        out.append('| context | ' + ' | '.join(f'{i}0–{i + 1}0%' for i in range(10)) + ' |')
        out.append('|' + '---|' * 11)
        dec = np.minimum((10 * pos / z['len']).astype(int), 9)
        for ctx in contexts:
            cells = []
            for i in range(10):
                mm = m & (dec == i)
                cells.append(f'{wm(f32(f"{ctx}_p_sc") - f32("s_p_sc"), mm):+.3f} / '
                             f'{wm(f32(f"{ctx}_p_fa") - f32("s_p_fa"), mm):+.3f}' if mm.any() else '—')
            out.append(f'| {ctx} | ' + ' | '.join(cells) + ' |')
        out.append('')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root', default='runs')
    p.add_argument('--model', default='models/Qwen3-1.7B', help='tokenizer')
    p.add_argument('--belief', default='correct_s', choices=['correct_s', 'correct'],
                   help='belief measure: sampled answers (default) or greedy')
    p.add_argument('--sections', nargs='*', default=['primary', 'belief', 'exit', 'decisions', 'swap', 'training',
                                                     'failures', 'signal'])
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    d = Data(a.runs_root, a.belief)
    out = ['# Why does ON/ON OPSD degrade thinking-mode accuracy? (Qwen3-1.7B, E1–E4)\n']
    secs = dict(primary=lambda: section_primary(d, out),
                belief=lambda: section_belief(d, out), exit=lambda: section_exit(d, out),
                decisions=lambda: section_decisions(d, out), swap=lambda: section_swap(d, out),
                training=lambda: section_training(d, out), failures=lambda: section_failures(d, out, tok),
                signal=lambda: section_signal(d, out, tok))
    for s in a.sections:
        secs[s]()
    print('\n'.join(out))


if __name__ == '__main__':
    main()
