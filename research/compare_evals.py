"""Compare avg@16 evals of several models on the same benchmarks, paired by problem.

Reads <run>/eval/<bench>_38912/*.jsonl (graded records with is_correct, response_tokens, cap_hit,
self_correction_proxy_per_1k_tokens). Reports per-benchmark accuracy, behaviour (length, cap hits,
self-correction proxy), and a paired bootstrap over problems for each model vs the first one and for
each consecutive pair.
Usage: python -m research.compare_evals --runs-root DIR --runs base vanilla-a vanilla-b [--seed 0]
"""
import argparse
import glob
import json
import random
from collections import defaultdict
from statistics import mean, median

BENCHES = ('aime24', 'aime25', 'hmmt25')
PAPER = {'aime24': (0.502, 0.435), 'aime25': (0.398, 0.302), 'hmmt25': (0.215, 0.185)}


def load(root, run):
    acc, beh = {}, {}
    for b in BENCHES:
        f = [p for p in glob.glob(f'{root}/{run}/eval/{b}_38912/*.jsonl')][0]
        per_q, rows = defaultdict(list), []
        for line in open(f):
            r = json.loads(line)
            per_q[(b, str(r['question_id']))].append(r['is_correct'] in (True, 'True', 1))
            rows.append(r)
        acc.update({q: mean(v) for q, v in per_q.items()})
        tok = [int(r['response_tokens']) for r in rows]
        beh[b] = dict(n=len(rows), mean_len=mean(tok), median_len=median(tok),
                      cap_hit=mean(r['cap_hit'] in (True, 'True') for r in rows),
                      selfcorr=mean(float(r['self_correction_proxy_per_1k_tokens']) for r in rows))
    return acc, beh


def bootstrap(a, b, seed, n=10000):
    qs = sorted(a)
    d = [b[q] - a[q] for q in qs]
    rng = random.Random(seed)
    boots = sorted(mean(rng.choice(d) for _ in qs) for _ in range(n))
    worse = sum(x < 0 for x in d); better = sum(x > 0 for x in d)
    return mean(d), boots[int(.025 * n)], boots[int(.975 * n)], sum(x >= 0 for x in boots) / n, worse, better


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root', required=True)
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    data = {r: load(a.runs_root, r) for r in a.runs}
    print('bench   ' + ''.join(f'{r:>26s}' for r in a.runs) + '   paper base -> OPSD')
    for b in BENCHES + ('average',):
        cells = []
        for r in a.runs:
            acc = data[r][0]
            cells.append(mean(v for (bb, _), v in acc.items() if b in (bb, 'average')))
        paper = (f'{PAPER[b][0]:.3f} -> {PAPER[b][1]:.3f}' if b in PAPER else
                 f'{mean(v[0] for v in PAPER.values()):.3f} -> {mean(v[1] for v in PAPER.values()):.3f}')
        print(f'{b:8s}' + ''.join(f'{c:26.3f}' for c in cells) + f'   {paper}')
    pairs = [(a.runs[0], r) for r in a.runs[1:]] + list(zip(a.runs[1:], a.runs[2:]))
    print('\npaired bootstrap over problems (B - A): mean, 95% CI, P(diff>=0), problems worse/better')
    for x, y in dict.fromkeys(pairs):
        m, lo, hi, pge, w, bt = bootstrap(data[x][0], data[y][0], a.seed)
        print(f'  {y} - {x}: {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]  P={pge:.4f}  worse/better {w}/{bt}')
    print('\nbench   model' + ' ' * 22 + 'mean_len  median  cap_hit  selfcorr/1k')
    for b in BENCHES:
        for r in a.runs:
            s = data[r][1][b]
            print(f'{b:8s}{r:27s}{s["mean_len"]:8.0f}{s["median_len"]:8.0f}{s["cap_hit"]:9.3f}{s["selfcorr"]:12.2f}')


if __name__ == '__main__':
    main()
