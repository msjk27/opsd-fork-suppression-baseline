"""Summarize research.appd_probe output: per-condition medians and paired comparisons.

Comparisons (paired over traces, two-sided sign test):
  dense vs base        - the paper's protocol (dense trace capped at 1536, base at 3072)
  dense vs base_c1536  - same 1536-token window: the context effect alone
  sparse vs base       - answer-only context, same 3072-token window
Usage: python -m research.appd_summary RUN_DIR [RUN_DIR ...]
"""
import json
import sys
from math import comb
from statistics import median


def sign_p(down, up):
    n = down + up
    if n == 0:
        return 1.0
    k = max(down, up)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


CONDS = ('base', 'sparse', 'dense', 'base_c1536')
PAIRS = (('dense', 'base'), ('dense', 'base_c1536'), ('sparse', 'base'))
METRICS = ('appD_fork', 'appD_lock', 'support_fork', 'support_lock', 'main_fork', 'main_lock')

for run in sys.argv[1:]:
    recs = [json.loads(l) for l in open(f'{run}/traces.jsonl')]
    print(f'\n== {run}  traces={len(recs)}  median trace tokens={median(r["trace_tokens"] for r in recs)}')
    print(f'{"metric":13s}' + ''.join(f'{c:>12s}' for c in CONDS) + '   ' +
          '   '.join(f'{a} vs {b}: down/up p' for a, b in PAIRS))
    for m in METRICS:
        meds = ''.join(f'{100 * median(r[c][m] for r in recs):11.2f}%' for c in CONDS)
        tests = []
        for a, b in PAIRS:
            down = sum(r[a][m] < r[b][m] for r in recs)
            up = sum(r[a][m] > r[b][m] for r in recs)
            tests.append(f'{down:2d}/{up:2d} p={sign_p(down, up):.1g}')
        print(f'{m:13s}{meds}   ' + '   '.join(f'{t:>24s}' for t in tests))
