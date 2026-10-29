"""Aggregate probe/training positions and graded upstream generation files."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from research.metrics import accuracy_metrics


def aggregate(rows):
    rollouts, positions = defaultdict(list), defaultdict(list)
    for r in rows:
        condition = r.get('condition', 'dense')
        key = (condition, r.get('step'), r.get('call_id'), r.get('question_id'),
               r.get('sample_id'), r.get('rank'), r['batch_row'])
        rollouts[key].append(r)
        positions[(condition, r['position'])].append(r)
    def stats(rs):
        keys = ('fork', 'lock', 'argmax_disagree', 'log_ratio', 'jsd', 'teacher_entropy', 'student_entropy')
        return dict(count=len(rs), **{k: sum(r[k] for r in rs)/len(rs) for k in keys})
    per_rollout = [dict(condition=k[0], rollout=list(k[1:]), **stats(rs)) for k, rs in rollouts.items()]
    conditions = sorted({r['condition'] for r in per_rollout})
    return {'rollouts': per_rollout,
            'macro_rollout_rates': {c: {m: sum(r[m] for r in per_rollout if r['condition']==c)/sum(r['condition']==c for r in per_rollout)
                                        for m in ('fork', 'lock')} for c in conditions},
            'positions': [dict(condition=c, position=p, **stats(rs)) for (c, p), rs in sorted(positions.items())]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--positions', nargs='+')
    p.add_argument('--graded')
    p.add_argument('--responses')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    def read(path):
        with open(path, encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]
    result = {}
    if a.positions:
        result['diagnostics'] = aggregate([r for path in a.positions for r in read(path)])
    if a.graded:
        result['accuracy'] = accuracy_metrics(read(a.graded))
    if a.responses:
        rs = read(a.responses)
        if not rs:
            raise ValueError('No responses')
        result['responses'] = {k: sum(r[k] for r in rs)/len(rs) for k in
                               ('response_tokens', 'reasoning_tokens', 'eos', 'cap_hit',
                                'self_correction_proxy_count', 'self_correction_proxy_per_1k_tokens')}
    if not result:
        p.error('At least one input is required')
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
