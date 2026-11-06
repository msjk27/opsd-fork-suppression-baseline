"""Rescore saved probe trajectories with the paper's Appendix D fork/lock heuristic.

Appendix D: renormalize the teacher's top-K (K=3) probabilities, take their entropy
normalized by log K; lock if <= 0.20, fork if >= 0.60. This differs from the §4.1 main-text
rule (full-softmax top-1 prob < .25 / > .65), which research.metrics implements.

Reads the sampled token IDs from a probe's positions.jsonl (condition 'base'), rebuilds
the same student/teacher prompts as research.probe, and reports both definitions per
teacher condition (rollout macro averages).
Usage: python -m research.rescore_topk --config PROBE_CONFIG_JSON --output OUT_JSON
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--k', type=int, default=3)
    a = p.parse_args()
    c = json.loads(Path(a.config).read_text())
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from research.probe import sparse_answer

    tok = AutoTokenizer.from_pretrained(c['teacher'])
    teacher = AutoModelForCausalLM.from_pretrained(c['teacher'], dtype=torch.bfloat16,
                                                   attn_implementation='sdpa').to(c['teacher_device']).eval()
    rows = load_dataset('parquet', data_files=c['data'], split='train')

    traj = defaultdict(list)  # (qid, sample) -> sampled token ids in order
    with open(Path(c['output_dir']) / 'positions.jsonl') as f:
        for line in f:
            r = json.loads(line)
            if r['condition'] == 'base':
                traj[(r['question_id'], r['sample_id'])].append((r['position'], r['token_id']))

    def prompt(text):
        return tok.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True,
                                       add_generation_prompt=True, enable_thinking=True, return_tensors='pt')

    out = defaultdict(lambda: defaultdict(list))
    for (qid, sid), pos in sorted(traj.items()):
        completion = torch.tensor([[t for _, t in sorted(pos)]])
        row = rows[qid]
        sparse = sparse_answer(row, c.get('sparse_answer_source', 'answer'))
        for cond, gold in [('base', None), ('sparse', sparse), ('dense', row['solution'])]:
            text = row['problem'] if gold is None else c['teacher_prompt_template'].format(
                prompt=row['problem'], gold_answer=gold)
            ids = prompt(text)
            seq = torch.cat([ids, completion], 1).to(c['teacher_device'])
            with torch.inference_mode():
                logits = teacher(seq, use_cache=False, logits_to_keep=completion.shape[1] + 1).logits[0, :-1].float()
                probs = logits.softmax(-1)
                top1 = probs.max(-1).values
                topk = probs.topk(a.k, -1).values
                q = topk / topk.sum(-1, keepdim=True)
                h = -(q * q.clamp_min(1e-12).log()).sum(-1) / math.log(a.k)
            o = out[cond]
            o['main_fork'].append((top1 < .25).float().mean().item())
            o['main_lock'].append((top1 > .65).float().mean().item())
            o['appD_fork'].append((h >= .60).float().mean().item())
            o['appD_lock'].append((h <= .20).float().mean().item())

    summary = {cond: {k: sum(v) / len(v) for k, v in o.items()} | {'rollouts': len(o['main_fork'])}
               for cond, o in out.items()}
    per_rollout = {cond: dict(o) for cond, o in out.items()}
    Path(a.output).write_text(json.dumps({'k': a.k, 'summary': summary, 'per_rollout': per_rollout}, indent=1))
    for cond, s in summary.items():
        print(cond, {k: round(v, 4) for k, v in s.items()})


if __name__ == '__main__':
    main()
