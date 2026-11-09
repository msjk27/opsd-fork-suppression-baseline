"""Fork/lock diagnostic following the paper's Appendix D protocol as closely as its text allows.

Per prompt: one student trace (base student prompt). The teacher (same checkpoint, i.e. the OPSD side)
scores it under three contexts:
  base   - problem only;                             student trace capped at --cap (3072)
  sparse - problem + correct final answer;           student trace capped at --cap (3072)
  dense  - problem + reference trace from a stronger model truncated to --ref-cap (2048) tokens;
           student trace capped at --dense-cap (1536)
Control: base_c1536 = base context with the student trace capped at --dense-cap, which separates the
context effect from the trace-length asymmetry.

Per position (from the teacher's full softmax):
  appD    - top-K (K=3) probabilities renormalized, entropy / log K; lock <= .20, fork >= .60
  support - support-aware variant (our reading of App. D): top-p .8 over the renormalized top-K;
            fork if >= 2 retained tokens are within a factor of 3 of the top token,
            lock if a single token is retained
  main    - §4.1 rule on the full softmax top-1 probability: fork < .25, lock > .65
Rates are per trace (fraction of scored positions). Unstated in the paper: student sampling settings,
which 60 prompts, which reference model.

Traces either come from an earlier probe's positions.jsonl (--traces-from, condition 'base') or are
sampled here (--temperature/--top-p).
Usage: python -m research.appd_probe --model M --data P --out DIR [--start S --limit N]
         [--traces-from RUN_DIR | --temperature T --top-p P --seed 42]
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

TEMPLATE = ("{prompt}\n\nThis is an example for a response to the question:\n{gold_answer}\n\n"
            "Now answer with a response of your own, including the thinking process.")


def rates(probs, k=3):
    """Per-position fork/lock flags under the three definitions, as rates over positions."""
    import torch
    top1 = probs.max(-1).values
    q = probs.topk(k, -1).values
    q = q / q.sum(-1, keepdim=True)
    h = -(q * q.clamp_min(1e-12).log()).sum(-1) / math.log(k)
    # support-aware: smallest prefix of the sorted top-K whose mass reaches .8 is retained
    keep = (q.cumsum(-1) - q) < 0.8
    competitive = (keep & (q >= q[:, :1] / 3)).sum(-1)
    flags = dict(appD_fork=h >= .60, appD_lock=h <= .20,
                 support_fork=competitive >= 2, support_lock=keep.sum(-1) == 1,
                 main_fork=top1 < .25, main_lock=top1 > .65)
    return {name: f.float().mean().item() for name, f in flags.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--limit', type=int, default=60)
    p.add_argument('--traces-from')
    p.add_argument('--temperature', type=float)
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cap', type=int, default=3072)
    p.add_argument('--dense-cap', type=int, default=1536)
    p.add_argument('--ref-cap', type=int, default=2048)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if (a.traces_from is None) == (a.temperature is None):
        raise SystemExit('give exactly one of --traces-from or --temperature')

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 attn_implementation='sdpa').to(a.device).eval()
    rows = load_dataset('parquet', data_files=a.data, split='train')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'args.json').write_text(json.dumps(vars(a), indent=2))

    traces = {}
    if a.traces_from:
        pos = defaultdict(list)
        with open(Path(a.traces_from) / 'positions.jsonl') as f:
            for line in f:
                r = json.loads(line)
                if r['condition'] == 'base' and a.start <= r['question_id'] < a.start + a.limit:
                    pos[r['question_id']].append((r['position'], r['token_id']))
        traces = {q: [t for _, t in sorted(v)][:a.cap] for q, v in pos.items()}

    def prompt(text):
        return tok.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True,
                                       add_generation_prompt=True, enable_thinking=True, return_tensors='pt')

    def score(text, completion):
        ids = prompt(text)
        seq = torch.cat([ids, completion], 1).to(a.device)
        with torch.inference_mode():
            logits = model(seq, use_cache=False, logits_to_keep=completion.shape[1] + 1).logits[0, :-1]
            return rates(logits.float().softmax(-1))

    for qid in range(a.start, min(a.start + a.limit, len(rows))):
        row = rows[qid]
        if qid not in traces:
            if a.traces_from:
                continue
            ids = prompt(row['problem']).to(a.device)
            with torch.inference_mode():
                gen = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=a.cap,
                                     do_sample=True, temperature=a.temperature, top_p=a.top_p, top_k=0,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            traces[qid] = gen[0, ids.shape[1]:].tolist()
        trace = torch.tensor([traces[qid]])
        ref = tok.decode(tok(row['solution']).input_ids[:a.ref_cap])
        rec = dict(question_id=qid, trace_tokens=trace.shape[1], token_ids=traces[qid],
                   base=score(row['problem'], trace),
                   sparse=score(TEMPLATE.format(prompt=row['problem'], gold_answer=row['Answer']), trace),
                   dense=score(TEMPLATE.format(prompt=row['problem'], gold_answer=ref), trace[:, :a.dense_cap]),
                   base_c1536=score(row['problem'], trace[:, :a.dense_cap]))
        with (out / 'traces.jsonl').open('a') as f:
            f.write(json.dumps(rec) + '\n')
        print(qid, {c: round(rec[c]['appD_fork'], 4) for c in ('base', 'sparse', 'dense', 'base_c1536')}, flush=True)


if __name__ == '__main__':
    main()
