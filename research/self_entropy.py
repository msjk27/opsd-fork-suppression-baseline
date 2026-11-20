"""Decision propensities and fork/lock rates of a model on fixed eval rollouts (E1-C, E2).

--model scores the rollouts of --run (its own, or another model's: identical states, different policy),
with temperature-1 logits as in the App. D probe:
  h3   top-3 probabilities renormalized, entropy / ln 3 (App. D):  fork h3 >= .60,  lock h3 <= .20
  top1 full-softmax top-1 probability (§4.1 rule):                 fork < .25,      lock > .65
Rates are over thinking positions (response tokens before the closing tag): the whole thinking, the
first 4,096 thinking tokens (length control), and per tenth of the thinking. At paragraph starts
(previous token ends in a blank line) it also records the probability of closing the thinking and of
the self-correction markers and of opening '**Final Answer**', the decisions that set how long the model
keeps thinking: their mean over paragraph starts, per tenth of the thinking, and at the rollout's own
commit points (the paragraph starts where it actually opened '**Final Answer**').

The forward pass runs in chunks with a KV cache, so 38k-token rollouts fit on a 24 GB GPU.
Usage: python -m research.self_entropy --model M --label NAME --run RUN --out DIR [--samples 4]
"""
import argparse
import json
import math
from pathlib import Path

import torch

from research.evalio import (CLOSE, COMMIT_WORDS, SELF_CORRECTION_WORDS, commit_points, load_records, paragraph_ids,
                             single_token_ids, split_thinking)


@torch.inference_mode()
def position_stats(model, ids, n_prompt, chunk=1024, head_chunk=1024):
    """Per response position: h3, top1, p(close), p(self-correction set). ids = prompt + response.
    Chunks stay small: attention over a 38k-token cache costs heads x chunk x 38k, and the output
    head costs chunk x vocab, so 1k-token chunks keep a 38k rollout under ~12 GB."""
    from transformers import DynamicCache
    cache, hs = DynamicCache(), []
    for i in range(0, len(ids), chunk):
        x = torch.tensor([ids[i:i + chunk]], device=model.device)
        out = model.model(input_ids=x, past_key_values=cache, use_cache=True)
        hs.append(out.last_hidden_state[0])
    h = torch.cat(hs)[n_prompt - 1:len(ids) - 1]   # positions predicting each response token
    res = {k: [] for k in ('h3', 'top1', 'p_close', 'p_sc', 'p_fa')}
    for j in range(0, h.shape[0], head_chunk):
        p = model.lm_head(h[j:j + head_chunk]).float().softmax(-1)
        q = p.topk(3, -1).values
        q = q / q.sum(-1, keepdim=True)
        res['h3'].append(-(q * q.clamp_min(1e-12).log()).sum(-1) / math.log(3))
        res['top1'].append(p.max(-1).values)
        res['p_close'].append(p[:, model.close_id].clone())   # a view would keep all of p alive
        res['p_sc'].append(p[:, model.sc_ids].sum(-1))
        res['p_fa'].append(p[:, model.fa_ids].sum(-1))
    return {k: torch.cat(v).cpu() for k, v in res.items()}


def summarize(st, ids_resp, think, para_ids, star_id=None, final_id=None):
    h3, top1 = st['h3'][:think], st['top1'][:think]
    rate = lambda m: m.float().mean().item() if m.numel() else None
    deciles = [((i * think) // 10, ((i + 1) * think) // 10) for i in range(10)]
    para = torch.tensor([j > 0 and ids_resp[j - 1] in para_ids for j in range(think)], dtype=torch.bool)
    dec = torch.clamp(torch.arange(think) * 10 // max(think, 1), max=9)
    by_dec = lambda x: [rate(x[:think][para & (dec == i)]) if (para & (dec == i)).any() else None for i in range(10)]
    cps = commit_points(ids_resp, think, para_ids, star_id, final_id) if star_id is not None else []
    at = lambda k, j: st[k][j].item() if j is not None else None
    first, last = (cps[0], cps[-1]) if cps else (None, None)
    return dict(
        n_commit=len(cps), first_commit=first, last_commit=last,
        commit_p_fa_first=at('p_fa', first), commit_p_sc_first=at('p_sc', first),
        commit_p_fa_last=at('p_fa', last), commit_p_sc_last=at('p_sc', last),
        para_sc_dec=by_dec(st['p_sc']), para_fa_dec=by_dec(st['p_fa']),
        think=think,
        fork=rate(h3 >= .6), lock=rate(h3 <= .2), fork_main=rate(top1 < .25), lock_main=rate(top1 > .65),
        fork_4k=rate(h3[:4096] >= .6), lock_4k=rate(h3[:4096] <= .2),
        fork_dec=[rate(h3[a:b] >= .6) for a, b in deciles],
        n_para=int(para.sum()),
        para_p_close=rate(st['p_close'][:think][para]) if para.any() else None,
        para_p_sc=rate(st['p_sc'][:think][para]) if para.any() else None,
        para_p_fa=rate(st['p_fa'][:think][para]) if para.any() else None,
        para_h3=rate(h3[para]) if para.any() else None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True)
    p.add_argument('--label', required=True, help='name of the scoring model')
    p.add_argument('--run', required=True, help='eval run whose rollouts are scored')
    p.add_argument('--out', required=True)
    p.add_argument('--runs-root', default='runs')
    p.add_argument('--samples', type=int, default=4, help='rollouts per problem (sample ids 0..n-1)')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    a = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation='sdpa')
    model = model.to('cuda').eval()
    model.close_id = tok.convert_tokens_to_ids(CLOSE)
    model.sc_ids = single_token_ids(tok, SELF_CORRECTION_WORDS)
    model.fa_ids = single_token_ids(tok, COMMIT_WORDS)
    para_ids = paragraph_ids(tok)
    star_id, final_id = single_token_ids(tok, ('**', 'Final'))
    recs = [r for r in load_records(a.runs_root, a.run) if r['sid'] < a.samples][a.shard::a.num_shards]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'shard{a.shard}.jsonl'
    done = set()
    if path.exists():
        done = {(r['bench'], r['qid'], r['sid']) for r in map(json.loads, open(path))}
    with open(path, 'a') as f:
        for n, r in enumerate(recs):
            if (r['bench'], r['qid'], r['sid']) in done:
                continue
            pid = tok(r['prompt'], add_special_tokens=False).input_ids
            rid = tok(r['response'], add_special_tokens=False).input_ids
            st = position_stats(model, pid + rid, len(pid))
            row = dict(bench=r['bench'], qid=r['qid'], sid=r['sid'], run=a.run, model=a.label, correct=r['correct'],
                       **summarize(st, rid, split_thinking(rid, model.close_id), para_ids, star_id, final_id))
            f.write(json.dumps(row) + '\n')
            f.flush()
            if n % 20 == 0:
                print(f'{a.run} shard {a.shard}: {n + 1}/{len(recs)}', flush=True)


if __name__ == '__main__':
    main()
