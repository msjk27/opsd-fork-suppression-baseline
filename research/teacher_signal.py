"""E4: what the OPSD training signal pushes, token by token, under different teacher contexts.

gen    On-policy rollouts from --student on OpenThoughts training prompts with the training rollout
       settings (thinking on, T 1.0, top-p 1.0, no top-k, 4,096-token cap).
score  For every rollout position, compare the student (problem only) with the frozen teacher
       (--teacher, problem + context in the training template) and store:
         jsd        per-token generalized JSD, alpha 0.5: the exact training loss (opsd.chunked_loss)
         lp_s/lp_t  log-prob of the sampled token under student / teacher (log-ratio = lp_t - lp_s)
         h3, top1   top-3 normalized entropy and top-1 probability (fork/lock rules)
         p_close    probability of closing the thinking
         p_sc       mass on self-correction markers (Wait / Hmm / Alternatively)
         p_ct       mass on contrast markers (But / However / Actually)
         p_cc       mass on conclusion markers (Therefore / Thus / Hence)
         p_fa       probability of '**' (at a paragraph start: opening '**Final Answer**')
       Contexts (columns of the context-variant parquet, see research.build_context_variants):
         none       problem only (the teacher sees what the student sees)
         answer     final answer only
         solution   the gold solution: what OPSD trains with
         shuffled   the solution of a different problem (format without the right content)
         cot        the R1 thinking trace for the same problem (right content, thinking style)
Output: <out>/rollouts.jsonl (gen) and <out>/signal_shard<i>.npz (score; float16 arrays over positions).
Usage:
  python -m research.teacher_signal gen   --student M --data P --out DIR [--prompts 128 --samples 2]
  python -m research.teacher_signal score --student M --teacher T --data P --out DIR [--contexts ...]
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from research.evalio import (CLOSE, COMMIT_WORDS, CONCLUSION_WORDS, CONTRAST_WORDS, SELF_CORRECTION_WORDS,
                             single_token_ids)

TEMPLATE = ("{prompt}\n\nThis is an example for a response to the question:\n{gold_answer}\n\n"
            "Now answer with a response of your own, including the thinking process.")  # trainer default
CONTEXT_COLUMNS = {'answer': 'answer_only', 'solution': 'solution', 'shuffled': 'solution_shuffled',
                   'cot': 'COT_Reason'}
SKIP = 3  # num_loss_tokens_to_skip in training


def chat(tok, text):
    return tok.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)


def pick_rows(n_rows, n, seed):
    return sorted(random.Random(seed).sample(range(n_rows), n))


def run_gen(a):
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    df = pd.read_parquet(a.data)
    tok = AutoTokenizer.from_pretrained(a.student)
    rows = pick_rows(len(df), a.prompts, a.seed)
    llm = LLM(model=a.student, dtype='bfloat16', gpu_memory_utilization=0.85, seed=a.seed)
    params = SamplingParams(n=a.samples, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=4096, seed=a.seed)
    outs = llm.generate([chat(tok, df.problem[i]) for i in rows], sampling_params=params)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'rollouts.jsonl', 'w') as f:
        for i, o in zip(rows, outs):
            for k, c in enumerate(o.outputs):
                f.write(json.dumps(dict(row=i, sid=k, ids=list(c.token_ids), finish=c.finish_reason)) + '\n')


@torch.inference_mode()
def completion_logits(model, prompt_ids, comp_ids):
    seq = torch.tensor([prompt_ids + comp_ids], device=model.device)
    return model(seq, use_cache=False, logits_to_keep=len(comp_ids) + 1).logits[0, :-1]


@torch.inference_mode()
def stats(logits, comp, sets, chunk=1024):
    out = {k: [] for k in ('lp', 'h3', 'top1', 'p_close', 'p_sc', 'p_ct', 'p_cc', 'p_fa')}
    for j in range(0, logits.shape[0], chunk):
        lp = logits[j:j + chunk].float().log_softmax(-1)
        p = lp.exp()
        out['lp'].append(lp.gather(-1, comp[j:j + chunk, None])[:, 0])
        q = p.topk(3, -1).values
        q = q / q.sum(-1, keepdim=True)
        out['h3'].append(-(q * q.clamp_min(1e-12).log()).sum(-1) / math.log(3))
        out['top1'].append(p.max(-1).values)
        for name, ids in sets.items():
            out[name].append(p[:, ids].sum(-1))
    return {k: torch.cat(v).cpu().numpy().astype(np.float16) for k, v in out.items()}


@torch.inference_mode()
def jsd(s_logits, t_logits, chunk=1024):
    from opsd.chunked_loss import _jsd_chunk
    return torch.cat([_jsd_chunk(s_logits[j:j + chunk].float(), t_logits[j:j + chunk].float(), 0.5, 1.0, 1.0)
                      for j in range(0, s_logits.shape[0], chunk)]).cpu().numpy().astype(np.float32)


def run_score(a):
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    df = pd.read_parquet(a.data)
    tok = AutoTokenizer.from_pretrained(a.student)
    load = lambda m: AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16,
                                                          attn_implementation='sdpa').to('cuda').eval()
    student = load(a.student)
    teacher = student if a.teacher == a.student else load(a.teacher)
    sets = {'p_close': [tok.convert_tokens_to_ids(CLOSE)], 'p_sc': single_token_ids(tok, SELF_CORRECTION_WORDS),
            'p_ct': single_token_ids(tok, CONTRAST_WORDS), 'p_cc': single_token_ids(tok, CONCLUSION_WORDS),
            'p_fa': single_token_ids(tok, COMMIT_WORDS)}
    rolls = [json.loads(l) for l in open(Path(a.out) / 'rollouts.jsonl')][a.shard::a.num_shards]
    acc = {}

    def add(name, arr):
        acc.setdefault(name, []).append(arr)

    for n, r in enumerate(rolls):
        problem = df.problem[r['row']]
        comp = r['ids']
        comp_t = torch.tensor(comp, device='cuda')
        s_logits = completion_logits(student, tok(chat(tok, problem), add_special_tokens=False).input_ids, comp)
        L = len(comp)
        add('row', np.full(L, r['row'], np.int32))
        add('sid', np.full(L, r['sid'], np.int8))
        add('pos', np.arange(L, dtype=np.int32))
        add('len', np.full(L, L, np.int32))
        add('token', np.array(comp, np.int32))
        for k, v in stats(s_logits, comp_t, sets).items():
            add(f's_{k}', v)
        for ctx in a.contexts:
            text = problem if ctx == 'none' else TEMPLATE.format(prompt=problem,
                                                                 gold_answer=df[CONTEXT_COLUMNS[ctx]][r['row']])
            t_logits = completion_logits(teacher, tok(chat(tok, text), add_special_tokens=False).input_ids, comp)
            add(f'{ctx}_jsd', jsd(s_logits, t_logits))
            for k, v in stats(t_logits, comp_t, sets).items():
                add(f'{ctx}_{k}', v)
            del t_logits
        del s_logits
        if n % 10 == 0:
            print(f'shard {a.shard}: {n + 1}/{len(rolls)} rollouts', flush=True)
    np.savez_compressed(Path(a.out) / f'signal_shard{a.shard}.npz', contexts=np.array(a.contexts),
                        skip=SKIP, **{k: np.concatenate(v) for k, v in acc.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='mode', required=True)
    g = sub.add_parser('gen')
    g.add_argument('--prompts', type=int, default=128)
    g.add_argument('--samples', type=int, default=2)
    g.add_argument('--seed', type=int, default=0)
    s = sub.add_parser('score')
    s.add_argument('--teacher', required=True)
    s.add_argument('--contexts', nargs='+', default=['answer', 'solution', 'shuffled', 'cot'])
    s.add_argument('--shard', type=int, default=0)
    s.add_argument('--num-shards', type=int, default=1)
    for x in (g, s):
        x.add_argument('--student', required=True)
        x.add_argument('--data', required=True)
        x.add_argument('--out', required=True)
    a = p.parse_args()
    (run_gen if a.mode == 'gen' else run_score)(a)


if __name__ == '__main__':
    main()
