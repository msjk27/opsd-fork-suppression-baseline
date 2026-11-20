"""E1: interventions on thinking length, to separate "thinks too little" from "thinks worse".

belief    (E1-A', belief probe; the main length analysis). At a cut t inside an existing rollout, moved to
          the nearest paragraph start (either direction, so no bias in the amount of text kept), append
          '**Final Answer**\\n\\boxed{' and decode only the answer: what the model would answer after t
          thinking tokens, with no extra computation. Every model ends its thinking this way (>= 99.6% of
          rollouts), so the probe at the end of the reasoning reproduces the rollout's own answer.
          Answers are read twice: 4 samples at the eval sampling (T 0.6, top-p 0.95; correct_s = fraction
          correct, the expectation avg@16 also measures) and greedily (correct, the modal answer).
            grid    every --step tokens before the end of the reasoning
            end     at the end of the reasoning (last '**Final Answer**' commit inside the thinking)
            pair:R  at the end-of-reasoning position of rollout k of run R on the same problem
swap      (E1-D, policy swap). Cut each rollout of --run where its reasoning ends (its final commit) and let
          --model continue from that exact state with the eval sampling. Continuing with the rollout's own
          model is the control; continuing with another model asks whether that model would have stopped
          there too, and whether its continuation changes the answer. With --cut-pair R the cut is instead
          where rollout k of R on the same problem ended its reasoning (snapped to a paragraph start), e.g.
          base's states at the length where OPSD stops.
exit      (E1-A, forced early exit). Re-uses the existing eval rollouts of --run. For every rollout and
          every cut t shorter than its natural thinking, keep the first t response tokens, append Qwen3's
          documented thinking-budget stop text and let the model write its final answer. This gives the
          accuracy the model would have had if it had stopped thinking after t tokens.
            grid cuts      fixed t values (the thinking curve).
            paired cuts    'pair:<R>' stops rollout k of problem q at the natural thinking length of rollout
                           k of run R on the same problem, i.e. --run made to think exactly as long as R.
          A cut at or beyond the natural thinking length keeps the rollout's own graded result.
continue  (E1-B, forced continuation, s1-style). Fresh rollouts from --model. Whenever the model closes
          its thinking before the problem's target L(q), the closing tag is replaced by "Wait" and
          generation continues (at most --max-forced times). L(q) = median natural thinking length of
          --target-run on that problem. Run the same targets on several models for a fair comparison.

Sampling follows the eval protocol: T 0.6, top-p 0.95, 38,912-token cap on the response.
Each shard appends one JSON line per result to <out>/shard<i>.jsonl and skips results already there.

Usage:
  python -m research.forcing exit --model M --run RUN --out DIR [--grid ...] [--pair-runs R ...]
  python -m research.forcing continue --model M --label NAME --target-run base --out DIR
Common: --runs-root runs --shard i --num-shards n
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from research.evalio import (CAP, CLOSE, SELF_CORRECTION_WORDS, TEMPERATURE, TOP_P, end_of_reasoning, grade,
                             load_records, paragraph_ids, single_token_ids, snap_to_paragraph, split_thinking)

# Qwen3's thinking-budget early-stop text (Qwen3 model card), without its stray '.' after the tag.
EXIT_TEXT = ("\n\nConsidering the limited time by the user, I have to give the solution based on the "
             "thinking directly now.\n</think>\n\n")
WAIT_TEXT = "\n\nWait"
PROBE_TEXT = "**Final Answer**\n\\boxed{"   # how the model itself states its answer at the end of thinking
ANSWER_CAP = 4096   # natural answer sections: p99 ~1.9k, max 3.3k tokens
GRID = (2048, 4096, 6144, 8192, 12288, 16384, 20480, 24576, 32768)


def seed_of(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


def exit_cuts(grid, paired):
    """(kind, cut) pairs for one rollout. paired maps reference run -> its rollout's thinking length."""
    return [('grid', t) for t in grid] + [(f'pair:{r}', t) for r, t in paired.items()]


def wait_splice(tok, gen, tail=8):
    """Replace a closed thinking ending with WAIT_TEXT, re-tokenizing the last few tokens so the
    boundary (e.g. '.\n' + '\n\nWait') tokenizes the way the model would have written it."""
    text = tok.decode(gen[-tail:], skip_special_tokens=False).rstrip()
    return gen[:-tail] + tok(text + WAIT_TEXT, add_special_tokens=False).input_ids


def done_keys(path, fields):
    keys = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                keys.add(tuple(r[k] for k in fields))
    return keys


def make_llm(model):
    from vllm import LLM
    return LLM(model=model, dtype='bfloat16', gpu_memory_utilization=0.9, trust_remote_code=True,
               enable_prefix_caching=True)


def tokenized(tok, recs):
    """Per rollout key: (response ids, thinking length, end of reasoning)."""
    close_id = tok.convert_tokens_to_ids(CLOSE)
    para = paragraph_ids(tok)
    star_id, final_id = single_token_ids(tok, ('**', 'Final'))
    out = {}
    for r in recs:
        ids = tok(r['response'], add_special_tokens=False).input_ids
        think = split_thinking(ids, close_id)
        out[(r['bench'], r['qid'], r['sid'])] = (ids, think, end_of_reasoning(ids, think, para, star_id, final_id))
    return out


def belief_cuts(end, step, paired):
    """(kind, cut) probes for one rollout: its end of reasoning, a grid before it, and paired lengths."""
    return [('end', end)] + [('grid', t) for t in range(step, end, step)] + [(f'pair:{r}', t) for r, t in paired.items()]


def run_belief(a):
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    tok = AutoTokenizer.from_pretrained(a.model)
    para = paragraph_ids(tok)
    probe_ids = tok(PROBE_TEXT, add_special_tokens=False).input_ids
    blank = tok('\n\n', add_special_tokens=False).input_ids
    recs = load_records(a.runs_root, a.run)[a.shard::a.num_shards]
    toks = tokenized(tok, recs)
    ref_ends = {ref: {k: v[2] for k, v in tokenized(tok, load_records(a.runs_root, ref)).items()}
                for ref in a.pair_runs}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'shard{a.shard}.jsonl'
    fields = ('bench', 'qid', 'sid', 'kind', 'cut')
    done = done_keys(path, fields)
    carried, groups = [], []
    for r in recs:
        key = (r['bench'], r['qid'], r['sid'])
        ids, think, end = toks[key]
        prompt_ids = tok(r['prompt'], add_special_tokens=False).input_ids
        jobs = []
        for kind, cut in belief_cuts(end, a.step, {ref: e[key] for ref, e in ref_ends.items()}):
            row = dict(bench=r['bench'], qid=r['qid'], sid=r['sid'], run=a.run, kind=kind, cut=cut, end=end,
                       think=think, natural_correct=r['correct'])
            if tuple(row[k] for k in fields) in done:
                continue
            used = end if kind == 'end' else snap_to_paragraph(ids, cut, para, nearest=True, upper=end)
            if kind != 'end' and used >= end:   # already done reasoning: the end-of-reasoning belief applies
                carried.append(dict(row, probed=False))
                continue
            sep = [] if ids[used - 1] in para else blank
            jobs.append((dict(row, cut_used=used), r, prompt_ids + ids[:used] + sep + probe_ids))
        if jobs:
            groups.append(sorted(jobs, key=lambda j: -j[0]['cut_used']))   # longest prefix first
    with open(path, 'a') as f:
        for row in carried:
            f.write(json.dumps(row) + '\n')
    n_jobs = sum(map(len, groups))
    print(f'shard {a.shard}: {len(recs)} rollouts, {len(carried)} carried, {n_jobs} probes', flush=True)
    if not n_jobs:
        return
    llm = make_llm(a.model)
    greedy = SamplingParams(temperature=0.0, max_tokens=48, stop=['\n'])
    sampled = lambda row: SamplingParams(n=a.answer_samples, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=48,
                                         stop=['\n'], seed=seed_of(*(row[k] for k in fields)))
    done_n = 0
    # A few rollouts at a time: their longest probe fills the prefix cache, the shorter probes then reuse it.
    for i in range(0, len(groups), a.group):
        block = groups[i:i + a.group]
        heads = [g[0] for g in block]
        rest = [j for g in block for j in g[1:]]
        with open(path, 'a') as f:
            for chunk in (heads, rest):
                if not chunk:
                    continue
                prompts = [TokensPrompt(prompt_token_ids=p) for _, _, p in chunk]
                outs_g = llm.generate(prompts, sampling_params=greedy, use_tqdm=False)
                outs_s = llm.generate(prompts, sampling_params=[sampled(row) for row, _, _ in chunk], use_tqdm=False)
                for (row, r, _), og, os_ in zip(chunk, outs_g, outs_s):
                    pred = og.outputs[0].text
                    preds = [c.text for c in os_.outputs]
                    f.write(json.dumps(dict(row, probed=True, pred=pred, correct=grade(r['bench'], PROBE_TEXT + pred, r),
                                            preds_s=preds, correct_s=mean(grade(r['bench'], PROBE_TEXT + x, r)
                                                                          for x in preds))) + '\n')
        done_n += len(heads) + len(rest)
        if (i // a.group) % 25 == 0:
            print(f'shard {a.shard}: {done_n}/{n_jobs} probes', flush=True)


def run_swap(a):
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    tok = AutoTokenizer.from_pretrained(a.model)
    close_id = tok.convert_tokens_to_ids(CLOSE)
    star_id = single_token_ids(tok, ('**',))[0]
    sc_ids = set(single_token_ids(tok, SELF_CORRECTION_WORDS))
    recs = [r for r in load_records(a.runs_root, a.run) if r['sid'] < a.samples][a.shard::a.num_shards]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'shard{a.shard}.jsonl'
    done = done_keys(path, ('bench', 'qid', 'sid'))
    recs = [r for r in recs if (r['bench'], r['qid'], r['sid']) not in done]
    toks = tokenized(tok, recs)
    cut_at = 'end'
    if a.cut_pair:
        cut_at = f'pair:{a.cut_pair}'
        para = paragraph_ids(tok)
        ref = tokenized(tok, load_records(a.runs_root, a.cut_pair))
        for k, (ids, think, end) in toks.items():
            toks[k] = (ids, think, min(end, snap_to_paragraph(ids, ref[k][2], para, nearest=True, upper=end)))
    print(f'shard {a.shard}: {len(recs)} rollouts of {a.run} continued by {a.label}', flush=True)
    if not recs:
        return
    llm = make_llm(a.model)
    for i in range(0, len(recs), a.batch):
        chunk = recs[i:i + a.batch]
        keys, prompts, params = [], [], []
        for r in chunk:
            key = (r['bench'], r['qid'], r['sid'])
            ids, _, end = toks[key]
            if CAP - end < 1:   # the rollout spent its whole budget before the cut (capped, no paragraph break
                continue        # near the end): nothing to continue, in every arm alike
            keys.append(key)
            prompts.append(TokensPrompt(prompt_token_ids=tok(r['prompt'], add_special_tokens=False).input_ids + ids[:end]))
            params.append(SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=CAP - end,
                                         seed=seed_of(a.label, *key)))
        outs = dict(zip(keys, llm.generate(prompts, sampling_params=params, use_tqdm=False))) if prompts else {}
        with open(path, 'a') as f:
            for r in chunk:
                key = (r['bench'], r['qid'], r['sid'])
                ids, think, end = toks[key]
                c = outs[key].outputs[0] if key in outs else None
                new, cont = (list(c.token_ids), c.text) if c else ([], '')
                closed_at = new.index(close_id) if close_id in new else None
                added = closed_at if closed_at is not None else len(new)
                text = tok.decode(ids[:end], skip_special_tokens=False) + cont
                f.write(json.dumps(dict(
                    bench=r['bench'], qid=r['qid'], sid=r['sid'], run=a.run, label=a.label, cut_at=cut_at,
                    end=end, think=think,
                    natural_correct=r['correct'], correct=grade(r['bench'], text, r), added_think=added,
                    closed=closed_at is not None, cap_hit=c.finish_reason == 'length' if c else True,
                    commits_first=bool(new) and new[0] == star_id, n_sc_added=sum(t in sc_ids for t in new[:added]),
                    continuation=cont, generated=c is not None)) + '\n')
        print(f'shard {a.shard}: {min(i + a.batch, len(recs))}/{len(recs)}', flush=True)


def run_exit(a):
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    tok = AutoTokenizer.from_pretrained(a.model)
    close_id = tok.convert_tokens_to_ids(CLOSE)
    exit_ids = tok(EXIT_TEXT, add_special_tokens=False).input_ids
    recs = load_records(a.runs_root, a.run)[a.shard::a.num_shards]
    refs = {r: {(x['bench'], x['qid'], x['sid']): x['think_tokens'] for x in load_records(a.runs_root, r)}
            for r in a.pair_runs}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'shard{a.shard}.jsonl'
    fields = ('bench', 'qid', 'sid', 'kind', 'cut')
    done = done_keys(path, fields)

    natural, jobs = [], []
    for r in recs:
        ids = tok(r['response'], add_special_tokens=False).input_ids
        think = split_thinking(ids, close_id)
        prompt_ids = tok(r['prompt'], add_special_tokens=False).input_ids
        paired = {ref: refs[ref][(r['bench'], r['qid'], r['sid'])] for ref in a.pair_runs}
        for kind, cut in exit_cuts(a.grid, paired):
            row = dict(bench=r['bench'], qid=r['qid'], sid=r['sid'], run=a.run, kind=kind, cut=cut,
                       natural_think=think, natural_correct=r['correct'])
            if tuple(row[k] for k in fields) in done:
                continue
            if cut >= think:
                natural.append(dict(row, forced=False, correct=r['correct']))
            else:
                jobs.append((row, r, ids, prompt_ids))
    with open(path, 'a') as f:
        for row in natural:
            f.write(json.dumps(row) + '\n')
    print(f'shard {a.shard}: {len(recs)} rollouts, {len(natural)} natural, {len(jobs)} forced exits', flush=True)
    if not jobs:
        return

    llm = make_llm(a.model)
    for i in range(0, len(jobs), a.batch):
        chunk = jobs[i:i + a.batch]
        prompts = [TokensPrompt(prompt_token_ids=p + ids[:row['cut']] + exit_ids) for row, _, ids, p in chunk]
        params = [SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, seed=seed_of(*(row[k] for k in fields)),
                                 max_tokens=min(ANSWER_CAP, CAP - row['cut'] - len(exit_ids)))
                  for row, *_ in chunk]
        outs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
        with open(path, 'a') as f:
            for (row, r, ids, _), o in zip(chunk, outs):
                c = o.outputs[0]
                text = tok.decode(ids[:row['cut']], skip_special_tokens=False) + EXIT_TEXT + c.text
                f.write(json.dumps(dict(row, forced=True, correct=grade(r['bench'], text, r),
                                        answer_tokens=len(c.token_ids), answer_finish=c.finish_reason,
                                        answer_text=c.text)) + '\n')
        print(f'shard {a.shard}: {min(i + a.batch, len(jobs))}/{len(jobs)} forced exits', flush=True)


def run_continue(a):
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind
    tok = AutoTokenizer.from_pretrained(a.model)
    close_id = tok.convert_tokens_to_ids(CLOSE)

    ref = load_records(a.runs_root, a.target_run)
    by_q = defaultdict(list)
    for r in ref:
        by_q[(r['bench'], r['qid'])].append(r)
    questions = sorted(by_q)[a.shard::a.num_shards]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'shard{a.shard}.jsonl'
    done = done_keys(path, ('bench', 'qid', 'sid'))
    items = {}
    for q in questions:
        rs = by_q[q]
        target = int(median(r['think_tokens'] for r in rs))
        prompt_ids = tok(rs[0]['prompt'], add_special_tokens=False).input_ids
        for k in range(a.samples):
            if (q[0], q[1], k) not in done:
                items[f'{q[0]}|{q[1]}|{k}'] = dict(bench=q[0], qid=q[1], sid=k, target=target, rec=rs[0],
                                                    prompt_ids=prompt_ids, gen=[], forced=0, waits=[],
                                                    phase='think', rounds=0, think=None)
    print(f'shard {a.shard}: {len(questions)} problems, {len(items)} rollouts to run', flush=True)
    if not items:
        return

    llm = make_llm(a.model)
    engine = llm.llm_engine

    def submit(rid):
        it = items[rid]
        params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=CAP - len(it['gen']),
                                seed=seed_of(a.label, rid, it['rounds']), output_kind=RequestOutputKind.FINAL_ONLY,
                                stop_token_ids=[close_id] if it['phase'] == 'think' else None)
        engine.add_request(f'{rid}#{it["rounds"]}', TokensPrompt(prompt_token_ids=it['prompt_ids'] + it['gen']),
                           params)
        it['rounds'] += 1

    def finish(it, f):
        text = tok.decode(it['gen'], skip_special_tokens=False)
        think = it['think'] if it['think'] is not None else len(it['gen'])
        f.write(json.dumps(dict(bench=it['bench'], qid=it['qid'], sid=it['sid'], run=a.label, target=it['target'],
                                think_tokens=think, response_tokens=len(it['gen']), forced=it['forced'],
                                waits=it['waits'], reached=think >= it['target'],
                                closed=it['think'] is not None, cap_hit=len(it['gen']) >= CAP,
                                correct=grade(it['bench'], text, it['rec']), response=text)) + '\n')
        f.flush()

    for rid in items:
        submit(rid)
    n_done = 0
    with open(path, 'a') as f:
        while engine.has_unfinished_requests():
            for out_ in engine.step():
                if not out_.finished:
                    continue
                rid = out_.request_id.rsplit('#', 1)[0]
                it, c = items[rid], out_.outputs[0]
                new = list(c.token_ids)
                closed = it['phase'] == 'think' and c.finish_reason == 'stop' and (
                    c.stop_reason == close_id or (new and new[-1] == close_id))
                if new and new[-1] == close_id:
                    new = new[:-1]
                it['gen'] += new
                room = CAP - len(it['gen'])
                if closed and len(it['gen']) < it['target'] and it['forced'] < a.max_forced and room > a.min_room:
                    it['gen'] = wait_splice(tok, it['gen'])
                    it['forced'] += 1
                    it['waits'].append(len(it['gen']))
                    submit(rid)
                elif closed and room > 0:
                    it['think'] = len(it['gen'])
                    it['gen'].append(close_id)
                    it['phase'] = 'answer'
                    submit(rid)
                else:
                    finish(it, f)
                    n_done += 1
                    if n_done % 20 == 0:
                        print(f'shard {a.shard}: {n_done}/{len(items)} done', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='mode', required=True)
    for name in ('belief', 'swap', 'exit', 'continue'):
        s = sub.add_parser(name)
        s.add_argument('--model', required=True)
        s.add_argument('--out', required=True)
        s.add_argument('--runs-root', default='runs')
        s.add_argument('--shard', type=int, default=0)
        s.add_argument('--num-shards', type=int, default=1)
    b = sub.choices['belief']
    b.add_argument('--run', required=True, help='eval run whose rollouts are probed')
    b.add_argument('--step', type=int, default=1024)
    b.add_argument('--pair-runs', nargs='*', default=[])
    b.add_argument('--group', type=int, default=4, help='rollouts per prefix-cache group')
    b.add_argument('--answer-samples', type=int, default=4, help='sampled answers per probe (eval sampling)')
    w = sub.choices['swap']
    w.add_argument('--run', required=True, help='eval run whose rollouts are cut at their end of reasoning')
    w.add_argument('--label', required=True, help='name of the continuing model')
    w.add_argument('--samples', type=int, default=16)
    w.add_argument('--cut-pair', help='cut where the paired rollout of this run ended its reasoning')
    w.add_argument('--batch', type=int, default=256)
    e = sub.choices['exit']
    e.add_argument('--run', required=True, help='eval run whose rollouts are cut')
    e.add_argument('--grid', type=int, nargs='*', default=list(GRID))
    e.add_argument('--pair-runs', nargs='*', default=[])
    e.add_argument('--batch', type=int, default=512)
    c = sub.choices['continue']
    c.add_argument('--label', required=True)
    c.add_argument('--target-run', required=True, help='eval run whose per-problem median thinking is L(q)')
    c.add_argument('--samples', type=int, default=16)
    c.add_argument('--max-forced', type=int, default=16)
    c.add_argument('--min-room', type=int, default=1024, help='no forcing when fewer tokens remain')
    a = p.parse_args()
    modes = {'belief': run_belief, 'swap': run_swap, 'exit': run_exit, 'continue': run_continue}
    modes[a.mode](a)


if __name__ == '__main__':
    main()
