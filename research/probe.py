"""Small HF probe on identical student trajectories under base/sparse/dense teachers.

This diagnostic uses HF sampling (no top-k truncation), not the upstream vLLM evaluator.
"""
import argparse
import json
from pathlib import Path


def last_boxed(text):
    """Content of the last balanced \\boxed{...} in text, or None."""
    start = text.rfind('\\boxed{')
    if start < 0:
        return None
    i, depth = start + len('\\boxed{'), 1
    for j in range(i, len(text)):
        depth += {'{': 1, '}': -1}.get(text[j], 0)
        if depth == 0:
            return text[i:j]
    return None


def sparse_answer(row, source):
    """Answer-only privileged context. 'answer' = dataset column (27% truncated in OpenThoughts 15k);
    'solution_boxed' = last \\boxed{} of the gold solution, falling back to the column."""
    if source == 'solution_boxed':
        return last_boxed(row['solution']) or row['Answer']
    return row['Answer']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/research/probe_1024.json')
    a = p.parse_args()
    c = json.loads(Path(a.config).read_text(encoding='utf-8'))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from datasets import load_dataset
    from research.signals import token_signals, write_positions
    from research.metrics import response_metrics
    from research.run import sha256
    import subprocess
    import platform
    out = Path(c['output_dir'])
    out.mkdir(parents=True, exist_ok=False)
    set_seed(c['seed'])
    tok = AutoTokenizer.from_pretrained(c['student'])
    teacher_tok = AutoTokenizer.from_pretrained(c['teacher'])
    if tok.get_vocab() != teacher_tok.get_vocab():
        raise ValueError('Teacher/student vocabularies must match')
    models = [AutoModelForCausalLM.from_pretrained(c[role], dtype=torch.bfloat16,
               attn_implementation='sdpa').to(c[role+'_device']).eval() for role in ('student', 'teacher')]
    student, teacher = models
    rows = load_dataset('parquet', data_files=c['data'], split='train')
    eos = student.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    manifest = dict(c, dataset_sha256=sha256(c['data']),
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    python=platform.python_version(), torch=torch.__version__,
                    student_model_commit=getattr(student.config, '_commit_hash', None),
                    teacher_model_commit=getattr(teacher.config, '_commit_hash', None))
    (out/'config.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    def prompt(text):
        ids = tok.apply_chat_template([{'role':'user', 'content':text}], tokenize=True,
                                     add_generation_prompt=True, enable_thinking=True, return_tensors='pt')
        if ids.shape[1] > c['max_prompt_length']:
            raise ValueError('Prompt exceeds configured cap; no silent truncation in probe')
        return ids

    def score(model, ids, completion):
        device = next(model.parameters()).device
        seq = torch.cat([ids, completion.cpu()], dim=1).to(device)
        # Keep C+1 logits, then shift; predictions align with completion token IDs.
        with torch.inference_mode():
            logits = model(seq, attention_mask=torch.ones_like(seq), use_cache=False,
                           logits_to_keep=completion.shape[1]+1).logits[:, :-1]
            return logits.float().log_softmax(-1).cpu()

    for qid, row in enumerate(rows.select(range(min(c['limit'], len(rows))))):
        text = row['problem']
        ids = prompt(text)
        for sample_id in range(c['samples']):
            with torch.inference_mode():
                generated = student.generate(ids.to(c['student_device']),
                    attention_mask=torch.ones_like(ids).to(c['student_device']),
                    max_new_tokens=c['max_tokens'], do_sample=True, temperature=c['temperature'],
                    top_p=c['top_p'], top_k=0, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            completion = generated[:, ids.shape[1]:].cpu()
            response = tok.decode(completion[0], skip_special_tokens=False)
            record = dict(question_id=qid, sample_id=sample_id, response=response,
                          **response_metrics(completion[0].tolist(), response, eos, c['max_tokens'], tok))
            with (out/'responses.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(record)+'\n')
            s = score(student, ids, completion)
            sparse = sparse_answer(row, c.get('sparse_answer_source', 'answer'))
            for condition, gold in [('base', None), ('sparse', sparse), ('dense', row['solution'])]:
                teacher_ids = prompt(text if gold is None else c['teacher_prompt_template'].format(prompt=text, gold_answer=gold))
                t = score(teacher, teacher_ids, completion)
                signals = token_signals(s, t, completion)
                write_positions(out/'positions.jsonl', signals, torch.ones_like(completion), completion,
                                {'question_id':qid, 'sample_id':sample_id, 'condition':condition})


if __name__ == '__main__':
    main()
