"""Pure Python metrics; lexical cues are proxies, never fork labels."""
import math
import re
from collections import defaultdict


def fork_lock(top_probability):
    if not 0 <= top_probability <= 1:
        raise ValueError("Expected a normalized top probability")
    return top_probability < 0.25, top_probability > 0.65


def accuracy_metrics(records, ks=(1, 2, 4, 8, 16)):
    groups = defaultdict(list)
    seen = set()
    for r in records:
        key = (r.get('dataset', ''), r['question_id'])
        if type(r['is_correct']) is not bool:
            raise ValueError('is_correct must be a graded boolean')
        sample_id = r.get('sample_id', r.get('generation_id'))
        if sample_id is not None:
            unique = key + (sample_id,)
            if unique in seen:
                raise ValueError('Duplicate sample_id')
            seen.add(unique)
        groups[key].append(r['is_correct'])
    if not groups:
        raise ValueError('No graded samples')
    result = {'questions': len(groups), 'samples': sum(map(len, groups.values()))}
    for k in ks:
        if k <= 0:
            raise ValueError('k must be positive')
        eligible = [g for g in groups.values() if len(g) >= k]
        result[f'eligible@{k}'] = len(eligible)
        result[f'avg@{k}'] = (sum(sum(g)/len(g) for g in eligible)/len(eligible)) if eligible else None
        result[f'pass@{k}'] = (sum(1 - math.comb(len(g)-sum(g), k)/math.comb(len(g), k)
                                 for g in eligible)/len(eligible)) if eligible else None
    return result


def response_metrics(token_ids, text, eos_ids, cap, tokenizer=None):
    eos_ids = set(eos_ids)
    stop = next((i for i, t in enumerate(token_ids) if t in eos_ids), None)
    length = len(token_ids) if stop is None else stop + 1
    # Count reasoning until closing tag; absent tag means censored reasoning.
    reasoning = text.split('</think>', 1)[0]
    cues = re.findall(r'\b(?:wait|hmm|maybe|however|verify|check|reconsider|backtrack)\b', reasoning.lower())
    return {'response_tokens': length, 'eos': stop is not None,
            'cap_hit': stop is None and length >= cap,
            'reasoning_tokens': len(tokenizer.encode(reasoning, add_special_tokens=False)) if tokenizer else None,
            'reasoning_closed': '</think>' in text, 'self_correction_proxy_count': len(cues),
            'self_correction_proxy_per_1k_tokens': len(cues)*1000/max(length, 1)}
