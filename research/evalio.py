"""Shared helpers for the graded avg@16 eval rollouts and their thinking/answer split.

Eval layout: <runs_root>/<run>/eval/<bench>_38912/<merged>.jsonl, one graded record per rollout
(written by run_generate.py --research_metrics; most fields are stored as strings).
"""
import glob
import json
import os

BENCHES = ('aime24', 'aime25', 'hmmt25')
CAP = 38912                      # eval generation cap (prompt excluded)
TEMPERATURE, TOP_P = 0.6, 0.95   # eval sampling protocol
CLOSE = '</think>'

# Paragraph-initial markers. Single Qwen3 tokens, checked by tests/test_e1e4.py.
SELF_CORRECTION_WORDS = ('Wait', ' Wait', ' wait', 'Hmm', ' Hmm', 'Alternatively', ' Alternatively')
CONTRAST_WORDS = ('But', ' But', 'However', ' However', 'Actually', ' Actually')
CONCLUSION_WORDS = ('Therefore', ' Therefore', 'Thus', ' Thus', ' Hence')
# A paragraph opening with '**' in Qwen3-1.7B thinking is '**Final Answer**' 95% of the time: committing to an answer.
COMMIT_WORDS = ('**',)


def to_bool(v):
    return v in (True, 'True', 'true', 1, '1')


def eval_file(runs_root, run, bench):
    files = sorted(glob.glob(os.path.join(runs_root, run, 'eval', f'{bench}_{CAP}', '*.jsonl')))
    if len(files) != 1:
        raise FileNotFoundError(f'expected one merged jsonl for {run}/{bench}, found {files}')
    return files[0]


def load_records(runs_root, run, benches=BENCHES):
    """Graded rollouts of one run, typed and sorted by (bench, qid, sid)."""
    out = []
    for b in benches:
        with open(eval_file(runs_root, run, b)) as f:
            for line in f:
                r = json.loads(line)
                r.update(bench=b, qid=int(r['question_id']),
                         sid=int(r['sample_id'] if r.get('sample_id') not in (None, 'None') else r['generation_id']),
                         correct=to_bool(r['is_correct']), think_tokens=int(r['reasoning_tokens']),
                         response_tokens=int(r['response_tokens']), capped=to_bool(r['cap_hit']),
                         closed=to_bool(r['reasoning_closed']))
                out.append(r)
    out.sort(key=lambda r: (r['bench'], r['qid'], r['sid']))
    keys = [(r['bench'], r['qid'], r['sid']) for r in out]
    if len(set(keys)) != len(keys):
        raise ValueError(f'duplicate rollouts in {run}')
    return out


_TASKS = {}


def grade(bench, text, rec):
    """Grade a response text with the benchmark's own grader (same as the eval)."""
    from evaluation.tasks import get_task
    task = _TASKS.setdefault(bench, get_task(bench))
    return bool(task.grade(text, task.get_gold(rec), rec)['is_correct'])


def split_thinking(ids, close_id):
    """Number of response tokens before the closing think tag (all of them if it never closes)."""
    return ids.index(close_id) if close_id in ids else len(ids)


def paragraph_ids(tok):
    """Token ids whose text ends a paragraph (a blank line); the next token opens a paragraph."""
    return {i for i in range(len(tok)) if tok.decode([i]).endswith('\n\n')}


def commit_points(ids, think, para, star_id, final_id):
    """Positions j < think where a paragraph opens with '**Final' ('**Final Answer**'): where the model
    commits to an answer inside its thinking."""
    return [j for j in range(1, think - 1) if ids[j] == star_id and ids[j + 1] == final_id and ids[j - 1] in para]


def snap_to_paragraph(ids, cut, para, window=512, nearest=False, upper=None):
    """A paragraph start j near cut (a paragraph opens at j when ids[j-1] ends a blank line).
    Default: the largest j <= cut within window. nearest=True: the closest j in either direction
    (ties go back), which does not bias the amount of text kept; j never exceeds upper.
    Returns cut itself when no paragraph start is within the window."""
    hi = min(len(ids), upper if upper is not None else len(ids))
    back = next((j for j in range(min(cut, hi), max(cut - window, 1) - 1, -1) if ids[j - 1] in para), None)
    if not nearest:
        return back if back is not None else cut
    fwd = next((j for j in range(max(cut, 1), min(cut + window, hi) + 1) if ids[j - 1] in para), None)
    cands = [j for j in (back, fwd) if j is not None]
    return min(cands, key=lambda j: (abs(j - cut), j)) if cands else cut


def end_of_reasoning(ids, think, para, star_id, final_id):
    """Where the reasoning of a rollout ends: its last '**Final Answer**' commit inside the thinking, or
    (no commit) the last paragraph start before the closing tag."""
    cps = commit_points(ids, think, para, star_id, final_id)
    return cps[-1] if cps else snap_to_paragraph(ids, think, para)


def single_token_ids(tok, words):
    ids = []
    for w in words:
        t = tok(w, add_special_tokens=False).input_ids
        if len(t) != 1:
            raise ValueError(f'{w!r} is not a single token: {t}')
        ids.append(t[0])
    return ids
