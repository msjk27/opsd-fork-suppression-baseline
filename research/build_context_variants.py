"""Add teacher-context variants to the OPSD training parquet (E4 signal analysis, E5 training controls).

Row order and every source column stay untouched, so all E5 arms see the same prompts in the same order as
the vanilla run. New columns, usable as gold_answer_key in a training config:
  answer_only          the final answer: last non-empty \\boxed{} of the gold solution, else of the R1 trace for
                       the same problem, else the last line of the solution (proofs with an empty box).
                       The `Answer` column is not used: 27% of its values are truncated (e.g. `\\dfrac{85`).
  answer_only_source   which of the three produced it
  solution_shuffled    the gold solution of a different problem with a different final answer
  shuffled_from_row    the row that solution came from
COT_Reason (the R1 thinking trace for the same problem) is already in the source file.
Usage: python -m research.build_context_variants SRC.parquet DST.parquet [--seed 0]
"""
import argparse
import random
import re

from research.probe import last_boxed


def derangement(n, seed):
    """Uniform random permutation with no fixed points (rejection sampling; ~e tries on average)."""
    rng = random.Random(seed)
    while True:
        perm = list(range(n))
        rng.shuffle(perm)
        if all(i != j for i, j in enumerate(perm)):
            return perm


def last_nonempty_box(text):
    """Content of the last \\boxed{} that holds something, stripped, or None."""
    while text:
        box = last_boxed(text)
        if box is not None and box.strip():
            return box.strip()
        cut = text.rfind('\\boxed{')
        if cut < 0:
            return None
        text = text[:cut]
    return None


def last_statement(solution):
    """Last line of the solution that says something (no box, not just math delimiters)."""
    for line in reversed(solution.splitlines()):
        line = line.strip()
        if '\\boxed' not in line and re.search(r'[A-Za-z0-9]', line):
            return line
    return None


def final_answer(solution, cot):
    """(answer, source) for one row; the answer is never empty."""
    for source, value in (('solution_box', last_nonempty_box(solution)), ('cot_box', last_nonempty_box(cot)),
                          ('solution_last_line', last_statement(solution))):
        if value:
            return value, source
    raise ValueError('no final answer found')


def normalize_answer(text):
    """Comparison key: spacing and LaTeX presentation differences do not make two answers different."""
    text = re.sub(r'\\text\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'\\[dt]frac', r'\\frac', text)
    text = re.sub(r'\\left|\\right|\\[,;!:]|\\ |\$|\s', '', text)
    return text.rstrip('.').lower()


def distinct_derangement(keys, seed):
    """Derangement of range(len(keys)) where row i never receives a row with the same key. Starts from
    derangement(n, seed) and repairs only the conflicting rows by swapping targets with random rows."""
    n = len(keys)
    perm = derangement(n, seed)
    rng = random.Random(seed + 1)
    bad = lambda i: perm[i] == i or keys[i] == keys[perm[i]]
    for i in range(n):
        while bad(i):
            j = rng.randrange(n)
            perm[i], perm[j] = perm[j], perm[i]
            if bad(i) or bad(j):
                perm[i], perm[j] = perm[j], perm[i]
    return perm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('src')
    p.add_argument('dst')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    import pandas as pd
    df = pd.read_parquet(a.src)
    answers = [final_answer(s, c) for s, c in zip(df.solution, df.COT_Reason)]
    df['answer_only'] = [x[0] for x in answers]
    df['answer_only_source'] = [x[1] for x in answers]
    perm = distinct_derangement([normalize_answer(x[0]) for x in answers], a.seed)
    df['solution_shuffled'] = df.solution.values[perm]
    df['shuffled_from_row'] = perm
    df.to_parquet(a.dst, index=False)
    print(f'{len(df)} rows -> {a.dst}; answer_only sources: {df.answer_only_source.value_counts().to_dict()}')


if __name__ == '__main__':
    main()
