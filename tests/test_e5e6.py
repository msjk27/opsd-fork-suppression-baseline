"""Tests for E5 (teacher-context controls) and E6 (decision-position loss masks)."""
import ast
import json
import subprocess
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

from research.build_context_variants import (derangement, distinct_derangement, final_answer, last_nonempty_box,
                                             last_statement, normalize_answer)
from research.evalio import paragraph_ids
from research.masking import (DECISION_MODES, decision_mask, decision_positions, match_by_value, thinking_positions,
                              token_lookup)
from research.probe import last_boxed
from research.run import load_config

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / 'models' / 'Qwen3-1.7B'
CTX = ROOT / 'data' / 'raw' / 'openthoughts_math_filtered_15k_ctx_v2.parquet'
PARA, CLOSE = 7, 9      # toy token ids: 7 ends a blank line, 9 closes the thinking
LOOKUP = token_lookup({PARA}, 20)


class AnswerExtraction(unittest.TestCase):
    def test_escaped_braces_do_not_count(self):
        self.assertEqual(last_boxed(r'\boxed{\left\{ x=1 \right.}'), r'\left\{ x=1 \right.')
        self.assertEqual(last_boxed(r'a \boxed{1} b \boxed{\frac{1}{2}} c'), r'\frac{1}{2}')
        self.assertEqual(last_boxed(r'\boxed{a \\ {b}}'), r'a \\ {b}')
        self.assertIsNone(last_boxed(r'\boxed{never closed'))
        self.assertIsNone(last_boxed('no box'))

    def test_final_answer_fallback_chain(self):
        self.assertEqual(last_nonempty_box(r'\boxed{4} then \boxed{ }'), '4')
        self.assertEqual(final_answer(r'so \boxed{ 5 }', r'\boxed{6}'), ('5', 'solution_box'))
        self.assertEqual(final_answer(r'so \boxed{ }', r'trace \boxed{6}'), ('6', 'cot_box'))
        self.assertEqual(final_answer('Proved.\n\\[\n\\boxed{}\n\\]', 'none'), ('Proved.', 'solution_last_line'))
        self.assertEqual(last_statement('x\n\\]\n'), 'x')
        with self.assertRaises(ValueError):
            final_answer(r'\boxed{}', '')

    def test_shuffle_never_gives_a_row_its_own_answer(self):
        keys = ['a'] * 30 + ['b'] * 10 + [str(i) for i in range(60)]
        perm = distinct_derangement(keys, 0)
        self.assertEqual(sorted(perm), list(range(100)))
        self.assertTrue(all(i != j and keys[i] != keys[j] for i, j in enumerate(perm)))
        self.assertEqual(perm, distinct_derangement(keys, 0))
        unique = [str(i) for i in range(50)]
        self.assertEqual(distinct_derangement(unique, 3), derangement(50, 3))   # nothing to repair

    def test_normalized_answers(self):
        self.assertEqual(normalize_answer(r'\dfrac{1}{2}'), normalize_answer(r' \frac{1}{2} '))
        self.assertEqual(normalize_answer(r'\text{C}'), normalize_answer('c'))
        self.assertNotEqual(normalize_answer('12'), normalize_answer('21'))


@unittest.skipUnless(CTX.is_file(), 'context parquet not available')
class ContextData(unittest.TestCase):
    def test_columns(self):
        import pandas as pd
        df = pd.read_parquet(CTX)
        src = pd.read_parquet(ROOT / 'data' / 'raw' / 'openthoughts_math_filtered_15k.parquet', columns=['problem', 'solution'])
        self.assertTrue((df.problem.values == src.problem.values).all())         # same prompts, same order
        self.assertTrue((df.solution.values == src.solution.values).all())
        self.assertTrue((df.answer_only.str.strip() == df.answer_only).all() and (df.answer_only != '').all())
        perm = df.shuffled_from_row.values
        self.assertTrue((df.solution_shuffled.values == df.solution.values[perm]).all())
        keys = [normalize_answer(a) for a in df.answer_only]
        self.assertFalse(any(i == j or keys[i] == keys[j] for i, j in enumerate(perm)))


class TeacherPrompts(unittest.TestCase):
    def test_only_the_teacher_prompt_changes_with_the_context_column(self):
        from datasets import Dataset
        from train import prepare_distil_dataset
        template = '{prompt}\n\nExample:\n{gold_answer}\n\nNow answer.'
        data = Dataset.from_dict(dict(problem=[f'cost $x{i}' for i in range(16)], solution=[f'S$ {i}' for i in range(16)],
                                      answer_only=[f'A{i}' for i in range(16)]))
        full = prepare_distil_dataset(data, 'problem', 'solution', template, seed=42)
        answer = prepare_distil_dataset(data, 'problem', 'answer_only', template, seed=42)
        self.assertEqual(full['prompt'], answer['prompt'])                       # same order, same student prompts
        i = int(full[0]['prompt'][0]['content'].split('x')[1])
        self.assertEqual(full[0]['teacher_prompt'][0]['content'], f'cost $x{i}\n\nExample:\nS$ {i}\n\nNow answer.')
        self.assertEqual(answer[0]['teacher_prompt'][0]['content'], f'cost $x{i}\n\nExample:\nA{i}\n\nNow answer.')


class Configs(unittest.TestCase):
    def test_arms_differ_from_the_paper_config_only_where_intended(self):
        base = json.loads((ROOT / 'configs/research/vanilla_4096_3090_paper.json').read_text())['train']
        expected = {'e5_answer': {'gold_answer_key': 'answer_only'}, 'e5_shuffled': {'gold_answer_key': 'solution_shuffled'},
                    'e5_cot': {'gold_answer_key': 'COT_Reason'}, 'e6_decision': {'research_mask': 'decision'},
                    'e6_matched': {'research_mask': 'decision_matched'}, 'e6_random': {'research_mask': 'decision_random'}}
        for name, change in expected.items():
            path = ROOT / f'configs/research/{name}_3090.json'
            cfg = load_config(path)
            self.assertEqual(cfg['gpus'], 8)
            if name.startswith('e5'):
                change = dict(change, train_file_path='data/raw/openthoughts_math_filtered_15k_ctx_v2.parquet')
            diff = {k: v for k, v in cfg['train'].items() if base.get(k) != v}
            self.assertEqual(diff, change, name)
            self.assertEqual(set(cfg['train']), set(base), name)

    def test_decision_mask_requires_the_chunked_loss(self):
        import tempfile
        cfg = json.loads((ROOT / 'configs/research/e6_decision_3090.json').read_text())
        cfg['train']['chunked_loss'] = False
        with tempfile.NamedTemporaryFile('w', suffix='.json') as f:
            json.dump(cfg, f); f.flush()
            with self.assertRaises(ValueError):
                load_config(f.name)


class Masks(unittest.TestCase):
    #          0  1  2  3     4  5     6  7  8      9     10 11    12 13 (pad)
    ids = torch.tensor([[1, 2, 3, PARA, 4, PARA, 5, 6, PARA, 8, PARA, 3, CLOSE, PARA, 2, 0]])
    valid = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])

    def loss(self):
        return torch.arange(16, dtype=torch.float32).unsqueeze(0) / 16

    def test_positions_follow_the_blank_line_and_stop_at_the_close(self):
        d = decision_positions(self.ids, self.valid, LOOKUP, CLOSE)[0].nonzero().flatten().tolist()
        self.assertEqual(d, [4, 6, 9, 11])                                       # 14 follows PARA but is after the close
        self.assertEqual(thinking_positions(self.ids, CLOSE)[0].sum().item(), 12)
        self.assertTrue(thinking_positions(torch.tensor([[1, 2]]), CLOSE).all())  # never closed: all thinking

    def test_skipped_tokens_are_never_decisions(self):
        ids = torch.tensor([[PARA, 1, 2, 3]])
        self.assertFalse(decision_positions(ids, torch.tensor([[0, 0, 0, 1]]), LOOKUP, CLOSE).any())

    def test_ids_beyond_the_tokenizer_are_safe(self):
        ids = torch.tensor([[50000, 1, PARA, 2]])
        self.assertEqual(decision_positions(ids, torch.ones_like(ids), LOOKUP, CLOSE)[0].nonzero().flatten().tolist(), [3])

    def test_decision_mode_drops_exactly_the_decisions(self):
        keep, s = decision_mask(self.ids, self.valid, LOOKUP, CLOSE, 'decision', self.loss(), 0)
        self.assertEqual((self.valid - keep)[0].nonzero().flatten().tolist(), [4, 6, 9, 11])
        self.assertAlmostEqual(s['dropped_frac'].item(), 4 / 12)
        self.assertEqual(s['dropped_frac'], s['decision_frac'])
        self.assertAlmostEqual(s['dropped_loss_share'].item(), s['decision_loss_share'].item())

    def test_controls_match_the_count_and_leave_decisions_in_the_loss(self):
        d = decision_positions(self.ids, self.valid, LOOKUP, CLOSE)
        region = thinking_positions(self.ids, CLOSE) & self.valid.bool() & ~d
        for mode in ('decision_matched', 'decision_random'):
            keep, _ = decision_mask(self.ids, self.valid, LOOKUP, CLOSE, mode, self.loss(), 5)
            dropped = (self.valid - keep).bool()
            self.assertEqual(dropped.sum().item(), min(d.sum().item(), region.sum().item()))
            self.assertFalse((dropped & d).any())
            self.assertFalse((dropped & ~region).any())                          # thinking region only, no padding
            self.assertTrue(torch.equal(keep, decision_mask(self.ids, self.valid, LOOKUP, CLOSE, mode, self.loss(), 5)[0]))

    def test_matched_control_removes_the_same_loss_mass(self):
        torch.manual_seed(0)
        n = 3000
        ids = torch.randint(1, 6, (1, n))
        ids[0, ::40] = PARA
        loss = torch.rand(1, n) ** 8
        loss[0, 1::40] += 2.0                                                   # decisions carry far more loss
        valid = torch.ones_like(ids)
        _, dec = decision_mask(ids, valid, LOOKUP, CLOSE, 'decision', loss, 1)
        _, mat = decision_mask(ids, valid, LOOKUP, CLOSE, 'decision_matched', loss, 1)
        _, ran = decision_mask(ids, valid, LOOKUP, CLOSE, 'decision_random', loss, 1)
        self.assertEqual(dec['dropped_frac'], mat['dropped_frac'])
        self.assertEqual(dec['dropped_frac'], ran['dropped_frac'])
        self.assertLess(ran['dropped_loss_share'], dec['dropped_loss_share'] / 2)
        # Candidates here top out below the decisions' loss, so the match takes the largest available ones.
        self.assertGreater(mat['dropped_loss_share'], ran['dropped_loss_share'] * 1.5)

    def test_match_by_value(self):
        self.assertEqual(sorted(match_by_value([5.0, 1.0], [10, 11, 12, 13], [0.9, 4.0, 6.5, 1.2])), [10, 11])
        self.assertEqual(len(match_by_value([1.0, 2.0, 3.0], [1], [2.0])), 1)

    def test_global_rng_is_untouched(self):
        torch.manual_seed(3)
        expected = torch.rand(1)
        torch.manual_seed(3)
        for mode in DECISION_MODES:
            decision_mask(self.ids, self.valid, LOOKUP, CLOSE, mode, self.loss(), 1)
        self.assertEqual(torch.rand(1), expected)


def chunked_fn(source):
    tree = ast.parse(source)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_compute_loss_chunked')
    ns = {'torch': torch, '__package__': 'opsd', '__name__': 'opsd.trainer'}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<chunked loss>', 'exec'), ns)
    return ns['_compute_loss_chunked']


class ChunkedLoss(unittest.TestCase):
    ids = torch.tensor([[1, 2, 3, PARA, 4, 5, CLOSE, PARA, 2, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])

    def run_loss(self, fn, research_mask, ratio=None):
        torch.manual_seed(0)
        n, v = self.ids.shape[1], 20
        raw = torch.randn(1, n, v, requires_grad=True)
        teacher = torch.randn(1, n, v)
        inputs = dict(prompt_ids=torch.ones(1, 2, dtype=torch.long), prompt_mask=torch.ones(1, 2, dtype=torch.long),
                      teacher_prompt_ids=torch.ones(1, 3, dtype=torch.long),
                      teacher_prompt_mask=torch.ones(1, 3, dtype=torch.long),
                      completion_ids=self.ids, completion_mask=self.mask)
        if ratio is not None:
            inputs['importance_sampling_ratio'] = ratio
        model = SimpleNamespace(training=True)
        student_model = object()
        from opsd.trainer import DistilTrainer
        obj = SimpleNamespace(
            top_entropy_quantile=1.0, beta=0.0, alpha=0.5, temperature=1.0, teacher_temperature=None,
            num_loss_tokens_to_skip=3, use_vllm=ratio is not None, vllm_importance_sampling_correction=True,
            generate_from_teacher=False, current_gradient_accumulation_steps=8, model=model, ref_model=object(),
            processing_class=None, _metrics={'train': defaultdict(list)}, _paragraph_lookup=LOOKUP, _think_close_id=CLOSE,
            args=SimpleNamespace(research_mask=research_mask, research_diagnostics=False, seed=42, chunked_loss_size=4),
            accelerator=SimpleNamespace(gather=lambda x: x, process_index=0))
        obj._completion_logits = lambda m, *a: raw if m is student_model else teacher
        obj._decision_loss_mask = lambda *a: DistilTrainer._decision_loss_mask(obj, *a)
        loss = fn(obj, student_model, inputs)
        loss.backward()
        return loss.detach(), raw.grad.abs().sum(-1)[0], obj._metrics['train']

    def test_vanilla_is_identical_to_the_committed_code(self):
        old = chunked_fn(subprocess.check_output(['git', 'show', 'fc90d10:opsd/trainer.py'], cwd=ROOT).decode())
        new = chunked_fn((ROOT / 'opsd/trainer.py').read_text(encoding='utf-8'))
        ratio = torch.tensor([[1., 1., 1., .5, 2., 1.5, 1., .8, 1.2, 0.]])
        for r in (None, ratio):
            l0, g0, m0 = self.run_loss(old, 'vanilla', r)
            l1, g1, m1 = self.run_loss(new, 'vanilla', r)
            self.assertTrue(torch.equal(l0, l1))
            self.assertTrue(torch.equal(g0, g1))
            self.assertEqual(dict(m0), dict(m1))                                # no extra metric in vanilla runs

    def test_masked_tokens_get_no_gradient_and_others_do(self):
        new = chunked_fn((ROOT / 'opsd/trainer.py').read_text(encoding='utf-8'))
        _, g, metrics = self.run_loss(new, 'decision')
        self.assertEqual(g.nonzero().flatten().tolist(), [3, 5, 6, 7, 8])       # 4 follows PARA and is dropped
        self.assertAlmostEqual(metrics['mask/dropped_frac'][0], 1 / 6)

    def test_importance_weight_is_not_changed_by_the_mask(self):
        new = chunked_fn((ROOT / 'opsd/trainer.py').read_text(encoding='utf-8'))
        ratio = torch.tensor([[1., 1., 1., 1., 9., 1., 1., 1., 1., 0.]])        # the masked token has an outlying ratio
        l_flat, _, _ = self.run_loss(new, 'decision', torch.tensor([[1., 1., 1., 1., 9., 1., 1., 1., 1., 0.]]))
        l_ref, _, _ = self.run_loss(new, 'decision', ratio * 1.0)
        self.assertTrue(torch.equal(l_flat, l_ref))
        l_none, _, _ = self.run_loss(new, 'decision')
        weight = (ratio[0, 3:9].sum() / 6).item()
        self.assertAlmostEqual(l_flat.item() / l_none.item(), weight, places=4)


class Report(unittest.TestCase):
    def test_end_to_end_on_a_synthetic_run_tree(self):
        import random, tempfile
        from research.e5e6_report import ARMS, E6, Results, section_accuracy, section_decisions, section_e6
        acc_of = {'base': .50, 'base-top1': .50, 'vanilla-3090-paper-s42': .30, 'vanilla-4x16-s42': .30, 'e6-decision-s42': .55, 'e6-matched-s42': .30}
        rng = random.Random(0)
        with tempfile.TemporaryDirectory() as root:
            for run, p in acc_of.items():
                n = 48 if run.startswith('e6') else 16
                for b in ('aime24', 'aime25', 'hmmt25'):
                    d = Path(root) / run / 'eval' / f'{b}_38912'
                    d.mkdir(parents=True)
                    with open(d / 'merged.jsonl', 'w') as f:
                        for q in range(30):
                            for k in range(n):
                                ok = k < 16 and q % 2 == 0 if run == 'e6-decision-s42' else rng.random() < p
                                f.write(json.dumps(dict(question_id=q, sample_id=k, is_correct=ok, reasoning_tokens=100,
                                                        response_tokens=110, cap_hit=False, reasoning_closed=True)) + '\n')
            for run, fa, sc in (('base', .004, .17), ('vanilla-4x16-s42', .008, .15), ('e6-decision-s42', .004, .17),
                                ('e6-matched-s42', .008, .15)):
                d = Path(root) / 'e1-dec' / f'{run}-on-base'
                d.mkdir(parents=True)
                with open(d / 'shard0.jsonl', 'w') as f:
                    for b in ('aime24', 'aime25', 'hmmt25'):
                        for q in range(30):
                            for k in range(4):
                                f.write(json.dumps(dict(model=run, run='base', bench=b, qid=q, sid=k, para_p_fa=fa + rng.gauss(0, 1e-3),
                                                        para_p_sc=sc + rng.gauss(0, 1e-2))) + '\n')
            for run, loss in (('e6-decision-s42', .09), ('e6-matched-s42', .088)):
                d = Path(root) / run / 'checkpoint' / 'checkpoint-234'
                d.mkdir(parents=True)
                log = [{'step': i, 'mask/dropped_frac': .02, 'mask/dropped_loss_share': loss, 'mask/decision_frac': .02,
                        'mask/decision_loss_share': .09} for i in range(3)]
                (d / 'trainer_state.json').write_text(json.dumps(dict(log_history=log)))
            res = Results(root, ARMS)
            self.assertEqual(len(res.rollouts('base')), 32 * 90)                 # base and base-top1 are pooled
            self.assertEqual(len(res.rollouts('e6-decision')), 48 * 90)
            out = []
            section_accuracy(res, out)
            section_decisions(res, out)
            section_e6(res, out)
        text = '\n'.join(out)
        self.assertIn('Balanced: **yes**', text)
        self.assertIn('Mask changed the policy: **yes**', text)
        self.assertIn('Primary endpoint interpretable (both checks passed): **yes**', text)
        self.assertIn('e6-random | 0 |', text)                                  # a missing arm is pending, not an error
        self.assertEqual(E6[0], 'e6-decision')
        # The rollouts beyond the first 16 of the decision arm are all wrong in this fixture: pooled acc is 1/6, avg@16 is 1/2.
        row = next(l for l in text.splitlines() if l.startswith('| e6-decision | 1'))
        self.assertIn('| 48 | 0.167 | 0.500 |', row)


@unittest.skipUnless(TOKENIZER.is_dir(), 'tokenizer not available')
class RealTokenizer(unittest.TestCase):
    def test_positions_equal_the_ones_measured_in_e1c(self):
        from transformers import AutoTokenizer
        from research.evalio import CLOSE as TAG, split_thinking
        tok = AutoTokenizer.from_pretrained(TOKENIZER)
        text = '<think>\nFirst.\n\nWait, no.\n\n**Final Answer**\n\\boxed{1}\n</think>\n\nDone.\n\nReally.'
        ids = tok(text, add_special_tokens=False).input_ids
        para, close = paragraph_ids(tok), tok.convert_tokens_to_ids(TAG)
        think = split_thinking(ids, close)
        measured = [j for j in range(3, think) if ids[j - 1] in para]            # self_entropy.summarize, after the skip
        t = torch.tensor([ids])
        valid = (torch.arange(len(ids)) >= 3).long().unsqueeze(0)
        got = decision_positions(t, valid, token_lookup(para, len(tok)), close)[0].nonzero().flatten().tolist()
        self.assertEqual(got, measured)
        self.assertEqual([tok.decode([ids[j]]) for j in got], ['Wait', '**'])


if __name__ == '__main__':
    unittest.main()
