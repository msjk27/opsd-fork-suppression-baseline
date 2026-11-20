"""Tests for the E1-E4 helpers. Tokenizer-dependent tests run only when models/Qwen3-1.7B is present."""
import unittest
from pathlib import Path

import torch

from research.build_context_variants import derangement
from research.evalio import (CLOSE, COMMIT_WORDS, CONCLUSION_WORDS, CONTRAST_WORDS, SELF_CORRECTION_WORDS,
                             single_token_ids, split_thinking)
from research.evalio import commit_points, end_of_reasoning, snap_to_paragraph
from research.forcing import EXIT_TEXT, PROBE_TEXT, belief_cuts, exit_cuts, seed_of
from research.self_entropy import summarize

TOKENIZER = Path(__file__).resolve().parents[1] / 'models' / 'Qwen3-1.7B'


class PureTests(unittest.TestCase):
    def test_exit_cuts_grid_then_pairs(self):
        self.assertEqual(exit_cuts((10, 20), {'a': 15}), [('grid', 10), ('grid', 20), ('pair:a', 15)])

    def test_commit_points_and_end_of_reasoning(self):
        P, S, F = 7, 50, 51                       # paragraph end, '**', 'Final'
        ids = [1, P, S, F, 2, P, 3, P, S, F, 4, 9]  # commits open at 2 and 8; thinking = first 11 tokens
        self.assertEqual(commit_points(ids, 11, {P}, S, F), [2, 8])
        self.assertEqual(end_of_reasoning(ids, 11, {P}, S, F), 8)
        self.assertEqual(commit_points(ids, 8, {P}, S, F), [2])      # the second commit is past the thinking
        no_commit = [1, P, 2, 3, P, 4, 5]
        self.assertEqual(end_of_reasoning(no_commit, 6, {P}, S, F), 5)  # last paragraph start before the close

    def test_snap_to_paragraph(self):
        ids = [1, 7, 2, 3, 4, 7, 5]
        self.assertEqual(snap_to_paragraph(ids, 4, {7}), 2)
        self.assertEqual(snap_to_paragraph(ids, 6, {7}), 6)
        self.assertEqual(snap_to_paragraph(ids, 4, {7}, window=1), 4)   # nothing within the window
        # nearest in either direction: paragraph starts at 2 and 6
        self.assertEqual(snap_to_paragraph(ids, 5, {7}, nearest=True), 6)
        self.assertEqual(snap_to_paragraph(ids, 4, {7}, nearest=True), 2)   # tie (2 back, 2 forward) goes back
        self.assertEqual(snap_to_paragraph(ids, 5, {7}, nearest=True, upper=5), 2)   # never past upper

    def test_belief_cuts(self):
        self.assertEqual(belief_cuts(2500, 1024, {'r': 900}),
                         [('end', 2500), ('grid', 1024), ('grid', 2048), ('pair:r', 900)])

    def test_split_thinking(self):
        self.assertEqual(split_thinking([1, 2, 9, 3], 9), 2)
        self.assertEqual(split_thinking([1, 2, 3], 9), 3)   # never closed: all tokens are thinking

    def test_derangement_has_no_fixed_points_and_is_reproducible(self):
        p = derangement(1000, 0)
        self.assertEqual(sorted(p), list(range(1000)))
        self.assertTrue(all(i != j for i, j in enumerate(p)))
        self.assertEqual(p, derangement(1000, 0))

    def test_seed_is_stable(self):
        self.assertEqual(seed_of('a', 1), seed_of('a', 1))
        self.assertNotEqual(seed_of('a', 1), seed_of('a', 2))

    def test_summarize_rates_and_paragraph_starts(self):
        h3 = torch.tensor([.9, .1, .7, .5, .1, .9])
        st = dict(h3=h3, top1=torch.tensor([.2, .9, .3, .5, .9, .1]),
                  p_close=torch.tensor([0., .1, .0, .3, .0, .0]), p_sc=torch.tensor([0., .2, 0., .4, 0., 0.]),
                  p_fa=torch.tensor([0., 0., .5, 0., .1, 0.]))
        ids = [5, 7, 5, 7, 5, 6]               # token 7 ends a paragraph
        s = summarize(st, ids, think=5, para_ids={7})
        self.assertAlmostEqual(s['fork'], 2 / 5)   # positions 0 and 2 (h3 >= .6) within the thinking
        self.assertAlmostEqual(s['lock'], 2 / 5)
        self.assertEqual(s['n_para'], 2)           # positions 2 and 4 follow token 7
        self.assertAlmostEqual(s['para_p_close'], 0.0)
        self.assertAlmostEqual(s['para_p_fa'], .3)   # mean of .5 and .1
        self.assertEqual(len(s['fork_dec']), 10)

    def test_summarize_commit_points(self):
        st = dict(h3=torch.zeros(6), top1=torch.ones(6), p_close=torch.zeros(6),
                  p_sc=torch.tensor([0., 0., .2, 0., 0., 0.]), p_fa=torch.tensor([0., 0., .9, 0., 0., 0.]))
        s = summarize(st, [5, 7, 50, 51, 5, 6], think=5, para_ids={7}, star_id=50, final_id=51)
        self.assertEqual((s['n_commit'], s['first_commit'], s['last_commit']), (1, 2, 2))
        self.assertAlmostEqual(s['commit_p_fa_last'], .9, places=6)
        self.assertAlmostEqual(s['commit_p_sc_first'], .2, places=6)
        self.assertEqual(len(s['para_fa_dec']), 10)

    def test_primary_endpoint_helpers(self):
        from research.e1e4_report import paired_dq, pboot, verdict
        dq = paired_dq({1: .2, 2: .4, 3: .1}, {1: .5, 2: .4, 3: .4, 4: 1.})   # b - a on common problems only
        self.assertEqual(sorted(dq), [1, 2, 3])
        self.assertAlmostEqual(dq[1], .3)
        m, c95, c99, _, _ = pboot({q: .05 for q in range(90)})
        self.assertAlmostEqual(m, .05)
        self.assertAlmostEqual(c99[0], .05)
        self.assertEqual(verdict((.01, .05), (.005, .06), +1), '**supported** (survives Bonferroni)')
        self.assertEqual(verdict((.01, .05), (-.001, .06), +1), 'supported at 95% only')
        self.assertEqual(verdict((-.05, -.01), (-.06, -.005), +1), '**contradicted**')
        self.assertEqual(verdict((-.01, .05), (-.02, .06), -1), 'not resolved')
        self.assertEqual(verdict((-.06, -.01), (-.07, -.005), -1), '**supported** (survives Bonferroni)')


@unittest.skipUnless(TOKENIZER.is_dir(), 'tokenizer not available')
class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(TOKENIZER)

    def test_marker_words_are_single_tokens(self):
        for words in (SELF_CORRECTION_WORDS, CONTRAST_WORDS, CONCLUSION_WORDS, COMMIT_WORDS):
            self.assertEqual(len(single_token_ids(self.tok, words)), len(words))

    def test_exit_text_closes_thinking_once(self):
        ids = self.tok(EXIT_TEXT, add_special_tokens=False).input_ids
        self.assertEqual(ids.count(self.tok.convert_tokens_to_ids(CLOSE)), 1)

    def test_probe_text_matches_the_models_own_answer_format(self):
        own = self.tok('\n\n**Final Answer**\n\\boxed{204}', add_special_tokens=False).input_ids
        probe = self.tok(PROBE_TEXT, add_special_tokens=False).input_ids
        self.assertEqual(own[1:1 + len(probe)], probe)

    def test_wait_splice_keeps_text_and_appends_wait(self):
        from research.forcing import wait_splice
        gen = self.tok('<think>\nSo the answer is 42.\n', add_special_tokens=False).input_ids
        out = self.tok.decode(wait_splice(self.tok, gen), skip_special_tokens=False)
        self.assertTrue(out.startswith('<think>\nSo the answer is 42.'))
        self.assertTrue(out.endswith('42.\n\nWait'))


if __name__ == '__main__':
    unittest.main()
