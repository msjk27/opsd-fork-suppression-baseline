import ast
import argparse
import os
import sys
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn.functional import kl_div
from research.metrics import fork_lock, accuracy_metrics, response_metrics
from research.masking import loss_mask
from research.signals import token_signals, write_positions
from research.analyze import aggregate
from research.run import load_config

ROOT = Path(__file__).resolve().parents[1]


class ResearchTests(unittest.TestCase):
    def test_paper_strict_boundaries(self):
        self.assertEqual(fork_lock(.25), (False, False))
        self.assertEqual(fork_lock(.65), (False, False))
        self.assertEqual(fork_lock(.24999), (True, False))
        self.assertEqual(fork_lock(.65001), (False, True))

    def test_mask_counts_padding_seed(self):
        valid = torch.tensor([[0, 1, 1, 1, 0], [1, 1, 0, 0, 0]])
        s = torch.tensor([[.1, .1, .2, .8, .1], [.1, .8, .1, .1, .1]])
        f = loss_mask(valid, s, s, 'fork')
        r = loss_mask(valid, s, s, 'random', 99)
        self.assertTrue(torch.equal(f.sum(-1), r.sum(-1)))
        self.assertTrue(torch.equal(r, loss_mask(valid, s, s, 'random', 99)))
        self.assertTrue(torch.all(r <= valid))
        self.assertIs(loss_mask(valid, s, s), valid)
        self.assertTrue(torch.equal(valid, torch.tensor([[0,1,1,1,0],[1,1,0,0,0]])))

    def test_all_masked_differentiable_zero(self):
        x = torch.tensor([[1., 2.]], requires_grad=True)
        mask = loss_mask(torch.ones_like(x), x*.01, x*.01, 'fork')
        loss = ((x*mask).sum(-1)/mask.sum(-1).clamp(min=1)).mean()
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(torch.equal(x.grad, torch.zeros_like(x)))

    def test_metrics_known_combinatorial_example(self):
        rs = [dict(question_id=0, sample_id=i, is_correct=(i==0)) for i in range(4)]
        m = accuracy_metrics(rs)
        self.assertEqual(m['avg@2'], .25)
        self.assertEqual(m['pass@2'], .5)
        self.assertEqual(m['pass@4'], 1)
        self.assertIsNone(m['pass@8'])
        with self.assertRaises(ValueError):
            accuracy_metrics(rs+[rs[0]])
        with self.assertRaises(ValueError):
            accuracy_metrics([dict(question_id=0,is_correct='false')])

    def test_eos_cap(self):
        self.assertFalse(response_metrics([5,2], '', [2], 2)['cap_hit'])
        self.assertTrue(response_metrics([5,6], '', [2], 2)['cap_hit'])
        self.assertEqual(response_metrics([5,2,2], '', [2], 3)['response_tokens'], 2)

    def test_signals_alignment_and_position_serialization(self):
        s = torch.tensor([[[.2,.8],[.7,.3]]]).log()
        t = torch.tensor([[[.8,.2],[.4,.6]]]).log()
        ids = torch.tensor([[0,1]])
        signals = token_signals(s,t,ids,chunk_size=1)
        self.assertAlmostEqual(signals['log_ratio'][0,0].item(), 1.38629436, places=5)
        self.assertTrue(torch.all(signals['jsd'] >= 0))
        self.assertTrue(torch.allclose(token_signals(s,s,ids)['jsd'], torch.zeros(1,2), atol=1e-7))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'positions.jsonl'
            write_positions(path, signals, torch.tensor([[1,0]]), ids, {'condition':'dense'})
            rows = [json.loads(l) for l in path.read_text().splitlines()]
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['position'],0)
            self.assertEqual(aggregate(rows)['macro_rollout_rates']['dense']['lock'],1)

    def test_every_config_matches_cli(self):
        for p in (ROOT/'configs/research').glob('*4096.json'):
            load_config(p)
        load_config(ROOT/'configs/research/vanilla_1024.json')

    def test_real_cli_parsers(self):
        for filename, argv in [
            ('train.py', ['--train_file_path','fixture.parquet','--output_dir','fixture',
                          '--research_mask','fork','--research_diagnostics','true']),
            ('run_generate.py', ['--model_name','fixture','--dataset','aime24',
                                 '--output_dir','fixture','--seed','42','--research_metrics'])]:
            tree = ast.parse((ROOT/filename).read_text(encoding='utf-8'))
            funcs = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('parse_bool','parse_args')]
            ns = {'argparse':argparse,'os':os}
            exec(compile(ast.Module(body=funcs,type_ignores=[]), filename,'exec'),ns)
            with patch.object(sys,'argv',[filename]+argv):
                parsed = ns['parse_args']()
            if filename == 'train.py':
                self.assertTrue(parsed.research_diagnostics)
            else:
                self.assertEqual(parsed.seed,42)

    def test_upstream_loss_and_gradient_compatibility(self):
        # Execute actual _compute_loss bodies without importing the GPU/TRL stack.
        upstream = subprocess.check_output(['git','show','46ccfbd:opsd/trainer.py'], cwd=ROOT).decode()
        modified = (ROOT/'opsd/trainer.py').read_text(encoding='utf-8')
        def extract(source):
            tree = ast.parse(source)
            fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name=='_compute_loss')
            ns = {'torch':torch,'kl_div':kl_div}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), '<actual loss>', 'exec'), ns)
            return ns['_compute_loss']
        old, new = extract(upstream), extract(modified)
        inputs = {'prompt_ids':torch.ones(1,2,dtype=torch.long), 'prompt_mask':torch.ones(1,2),
                  'teacher_prompt_ids':torch.ones(1,3,dtype=torch.long), 'teacher_prompt_mask':torch.ones(1,3),
                  'completion_ids':torch.tensor([[1,0,2]]),'completion_mask':torch.tensor([[1,1,0]])}
        for alpha in (0.,.5,1.):
            for mode in ('vanilla','fork','random'):
                grads, losses = [], []
                for fn in (old,new):
                    raw = torch.tensor([[[1.,2.,3.],[2.,1.,0.],[0.,1.,2.]]],requires_grad=True)
                    teacher = torch.tensor([[[3.,2.,1.],[0.,1.,2.],[1.,0.,2.]]]).log_softmax(-1)
                    model = SimpleNamespace(training=True)
                    obj = SimpleNamespace(num_loss_tokens_to_skip=1,ref_model=object(),teacher_temperature=1.,
                        top_entropy_quantile=1., beta=0.,alpha=alpha,use_vllm=False,
                        current_gradient_accumulation_steps=8,model=model,_metrics={'train':defaultdict(list)},
                        args=SimpleNamespace(research_mask=mode,research_diagnostics=False,seed=42),
                        state=SimpleNamespace(global_step=0),accelerator=SimpleNamespace(gather=lambda x:x,process_index=0))
                    def logps(m,*args,**kw):
                        lp = raw.log_softmax(-1) if m is model else teacher
                        return lp.gather(-1,inputs['completion_ids'][...,None]).squeeze(-1),lp,-(lp.exp()*lp).sum(-1)
                    obj._get_per_token_logps_and_entropies=logps
                    loss = fn(obj,model,inputs)
                    loss.backward()
                    losses.append(loss.detach()); grads.append(raw.grad)
                # This fixture contains no student forks; all modes keep the same valid tokens.
                self.assertTrue(torch.equal(losses[0], losses[1]))
                self.assertTrue(torch.equal(grads[0], grads[1]))


if __name__ == '__main__':
    unittest.main()
