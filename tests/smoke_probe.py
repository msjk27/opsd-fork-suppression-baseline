"""Opt-in, download-free tiny Qwen3 integration smoke test.

Run: python -m tests.smoke_probe (requires transformers/datasets in the training env).
Random weights are synthetic fixtures; no research accuracy is measured.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    import pandas as pd
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        model_dir = root/'model'
        vocab = {'[UNK]':0,'[PAD]':1,'[EOS]':2,'question':3,'answer':4,'<think>':5,'</think>':6,'wait':7}
        raw = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
        raw.pre_tokenizer = Whitespace()
        tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]',pad_token='[PAD]', eos_token='[EOS]')
        tok.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}{% if enable_thinking %}<think>{% endif %}"
        tok.save_pretrained(model_dir)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=8, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=16,
                     eos_token_id=2,pad_token_id=1,bos_token_id=None))
        model.save_pretrained(model_dir)
        data = root/'data.parquet'
        pd.DataFrame([{'problem':'question','Answer':'answer','solution':'answer wait'}]).to_parquet(data)
        c = {'student':str(model_dir),'teacher':str(model_dir),'student_device':'cpu','teacher_device':'cpu',
             'data':str(data),'output_dir':str(root/'output'),'seed':42,'limit':1,'samples':2,
             'max_tokens':4,'max_prompt_length':64,'temperature':1.,'top_p':1.,
             'teacher_prompt_template':'{prompt} example {gold_answer}'}
        config = root/'config.json'
        config.write_text(json.dumps(c),encoding='utf-8')
        subprocess.run([sys.executable,'-m','research.probe','--config',str(config)],check=True)
        subprocess.run([sys.executable,'-m','research.analyze','--positions',str(root/'output/positions.jsonl'),
                        '--responses',str(root/'output/responses.jsonl'),'--output',str(root/'metrics.json')],check=True)
        m = json.loads((root/'metrics.json').read_text())
        assert set(m['diagnostics']['macro_rollout_rates']) == {'base','sparse','dense'}
        assert len(m['diagnostics']['rollouts']) == 6
        assert (root/'output/config.json').is_file()
        print('PASS: tiny Qwen3 generation -> paired teacher scores -> JSONL -> aggregate (2 samples, 3 conditions)')


if __name__ == '__main__':
    main()
