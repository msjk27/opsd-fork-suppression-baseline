"""Phase-1 vLLM throughput benchmark on ONE RTX 3090 (short, inference only).

A) training-rollout regime: T=1.0, top_k=50, cap 4096, concurrency 8 (one rank's
   per-step batch) and 64.
B) long-eval regime: T=0.6, top_p=0.95, 16 sequences forced to 16384 tokens
   (ignore_eos) to measure KV-bound decode speed as context grows.
"""
import json, sys, time
from vllm import LLM, SamplingParams


def main():
    model, data, out = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = json.load(open(data))
    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.9,
              max_model_len=40960, seed=42)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["problem"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True) for r in rows]
    res = {}

    def run(name, ps, sp):
        t = time.time()
        outs = llm.generate(ps, sp, use_tqdm=False)
        dt = time.time() - t
        lens = [len(o.outputs[0].token_ids) for o in outs]
        fin = [o.outputs[0].finish_reason for o in outs]
        res[name] = dict(n=len(ps), seconds=round(dt, 1), gen_tokens=sum(lens),
                         tok_per_s=round(sum(lens) / dt, 1), mean_len=round(sum(lens) / len(lens)),
                         cap_hit_frac=round(sum(f == "length" for f in fin) / len(fin), 3))
        print(name, res[name], flush=True)

    train_sp = SamplingParams(temperature=1.0, top_p=1.0, top_k=50, max_tokens=4096, seed=42)
    run("A_train_c8", prompts[:8], train_sp)
    run("A_train_c64", prompts, train_sp)
    long_sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=-1, max_tokens=16384,
                             ignore_eos=True, seed=42)
    run("B_long16k_c16", prompts[:16], long_sp)
    json.dump(res, open(out, "w"), indent=2)


if __name__ == "__main__":  # vLLM spawns workers that re-import this file
    main()
