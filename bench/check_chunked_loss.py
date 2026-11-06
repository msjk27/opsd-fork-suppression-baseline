"""GPU equivalence check: DistilTrainer._compute_loss (upstream) vs chunked_loss path.

Runs both loss implementations on the same real-model inputs without building the full
trainer (no FSDP/vLLM), then compares loss values, logged metrics, and parameter grads.
Usage: python bench/check_chunked_loss.py MODEL_PATH OUT_JSON
"""
import json, sys, time
from collections import defaultdict
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, ".")
from opsd.trainer import DistilTrainer  # noqa: E402

model_path, out_path = sys.argv[1], sys.argv[2]
dev = "cuda:0"
torch.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
model.config.use_cache = False
model.train()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


def fake_trainer(chunked, alpha, chunk=512):
    t = SimpleNamespace(
        args=SimpleNamespace(research_mask="vanilla", research_diagnostics=False, chunked_loss=chunked,
                             chunked_loss_size=chunk, seed=42, report_to=[]),
        num_loss_tokens_to_skip=3, model_kwarg_keys={"logits_to_keep"}, temperature=1.0,
        teacher_temperature=None, ref_model=model, top_entropy_quantile=1.0, beta=0.0, alpha=alpha,
        use_vllm=True, vllm_importance_sampling_correction=True, generate_from_teacher=False,
        current_gradient_accumulation_steps=8, model=model,
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
        accelerator=SimpleNamespace(gather=lambda x: x, is_main_process=True))
    for name in ("_get_per_token_logps_and_entropies", "_completion_logits", "_compute_loss_chunked"):
        setattr(t, name, getattr(DistilTrainer, name).__get__(t))
    return t


def make_inputs(C, B=2):
    g = torch.Generator().manual_seed(1)
    r = lambda n: torch.randint(0, 150000, (B, n), generator=g).to(dev)
    cmask = torch.ones(B, C, dtype=torch.long, device=dev)
    if B > 1:
        cmask[1, C - C // 4:] = 0  # second row ends early (padding)
    return {"prompt_ids": r(120), "prompt_mask": torch.ones(B, 120, dtype=torch.long, device=dev),
            "completion_ids": r(C), "completion_mask": cmask,
            "teacher_prompt_ids": r(700), "teacher_prompt_mask": torch.ones(B, 700, dtype=torch.long, device=dev),
            "importance_sampling_ratio": (0.8 + 0.4 * torch.rand(B, C, generator=g)).to(dev)}


def run(chunked, alpha, inputs, chunk=512):
    t = fake_trainer(chunked, alpha, chunk)
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.time()
    loss = DistilTrainer._compute_loss(t, model, inputs)
    loss.backward()
    torch.cuda.synchronize()
    grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters()
             if p.grad is not None and ("layers.27" in n or "layers.0." in n or "embed" in n)}
    return dict(loss=loss.item(), metrics={k: v[0] for k, v in t._metrics["train"].items()},
                peak_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                sec=round(time.time() - t0, 2)), grads


res = {}
inputs = make_inputs(1024)
def rel(a, b):
    return {n: round(((a[n] - b[n]).norm() / a[n].norm().clamp(min=1e-12)).item(), 4) for n in a}


for alpha in (0.5, 0.0, 1.0):
    up, gu = run(False, alpha, inputs)
    up2, gu2 = run(False, alpha, inputs)  # noise floor: upstream vs itself
    ch, gc = run(True, alpha, inputs)
    r_uc, r_uu = rel(gu, gc), rel(gu, gu2)
    res[f"alpha{alpha}_C1024"] = dict(upstream=up, chunked=ch, loss_abs_diff=abs(up["loss"] - ch["loss"]),
                                      max_rel_grad_diff_vs_chunked=max(r_uc.values()),
                                      max_rel_grad_diff_upstream_rerun=max(r_uu.values()),
                                      worst_tensors={n: (r_uc[n], r_uu[n]) for n in sorted(r_uc, key=r_uc.get)[-4:]})
    print(f"alpha={alpha}", json.dumps(res[f"alpha{alpha}_C1024"]), flush=True)

big = make_inputs(4096, B=1)
ch, _ = run(True, 0.5, big)
res["chunked_C4096_B1"] = ch
print("chunked C=4096", ch, flush=True)
try:
    up, _ = run(False, 0.5, big)
    res["upstream_C4096_B1"] = up
except torch.OutOfMemoryError:
    res["upstream_C4096_B1"] = {"oom": True}
print("upstream C=4096", res["upstream_C4096_B1"], flush=True)
json.dump(res, open(out_path, "w"), indent=2)
