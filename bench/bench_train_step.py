"""Phase-1 single-GPU memory/time probe for one OPSD microbatch (no optimizer step).

Student: prompt P_s + completion C (with grad). Teacher: prompt P_t + same C (no grad).
Loss: generalized JSD alpha=0.5 over the full vocabulary on every completion position.
  upstream : materialize [C, V] log-probs for student and teacher (as opsd/trainer.py does)
  chunked  : keep last hidden states, compute lm_head + JSD per 512-token chunk under
             activation checkpointing (mathematically the same loss)
Modes: lora (r=64, all linear; teacher = adapter disabled) and full (all params trainable;
       optimizer state NOT allocated -> lower bound for FSDP-free full FT).
"""
import json, math, sys, time
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM

model_path, out_path = sys.argv[1], sys.argv[2]
P_S, P_T, C, ALPHA, CHUNK = 200, 900, 4096, 0.5, 512
dev = "cuda:0"
torch.manual_seed(42)


def jsd(s, t):  # s, t: log-probs [..., V]
    a = torch.tensor(ALPHA, dtype=s.dtype, device=s.device)
    mix = torch.logsumexp(torch.stack([s + torch.log(1 - a), t + torch.log(a)]), dim=0)
    return (a * F.kl_div(mix, t, reduction="none", log_target=True)
            + (1 - a) * F.kl_div(mix, s, reduction="none", log_target=True)).sum(-1)


def build(mode):
    m = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                                             attn_implementation="sdpa").to(dev)
    m.config.use_cache = False
    m.train()  # HF applies gradient checkpointing only in training mode (Qwen3 has no dropout)
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if mode == "lora":
        from peft import LoraConfig, get_peft_model
        m.enable_input_require_grads()
        m = get_peft_model(m, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    m.train()  # HF gradient checkpointing is only active in train mode (Qwen3 has no dropout)
    return m


def teacher_ctx(m, mode):
    return m.disable_adapter() if mode == "lora" else torch.autocast("cuda", enabled=False)


def step(m, mode, loss_kind, s_ids, t_ids):
    body = m.get_base_model() if mode == "lora" else m
    lm_head = body.lm_head
    if loss_kind == "upstream":
        s = F.log_softmax(m(input_ids=s_ids, logits_to_keep=C + 1).logits[:, :-1], -1)
        with torch.no_grad(), teacher_ctx(m, mode):
            t = F.log_softmax(m(input_ids=t_ids, logits_to_keep=C + 1).logits[:, :-1], -1)
        loss = jsd(s, t).mean()
    else:
        hs = body.model(input_ids=s_ids).last_hidden_state[:, -C - 1:-1]
        with torch.no_grad(), teacher_ctx(m, mode):
            ht = body.model(input_ids=t_ids).last_hidden_state[:, -C - 1:-1]

        def chunk_loss(h_s, h_t):
            s = F.log_softmax(lm_head(h_s).float(), -1)
            with torch.no_grad():
                t = F.log_softmax(lm_head(h_t).float(), -1)
            return jsd(s, t).sum()
        loss = sum(checkpoint(chunk_loss, hs[:, i:i + CHUNK], ht[:, i:i + CHUNK], use_reentrant=False)
                   for i in range(0, C, CHUNK)) / C
    loss.backward()
    return loss.item()


results = {}
for mode in ("lora", "full"):
    m = build(mode)
    comp = torch.randint(0, 150000, (1, C), device=dev)
    s_ids = torch.cat([torch.randint(0, 150000, (1, P_S), device=dev), comp], 1)
    t_ids = torch.cat([torch.randint(0, 150000, (1, P_T), device=dev), comp], 1)
    for loss_kind in ("chunked", "upstream"):
        key = f"{mode}_{loss_kind}"
        try:
            m.zero_grad(set_to_none=True)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            step(m, mode, loss_kind, s_ids, t_ids)  # warmup
            m.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(2):
                step(m, mode, loss_kind, s_ids, t_ids)
            torch.cuda.synchronize()
            results[key] = dict(sec_per_microbatch=round((time.time() - t0) / 2, 2),
                                peak_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        except torch.OutOfMemoryError:
            results[key] = dict(oom=True, peak_GiB_before_oom=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        print(key, results[key], flush=True)
    del m; torch.cuda.empty_cache()

results["setup"] = dict(P_student=P_S, P_teacher=P_T, completion=C, alpha=ALPHA, chunk=CHUNK,
                        note="no optimizer state allocated; full FT Adam fp32 would add ~16 B/param unsharded")
json.dump(results, open(out_path, "w"), indent=2)
