"""Memory-bounded generalized JSD for OPSD (same math as DistilTrainer._compute_loss).

The upstream loss materializes several [B, C, V] full-vocabulary tensors at once
(log-probs for student and teacher, the stacked mixture, two KL terms, plus
autograd copies). With C=4096 and V≈152k that is ~1.2 GB each in bf16 and does not
fit next to full fine-tuning on a 24 GB GPU.

Here only the raw student/teacher logits are kept at full size. Each slice of
``chunk`` completion positions runs the exact upstream op sequence (temperature
division, log_softmax, mixture, kl_div, sum over vocab) under activation
checkpointing, so its intermediates are freed after the forward and recomputed in
backward. Every op is row-wise over the vocabulary, so per-token results match the
unchunked computation.
"""
import torch
from torch.nn.functional import kl_div, log_softmax
from torch.utils.checkpoint import checkpoint
from trl.trainer.utils import entropy_from_logits, selective_log_softmax


def _jsd_chunk(s_logits, t_logits, alpha, s_temp, t_temp):
    """Per-token generalized JSD for one chunk; mirrors the upstream expression."""
    s_logps = log_softmax(s_logits / s_temp, dim=-1)
    t_logps = log_softmax(t_logits / t_temp, dim=-1)
    if alpha == 0:  # forward KL
        kl_loss = kl_div(s_logps, t_logps, reduction="none", log_target=True)
    elif alpha == 1:  # reverse KL
        kl_loss = kl_div(t_logps, s_logps, reduction="none", log_target=True)
    else:
        a = torch.tensor(alpha, dtype=s_logps.dtype)
        mixture = torch.logsumexp(torch.stack([s_logps + torch.log(1 - a), t_logps + torch.log(a)]), dim=0)
        kl_teacher = kl_div(mixture, t_logps, reduction="none", log_target=True)
        kl_student = kl_div(mixture, s_logps, reduction="none", log_target=True)
        kl_loss = a * kl_teacher + (1 - a) * kl_student
    return kl_loss.sum(-1)


def chunked_jsd(student_logits, teacher_logits, completion_ids, alpha, student_temperature,
                teacher_temperature, chunk=512):
    """Return (per_token_loss, student_token_logps, teacher_token_logps, student_entropy).

    ``student_logits`` [B, C, V] carries grad; ``teacher_logits`` [B, C, V] does not.
    Only ``per_token_loss`` is differentiable; the other outputs are for logging.
    """
    losses, s_sel, t_sel, ents = [], [], [], []
    for i in range(0, completion_ids.size(1), chunk):
        s = student_logits[:, i:i + chunk]
        t = teacher_logits[:, i:i + chunk]
        ids = completion_ids[:, i:i + chunk]
        losses.append(checkpoint(_jsd_chunk, s, t, alpha, student_temperature, teacher_temperature,
                                 use_reentrant=False))
        with torch.no_grad():
            s_scaled = s.detach() / student_temperature
            s_sel.append(selective_log_softmax(s_scaled, ids))
            t_sel.append(selective_log_softmax(t / teacher_temperature, ids))
            ents.append(entropy_from_logits(s_scaled))
    return torch.cat(losses, 1), torch.cat(s_sel, 1), torch.cat(t_sel, 1), torch.cat(ents, 1)
