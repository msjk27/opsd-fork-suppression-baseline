"""Loss-only masking. Never alter attention or the generated trajectory."""
import torch


def loss_mask(valid, teacher_top, student_top, mode='vanilla', seed=42):
    if mode not in ('vanilla', 'fork', 'random'):
        raise ValueError(mode)
    if mode == 'vanilla':
        return valid
    # Team hypothesis: preserve alternatives where the unprivileged student forks.
    # This is NOT the paper's teacher-defined fork diagnostic.
    candidates = (student_top < 0.25) & valid.bool()
    if mode == 'fork':
        return valid * (~candidates).to(valid.dtype)
    result = valid.clone()
    gen = torch.Generator(device=valid.device).manual_seed(seed)
    for row in range(valid.shape[0]):
        indices = valid[row].bool().nonzero().flatten()
        count = int(candidates[row].sum())
        selected = indices[torch.randperm(len(indices), generator=gen, device=valid.device)[:count]]
        result[row, selected] = 0
    return result
