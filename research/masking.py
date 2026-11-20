"""Loss-only masking. Never alter attention or the generated trajectory."""
from bisect import bisect_left

import torch

# E6. A decision position is a paragraph start inside the thinking: the same positions E1-C and E4 measure
# (where the model chooses between going on, self-correcting and committing to an answer).
#   decision          drop the decision positions from the loss
#   decision_matched  keep them; drop as many other thinking positions, matched one to one on their current loss
#                     (same token count and same loss mass removed: the control for "less signal")
#   decision_random   keep them; drop as many other thinking positions at random (same count only)
DECISION_MODES = ('decision', 'decision_matched', 'decision_random')


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


def token_lookup(token_ids, vocab_size, device=None):
    """Boolean table over token ids. The extra last entry (False) stands for any id beyond the tokenizer."""
    table = torch.zeros(vocab_size + 1, dtype=torch.bool, device=device)
    table[torch.tensor(sorted(token_ids), dtype=torch.long, device=device)] = True
    return table


def thinking_positions(completion_ids, close_id):
    """True before the first closing think tag (everywhere if the rollout never closes it)."""
    return (completion_ids == close_id).long().cumsum(1) == 0


def decision_positions(completion_ids, valid, paragraph_lookup, close_id):
    """Loss-eligible positions j inside the thinking whose previous token ends a blank line. The loss at j
    trains the prediction of token j, so this is the choice of how the new paragraph opens."""
    previous = torch.zeros_like(completion_ids, dtype=torch.bool)
    ids = completion_ids[:, :-1].clamp(max=paragraph_lookup.numel() - 1)
    previous[:, 1:] = paragraph_lookup[ids]
    return previous & thinking_positions(completion_ids, close_id) & valid.bool()


def match_by_value(target_values, candidates, candidate_values):
    """For each target (largest first) take the unused candidate with the closest value. Candidates must come
    in random order: the stable sort then breaks ties (common in bf16 losses) at random."""
    order = sorted(range(len(candidates)), key=candidate_values.__getitem__)
    values = [candidate_values[i] for i in order]
    items = [candidates[i] for i in order]
    picked = []
    for v in sorted(target_values, reverse=True):
        if not items:
            break
        k = bisect_left(values, v)
        if k == len(values) or (k > 0 and v - values[k - 1] <= values[k] - v):
            k -= 1
        picked.append(items.pop(k))
        values.pop(k)
    return picked


@torch.no_grad()
def decision_mask(completion_ids, valid, paragraph_lookup, close_id, mode, per_token_loss, seed):
    """(mask for the loss, statistics). Controls choose within the same rollout and the same region (thinking),
    never touch a decision position, and use their own generator, so global RNG streams are unaffected."""
    if mode not in DECISION_MODES:
        raise ValueError(mode)
    eligible = valid.bool()
    decision = decision_positions(completion_ids, valid, paragraph_lookup, close_id)
    loss = per_token_loss.detach().float()
    if mode == 'decision':
        drop = decision
    else:
        drop = torch.zeros_like(decision)
        pool = thinking_positions(completion_ids, close_id) & eligible & ~decision
        gen = torch.Generator(device='cpu').manual_seed(seed)
        loss_cpu = loss.cpu()
        for row in range(valid.shape[0]):
            targets = decision[row].nonzero().flatten().cpu()
            others = pool[row].nonzero().flatten().cpu()
            if not len(targets) or not len(others):
                continue
            others = others[torch.randperm(len(others), generator=gen)]
            if mode == 'decision_random':
                picked = others[:len(targets)].tolist()
            else:
                picked = match_by_value(loss_cpu[row, targets].tolist(), others.tolist(),
                                        loss_cpu[row, others].tolist())
            drop[row, torch.tensor(picked, dtype=torch.long, device=drop.device)] = True
    total_tokens = eligible.sum().clamp(min=1).float()
    total_loss = (loss * eligible).sum().clamp(min=1e-12)
    stats = {'dropped_frac': drop.sum() / total_tokens, 'dropped_loss_share': (loss * drop).sum() / total_loss,
             'decision_frac': decision.sum() / total_tokens, 'decision_loss_share': (loss * decision).sum() / total_loss}
    return valid * (~drop).to(valid.dtype), stats
