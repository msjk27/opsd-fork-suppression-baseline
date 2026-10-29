"""Position-wise full-distribution diagnostics on identical prefixes."""
import torch


@torch.no_grad()
def token_signals(student, teacher, sampled_ids, chunk_size=32):
    """Inputs are normalized log probabilities [B,C,V]; output [B,C] tensors."""
    result = {k: [] for k in ('teacher_top', 'student_top', 'teacher_entropy',
                             'student_entropy', 'argmax_disagree', 'log_ratio', 'jsd')}
    for start in range(0, student.shape[1], chunk_size):
        s, t = student[:, start:start+chunk_size].float(), teacher[:, start:start+chunk_size].float()
        p, q = s.exp(), t.exp()
        mix = torch.logaddexp(s, t) - 0.6931471805599453
        ids = sampled_ids[:, start:start+chunk_size, None]
        values = (q.amax(-1), p.amax(-1), -(q*t).sum(-1), -(p*s).sum(-1),
                  s.argmax(-1).ne(t.argmax(-1)), (t-s).gather(-1, ids).squeeze(-1),
                  0.5*((p*(s-mix)).sum(-1)+(q*(t-mix)).sum(-1)))
        for key, value in zip(result, values):
            result[key].append(value.cpu())
    return {k: torch.cat(v, dim=1) for k, v in result.items()}


def write_positions(path, signals, valid, token_ids, metadata):
    import json
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    valid, token_ids = valid.detach().cpu(), token_ids.detach().cpu()
    with path.open('a', encoding='utf-8') as f:
        for b in range(valid.shape[0]):
            for pos in valid[b].bool().nonzero().flatten().tolist():
                row = dict(metadata, batch_row=b, position=pos, token_id=int(token_ids[b, pos]))
                row.update({k: v[b, pos].item() for k, v in signals.items()})
                row.update(fork=row['teacher_top'] < .25, lock=row['teacher_top'] > .65)
                f.write(json.dumps(row, allow_nan=False)+'\n')
