import torch
import torch.nn.functional as F


def hard_negative_infonce(eeg_emb, clip_emb, tau=0.07, hard_k=48):
    """
    Bidirectional InfoNCE with hard negative mining.

    Args:
        eeg_emb:  (B, D) normalized EEG embeddings
        clip_emb: (B, D) normalized CLIP image embeddings
        tau:      temperature
        hard_k:   number of hard negatives per sample
    """
    B      = eeg_emb.size(0)
    logits = eeg_emb @ clip_emb.T / tau
    labels = torch.arange(B, device=eeg_emb.device)

    with torch.no_grad():
        sim = eeg_emb @ clip_emb.T
        sim.fill_diagonal_(-1e9)
        k        = min(hard_k, B - 1)
        hard_idx = sim.topk(k, dim=1).indices

    mask = torch.zeros(B, B, dtype=torch.bool, device=eeg_emb.device)
    mask.scatter_(1, hard_idx, True)
    mask[range(B), labels] = True

    logits = logits.masked_fill(~mask, -1e9)

    loss_e2i = F.cross_entropy(logits,   labels)
    loss_i2e = F.cross_entropy(logits.T, labels)
    return (loss_e2i + loss_i2e) / 2
