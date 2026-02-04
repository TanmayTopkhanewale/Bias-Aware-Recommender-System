import torch
import torch.nn.functional as F

def contrastive_loss(user_vec, pos_item_vec, neg_item_vecs):
    """
    user_vec: (B, D)
    pos_item_vec: (B, D)
    neg_item_vecs: (B, K, D)
    """

    pos_scores = (user_vec * pos_item_vec).sum(dim=1)  # (B,)
    neg_scores = torch.bmm(
        neg_item_vecs, user_vec.unsqueeze(2)
    ).squeeze(2)  # (B, K)

    logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

    return F.cross_entropy(logits, labels)
