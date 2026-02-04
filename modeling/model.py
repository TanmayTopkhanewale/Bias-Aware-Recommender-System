import torch
import torch.nn as nn

class TwoTower(nn.Module):
    def __init__(self, num_users, num_items, dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, u, i):
        return (self.user_emb(u) * self.item_emb(i)).sum(dim=1)