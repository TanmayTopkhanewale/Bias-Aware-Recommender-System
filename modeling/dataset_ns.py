import torch
from torch.utils.data import Dataset
import pandas as pd
import random

class InteractionDatasetNS(Dataset):
    def __init__(self, path, num_items, num_negatives=5):
        df = pd.read_csv(path)

        self.users = df.user_idx.values
        self.pos_items = df.item_idx.values
        self.num_items = num_items
        self.num_negatives = num_negatives

        # Fast lookup: user -> set of interacted items
        self.user_item_set = {}
        for u, i in zip(self.users, self.pos_items):
            self.user_item_set.setdefault(u, set()).add(i)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        pos = self.pos_items[idx]

        negs = []
        while len(negs) < self.num_negatives:
            neg = random.randint(0, self.num_items - 1)
            if neg not in self.user_item_set[u]:
                negs.append(neg)

        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(negs, dtype=torch.long),
        )
