import torch
from torch.utils.data import Dataset
import pandas as pd

class InteractionDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        self.users = torch.tensor(df.user_idx.values, dtype=torch.long)
        self.items = torch.tensor(df.item_idx.values, dtype=torch.long)
        self.weights = torch.tensor(df.interaction_weight.values, dtype=torch.float)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.weights[idx]
