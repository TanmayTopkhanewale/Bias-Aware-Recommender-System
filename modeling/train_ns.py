import torch
from torch.utils.data import DataLoader
from dataset_ns import InteractionDatasetNS
from model import TwoTower
from loss_contrastive import contrastive_loss
import pandas as pd
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("interactions_encoded.csv")

NUM_USERS = df.user_idx.max() + 1
NUM_ITEMS = df.item_idx.max() + 1

dataset = InteractionDatasetNS(
    "interactions_encoded.csv",
    num_items=NUM_ITEMS,
    num_negatives=5
)

loader = DataLoader(dataset, batch_size=1024, shuffle=True)

model = TwoTower(NUM_USERS, NUM_ITEMS, dim=64).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    total_loss = 0

    for u, pos, negs in loader:
        u, pos, negs = u.to(DEVICE), pos.to(DEVICE), negs.to(DEVICE)

        u_vec = model.user_emb(u)
        pos_vec = model.item_emb(pos)
        neg_vecs = model.item_emb(negs)

        loss = contrastive_loss(u_vec, pos_vec, neg_vecs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(u)

    print(f"Epoch {epoch+1} | Loss: {total_loss / len(dataset):.4f}")
import os
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("interactions_encoded.csv")

NUM_USERS = df.user_idx.max() + 1
NUM_ITEMS = df.item_idx.max() + 1

dataset = InteractionDatasetNS(
    "interactions_encoded.csv",
    num_items=NUM_ITEMS,
    num_negatives=5
)

loader = DataLoader(dataset, batch_size=1024, shuffle=True)

model = TwoTower(NUM_USERS, NUM_ITEMS, dim=64).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    total_loss = 0

    for u, pos, negs in loader:
        u, pos, negs = u.to(DEVICE), pos.to(DEVICE), negs.to(DEVICE)

        u_vec = model.user_emb(u)
        pos_vec = model.item_emb(pos)
        neg_vecs = model.item_emb(negs)

        loss = contrastive_loss(u_vec, pos_vec, neg_vecs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(u)

    print(f"Epoch {epoch+1} | Loss: {total_loss / len(dataset):.4f}")

import os
os.makedirs("artifacts", exist_ok=True)

torch.save(model.user_emb.weight.detach().cpu(), "artifacts/user_embeddings.pt")
torch.save(model.item_emb.weight.detach().cpu(), "artifacts/item_embeddings.pt")
