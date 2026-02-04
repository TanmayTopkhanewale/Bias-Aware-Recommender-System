import torch
from torch.utils.data import DataLoader
from dataset import InteractionDataset
from model import TwoTower

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

dataset = InteractionDataset("interactions_encoded.csv")
loader = DataLoader(dataset, batch_size=1024, shuffle=True)

num_users = int(dataset.users.max()) + 1
num_items = int(dataset.items.max()) + 1

model = TwoTower(num_users, num_items, dim=64).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

for epoch in range(5):
    total_loss = 0
    for u, i, w in loader:
        u, i, w = u.to(DEVICE), i.to(DEVICE), w.to(DEVICE)

        pred = model(u, i)
        loss = loss_fn(pred, w)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(u)

    print(f"Epoch {epoch+1} | Loss: {total_loss / len(dataset):.4f}")
