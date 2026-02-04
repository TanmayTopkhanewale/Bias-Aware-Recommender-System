import torch
import pandas as pd
import numpy as np
from ingestion.db import get_connection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load embeddings
user_emb = torch.load("artifacts/user_embeddings.pt").to(DEVICE)
item_emb = torch.load("artifacts/item_embeddings.pt").to(DEVICE)

# Load mappings (cleanly)
user_map = pd.read_csv("data/user2idx.csv", header=None, names=["user_id", "user_idx"])
item_map = pd.read_csv("data/item2idx.csv", header=None, names=["business_id", "item_idx"])

# Drop bad rows
user_map = user_map.dropna()
item_map = item_map.dropna()

user2idx = dict(zip(user_map.user_id, user_map.user_idx))
idx2item = dict(zip(item_map.item_idx, item_map.business_id))


def random_user_id():
    # Sample a random user_id from the mapping
    return user_map.user_id.sample(1).iloc[0]


def fetch_business_details(business_ids):
    if not business_ids:
        return pd.DataFrame()

    query = """
        SELECT business_id, name, categories, city, state, stars
        FROM businesses
        WHERE business_id = ANY(%s)
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (business_ids,))
            rows = cur.fetchall()

    details = pd.DataFrame(
        rows,
        columns=["business_id", "name", "categories", "city", "state", "stars"],
    )
    # Preserve recommendation order
    order = {bid: i for i, bid in enumerate(business_ids)}
    details["_order"] = details["business_id"].map(order)
    details = details.sort_values("_order").drop(columns=["_order"])
    return details


def recommend_for_user(user_id, k=10):
    if user_id not in user2idx:
        raise ValueError("User not found")

    uidx = int(user2idx[user_id])
    uvec = user_emb[uidx]

    scores = torch.matmul(item_emb, uvec)
    topk = torch.topk(scores, k)

    return [idx2item[int(i)] for i in topk.indices]
    

# -------- Example --------
if __name__ == "__main__":
    sample_user = random_user_id()
    # pick any trained user
    recs = recommend_for_user(sample_user, k=3)

    print("User:", sample_user)
    print("Recommended businesses:")

    details = fetch_business_details(recs)
    if details.empty:
        for b in recs:
            print(b)
    else:
        for _, row in details.iterrows():
            print(
                f"{row['name']} | {row['categories']} "
                f"{row['city']}, {row['state']} | stars: {row['stars']} "
                f"(id: {row['business_id']})"
            )
