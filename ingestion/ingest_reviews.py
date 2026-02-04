import json
from psycopg2.extras import execute_values
from tqdm import tqdm
from db import get_connection

BATCH_SIZE = 2000

STAR_WEIGHT = {
    1: -1.0,
    2: -0.5,
    3: 0.2,
    4: 0.7,
    5: 1.0
}

def ingest_reviews(json_path):
    conn = get_connection()
    cur = conn.cursor()
    def load_valid_users(cur):
        cur.execute("SELECT user_id FROM users")
        return set(row[0] for row in cur.fetchall())
    valid_users = load_valid_users(cur)
    review_insert = """
        INSERT INTO reviews (
            review_id, user_id, business_id, stars, text, date
        )
        VALUES %s
        ON CONFLICT (review_id) DO NOTHING
    """

    event_insert = """
        INSERT INTO user_events (
            user_id, business_id, event_type, event_strength, event_time
        )
        VALUES %s
    """

    review_batch = []
    event_batch = []

    with open(json_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Ingesting reviews"):
            r = json.loads(line)
            if r["user_id"] not in valid_users:
                continue
            review_batch.append((
                r["review_id"],
                r["user_id"],
                r["business_id"],
                r["stars"],
                r["text"],
                r["date"]
            ))

            event_batch.append((
                r["user_id"],
                r["business_id"],
                "review",
                STAR_WEIGHT[r["stars"]],
                r["date"]
            ))

            if len(review_batch) == BATCH_SIZE:
                execute_values(cur, review_insert, review_batch)
                execute_values(cur, event_insert, event_batch)
                conn.commit()
                review_batch.clear()
                event_batch.clear()

        if review_batch:
            execute_values(cur, review_insert, review_batch)
            execute_values(cur, event_insert, event_batch)
            conn.commit()

    cur.close()
    conn.close()

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python ingest_reviews.py <path_to_review.json>")
        sys.exit(1)

    ingest_reviews(sys.argv[1])