import json
from psycopg2.extras import execute_values
from tqdm import tqdm
from db import get_connection

BATCH_SIZE = 1000

def ingest_users(json_path):
    conn = get_connection()
    cur = conn.cursor()

    insert_query = """
        INSERT INTO users (
            user_id, name, review_count, average_stars, yelping_since
        )
        VALUES %s
        ON CONFLICT (user_id) DO NOTHING
    """

    batch = []

    with open(json_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Ingesting users"):
            u = json.loads(line)

            batch.append((
                u["user_id"],
                u["name"],
                u["review_count"],
                u["average_stars"],
                u["yelping_since"]
            ))

            if len(batch) == BATCH_SIZE:
                execute_values(cur, insert_query, batch)
                conn.commit()
                batch.clear()

        if batch:
            execute_values(cur, insert_query, batch)
            conn.commit()

    cur.close()
    conn.close()
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python ingest_users.py <path_to_user.json>")
        sys.exit(1)

    ingest_users(sys.argv[1])
