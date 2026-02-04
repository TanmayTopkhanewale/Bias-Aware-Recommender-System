import json
from psycopg2.extras import execute_values
from tqdm import tqdm
from db import get_connection

BATCH_SIZE = 1000

def ingest_businesses(json_path):
    conn = get_connection()
    cur = conn.cursor()

    insert_query = """
        INSERT INTO businesses (
            business_id, name, city, state,
            stars, review_count, categories,
            latitude, longitude
        )
        VALUES %s
        ON CONFLICT (business_id) DO NOTHING
    """

    batch = []

    with open(json_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Ingesting businesses"):
            b = json.loads(line)

            batch.append((
                b["business_id"],
                b["name"],
                b["city"],
                b["state"],
                b["stars"],
                b["review_count"],
                b["categories"],
                b["latitude"],
                b["longitude"]
            ))

            # When batch is full → write to DB
            if len(batch) == BATCH_SIZE:
                execute_values(cur, insert_query, batch)
                conn.commit()
                batch.clear()

        # insert remaining rows
        if batch:
            execute_values(cur, insert_query, batch)
            conn.commit()

    cur.close()
    conn.close()
    
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python ingest_businesses.py <path_to_business.json>")
        sys.exit(1)

    ingest_businesses(sys.argv[1])
