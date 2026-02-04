import psycopg2

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="yelp",
        user="postgres",
        password="postgres"
    )
