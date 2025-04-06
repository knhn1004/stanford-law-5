import os
from dotenv import load_dotenv
import psycopg2

# Load environment variables
load_dotenv()

# Get database URL from environment
db_url = os.getenv("DB_URL")


def init_db():
    """Initialize the database with required extensions and tables."""
    print("Initializing database...")

    # Read the SQL file
    with open("init_db.sql", "r") as f:
        sql = f.read()

    # Connect to the database and execute SQL
    try:
        conn = psycopg2.connect(db_url)
        conn.autocommit = True  # Required for creating extensions

        with conn.cursor() as cur:
            print("Executing SQL initialization...")
            cur.execute(sql)
            print("Database initialization completed successfully!")

    except psycopg2.Error as e:
        print(f"Error initializing database: {e}")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    init_db()
