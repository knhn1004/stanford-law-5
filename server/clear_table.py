import os
from dotenv import load_dotenv
import psycopg2

# Load environment variables
load_dotenv()

# Get database URL from environment
db_url = os.getenv("DB_URL")

def clear_stanford_law_contracts():
    """Clear all rows from the stanford_law_contracts table."""
    table_name = "stanford_law_contracts"
    print(f"Clearing all rows from table: {table_name}")

    try:
        conn = psycopg2.connect(db_url)
        
        with conn.cursor() as cur:
            cur.execute('TRUNCATE TABLE "stanford_law_contracts" CASCADE')
            conn.commit()
            print(f"Successfully cleared all rows from {table_name}!")

    except psycopg2.Error as e:
        print(f"Error clearing table: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    clear_stanford_law_contracts() 