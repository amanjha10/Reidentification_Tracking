from dotenv import load_dotenv
import os
import psycopg2

load_dotenv()

def get_db_connection():
    return psycopg2.connect(os.getenv("DATABASE_URL"))

print("Checked")
