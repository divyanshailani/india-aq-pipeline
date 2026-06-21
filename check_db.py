import os, sys
import psycopg2
sys.path.insert(0, '/Users/divyanshailani/Desktop/pow-eda-pipeline')
from src.config import DB_CONFIG

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

def get_cols(table):
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = %s;", (table,))
    cols = [r[0] for r in cur.fetchall()]
    print(f"{table} columns:", cols)

cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
tables = [r[0] for r in cur.fetchall()]
print("Tables:", tables)

if "stations" in tables: get_cols("stations")
if "daily_features" in tables: get_cols("daily_features")
if "viirs_data" in tables: get_cols("viirs_data")

conn.close()
