# view_db.py
import sqlite3
from pathlib import Path

# Try both common paths
possible_paths = [
    "data/database/king_arthur.db",
    "king_arthur.db",
    "chroma_db/chroma.sqlite3",  # ChromaDB uses sqlite internally
]

db_path = None
for p in possible_paths:
    if Path(p).exists():
        db_path = p
        break

if not db_path:
    print("❌ No database file found. Check paths.")
    exit()

print(f"📂 Using database: {db_path}\n")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 1. Show all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]
print("📋 Tables:", tables)

# 2. For each table, show row count and first few rows
for table in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    print(f"\n── {table} ({count} rows) ──")
    if count > 0:
        cursor.execute(f"SELECT * FROM {table} LIMIT 5")
        rows = cursor.fetchall()
        # Get column names
        col_names = [desc[0] for desc in cursor.description]
        print("  Columns:", ", ".join(col_names))
        for i, row in enumerate(rows, 1):
            print(f"  Row {i}: {dict(zip(col_names, row))}")
        if count > 5:
            print(f"  ... and {count - 5} more rows")

conn.close()