# Run this in Python to reset the database
import os
import sqlite3

# Close any existing connections
try:
    from database.dataManager import close_connection
    close_connection()
except:
    pass

# Delete the old database
if os.path.exists("king_arthur.db"):
    os.remove("king_arthur.db")
    print("✅ Old database deleted")

# Delete ChromaDB embeddings (optional, to start fresh)
import shutil
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")
    print("✅ Old ChromaDB deleted")

print("Now restart your app - new database will be created with all columns")