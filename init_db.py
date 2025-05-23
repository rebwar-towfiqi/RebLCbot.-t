# init_db.py

from admin.database import init_db

if __name__ == "__main__":
    init_db()
    print("✅ Database initialized.")

import sqlite3

conn = sqlite3.connect("laws.db")
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS famous_cases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        summary TEXT
    )
""")

conn.commit()
conn.close()

print("✅ جدول famous_cases با موفقیت ساخته شد.")
