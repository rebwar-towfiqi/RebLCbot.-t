#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_cases_db.py – ایجاد پایگاه داده پرونده‌های مشهور از فایل‌های متنی
"""

import sqlite3
from pathlib import Path

# مسیر پوشه پرونده‌ها
CASES_DIR = Path("legal_documents/famous cases")
DB_FILE = Path("laws.db")

# اطمینان از وجود پایگاه داده و جدول
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS famous_cases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        summary TEXT
    )
""")

# حذف داده‌های قبلی برای ساخت مجدد (اختیاری)
cursor.execute("DELETE FROM famous_cases")

# بارگذاری فایل‌ها
for file in sorted(CASES_DIR.glob("*.txt")):
    try:
        title = file.stem.replace("_", " ").strip()
        summary = file.read_text(encoding="utf-8").strip()
        if summary:
            cursor.execute("INSERT INTO famous_cases (title, summary) VALUES (?, ?)", (title, summary))
            print(f"✅ {title}")
    except Exception as e:
        print(f"❌ خطا در {file.name}: {e}")

conn.commit()
conn.close()
print("🎉 پایگاه داده پرونده‌های مشهور با موفقیت ساخته شد.")
