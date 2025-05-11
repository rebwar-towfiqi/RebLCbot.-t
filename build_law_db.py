#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_law_db.py  –  تبدیل فایل‌های متنی قوانین ایران به یک دیتابیس SQLite و فایل JSONL

نحوهٔ استفاده:
    python build_law_db.py

ورودی:
    همهٔ فایل‌های *.txt در پوشهٔ جاری (UTF‑8)
    مانند: civil_law.txt، commercial_law.txt، criminal_law.txt …

خروجی:
    iran_laws.db   → جدول articles(code TEXT, id INT, text TEXT)
    iran_laws.jsonl → یک خط JSON در ازای هر ماده

کتابخانه‌های موردنیاز: فقط استاندارد پایتون (sqlite3، json، re)
"""

import json
import re
import sqlite3
from pathlib import Path
from datetime import datetime

# ---------- پیکربندی ----------
SRC_DIR = Path('.')                # پوشهٔ حاوی فایل‌های txt
DB_FILE = Path('iran_laws.db')
JSONL_FILE = Path('iran_laws.jsonl')

# الگوی شناسایی شمارهٔ ماده «ماده 234» یا «ماده ۲۳۴»
PERSIAN_TO_LATIN = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')
ARTICLE_RE = re.compile(r'ماده\s+(\d+)[\s\-—–.]*')


def extract_articles(text: str):
    """Yield (id:int, text:str) tuples from full law text."""
    text = text.translate(PERSIAN_TO_LATIN)
    parts = ARTICLE_RE.split(text)
    for i in range(1, len(parts), 2):
        try:
            aid = int(parts[i])
        except ValueError:
            continue
        body = parts[i + 1].strip()
        if body:
            yield aid, body


# ---------- مرحلهٔ 1: یافتن فایل‌ها ----------
files = sorted(f for f in SRC_DIR.glob('*.txt') if f.stat().st_size > 0)
if not files:
    raise SystemExit('⚠️  هیچ فایل txt پیدا نشد!')

print('🔍 یافت شد:', ', '.join(f.name for f in files))

# ---------- مرحلهٔ 2: ساخت دیتابیس ----------
DB_FILE.unlink(missing_ok=True)
conn = sqlite3.connect(DB_FILE)
conn.execute('CREATE TABLE articles (code TEXT, id INTEGER, text TEXT, PRIMARY KEY(code, id))')

json_out = JSONL_FILE.open('w', encoding='utf-8')

total_articles = 0

for f in files:
    # تعیین شناسهٔ قانون براساس نام فایل → "civil_law.txt" → "civil"
    code = (
        f.stem.replace('_law', '')
             .replace(' ', '_')
             .lower()
    )

    text = f.read_text(encoding='utf-8')
    seen_ids = set()

    for aid, body in extract_articles(text):
        if aid in seen_ids:
            continue  # رفع تکرار
        seen_ids.add(aid)

        conn.execute('INSERT OR IGNORE INTO articles VALUES (?,?,?)', (code, aid, body))
        json_out.write(json.dumps({'code': code, 'id': aid, 'text': body}, ensure_ascii=False) + '\n')

    print(f'  {code:20s} → {len(seen_ids):4d} ماده')
    total_articles += len(seen_ids)

conn.commit()
conn.close()
json_out.close()

print(f'\n✅ تمام شد! {total_articles} ماده در {DB_FILE} و {JSONL_FILE}.')
print('⏱️ ', datetime.utcnow().isoformat(' ', timespec='seconds'), 'UTC')
