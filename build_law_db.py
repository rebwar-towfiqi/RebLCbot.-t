#!/usr/bin/env python3
"""
build_law_db.py – Convert law text files to SQLite & JSONL with country tagging.

Usage:
    python build_law_db.py [-s SOURCE_DIR] [-d DB_FILE] [-j JSONL_FILE] [--overwrite]

Features:
~~~~~~~~~
* **Pure std-lib** – only uses built-in modules.
* **Persian-digit normalisation**
* **Supports all file types**, even those without article numbers.
* **Country tagging**: Iran vs France based on filename.
* **Idempotent**: running twice doesn’t duplicate rows.
* **Progress logging** and summary timings.
"""

import argparse
import json
import logging
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator, Tuple

# Constants
DEFAULT_SRC_DIR = Path.cwd()
DEFAULT_DB_FILE = Path("iran_laws.db")
DEFAULT_JSONL_FILE = Path("iran_laws.jsonl")

PERSIAN_TO_LATIN = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
ARTICLE_RE = re.compile(r"ماده\s+(\d+)[\s\-—–.]*", re.MULTILINE)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
logger = logging.getLogger("build_law_db")


# ---------------- #
# Helper functions #
# ---------------- #

def extract_articles(text: str) -> Iterator[Tuple[int, str]]:
    """Yield (article_id, article_text) pairs from full law text."""
    text = text.translate(PERSIAN_TO_LATIN)
    parts = ARTICLE_RE.split(text)
    for idx in range(1, len(parts), 2):
        article_id_raw, body = parts[idx], parts[idx + 1]
        try:
            aid = int(article_id_raw)
        except ValueError:
            continue
        body = body.strip()
        if body:
            yield aid, body


def iter_text_files(directory: Path) -> list[Path]:
    """Return sorted non-empty .txt files in directory."""
    return sorted(p for p in directory.glob("*.txt") if p.stat().st_size > 0)


def guess_country(filename: str) -> str:
    """Guess country based on filename."""
    name = filename.lower()

    if "iran" in name or "ایران" in name:
        return "ایران"
    elif "france" in name or "فرانسه" in name:
        return "فرانسه"
    elif "germany" in name or "آلمان" in name:
        return "آلمان"
    elif "usa" in name or "america" in name or "آمریکا" in name:
        return "ایالات متحده"
    elif "uk" in name or "england" in name or "بریتانیا" in name:
        return "بریتانیا"
    elif "روسیه" in name or "russia" in name:
        return "روسیه"
    elif "hammurabi" in name or "babylon" in name or "حمورابی" in name:
        return "بابل"
    else:
        return "نامشخص"


# --------------------- #
# Main database builder #
# --------------------- #

def build_database(src_dir: Path, db_file: Path, jsonl_file: Path, *, overwrite: bool = False) -> None:
    start_ts = datetime.utcnow()

    txt_files = iter_text_files(src_dir)
    if not txt_files:
        logger.error("No .txt files found in %s", src_dir)
        sys.exit(1)
    logger.info("Found %d source files: %s", len(txt_files), ", ".join(f.name for f in txt_files))

    if overwrite and db_file.exists():
        db_file.unlink()

    with sqlite3.connect(db_file) as conn:
        # Create tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS laws (
                code TEXT PRIMARY KEY,
                country TEXT NOT NULL,
                title TEXT NOT NULL,
                full_text TEXT NOT NULL,
                source_file TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                code TEXT,
                id INTEGER,
                text TEXT,
                PRIMARY KEY(code, id)
            )
        """)

        # Open JSONL file
        json_mode = "w" if overwrite else "a"
        jsonl_file.parent.mkdir(parents=True, exist_ok=True)
        total_laws = 0
        total_articles = 0

        with jsonl_file.open(json_mode, encoding="utf-8") as json_out:
            for txt_path in txt_files:
                code = txt_path.stem.replace(" ", "_").lower()
                country = guess_country(txt_path.name)
                title = txt_path.stem
                full_text = txt_path.read_text(encoding="utf-8")

                # Insert full law
                conn.execute(
                    "INSERT OR IGNORE INTO laws VALUES (?, ?, ?, ?, ?, datetime('now'))",
                    (code, country, title, full_text, str(txt_path))
                )

                # Write full law to JSONL
                json_out.write(json.dumps({
                    "type": "law",
                    "code": code,
                    "country": country,
                    "title": title,
                    "source_file": str(txt_path),
                    "full_text": full_text
                }, ensure_ascii=False) + "\n")
                total_laws += 1

                # Extract and insert articles if any
                seen_ids = set()
                for aid, body in extract_articles(full_text):
                    if aid in seen_ids:
                        continue
                    seen_ids.add(aid)
                    conn.execute(
                        "INSERT OR IGNORE INTO articles VALUES (?, ?, ?)",
                        (code, aid, body)
                    )
                    json_out.write(json.dumps({
                        "type": "article",
                        "code": code,
                        "id": aid,
                        "text": body
                    }, ensure_ascii=False) + "\n")
                total_articles += len(seen_ids)
                logger.info("%-20s → %4d ماده", code, len(seen_ids))

    duration = (datetime.utcnow() - start_ts).total_seconds()
    logger.info("Done! %d laws (%d articles) saved in %s & %s (%.1fs)", total_laws, total_articles, db_file, jsonl_file, duration)
    logger.info("UTC timestamp: %s", datetime.utcnow().isoformat(" ", timespec="seconds"))


# -------------- #
# CLI interface  #
# -------------- #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SQLite & JSONL from law text files.")
    parser.add_argument("-s", "--src-dir", type=Path, default=DEFAULT_SRC_DIR, help="Directory containing *.txt files")
    parser.add_argument("-d", "--db-file", type=Path, default=DEFAULT_DB_FILE, help="Output SQLite database path")
    parser.add_argument("-j", "--jsonl-file", type=Path, default=DEFAULT_JSONL_FILE, help="Output JSONL file path")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing outputs and rebuild from scratch")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)

def init_db(db_path: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS famous_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            summary TEXT
        );
    """)
    conn.commit()
    return conn

def insert_case(conn, title: str, summary: str):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO famous_cases (title, summary) VALUES (?, ?)",
        (title, summary)
    )
    conn.commit()

def process_files(src_dir: str, conn):
    for filename in os.listdir(src_dir):
        if filename.endswith(".txt"):
            path = os.path.join(src_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                title = os.path.splitext(filename)[0]
                insert_case(conn, title, content)
                print(f"✅ پرونده '{title}' وارد شد.")

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    build_database(args.src_dir, args.db_file, args.jsonl_file, overwrite=args.overwrite)


if __name__ == "__main__":
    main()