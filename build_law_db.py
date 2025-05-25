
#!/usr/bin/env python3
"""build_law_db.py – Convert Iranian law text files to SQLite & JSONL.

Usage
-----
    python build_law_db.py [-s SOURCE_DIR] [-d DB_FILE] [-j JSONL_FILE] [--overwrite]

Features
~~~~~~~~
* **Pure std-lib** – only `sqlite3`, `json`, `re`, `argparse`, `logging`, `datetime`, `pathlib` are used.
* **Persian-digit normalisation** before regex processing.
* **Idempotent**: running twice doesn’t duplicate rows (PRIMARY KEY).  
  Use `--overwrite` to rebuild from scratch.
* **Progress logging** and summary timings.
* **PEP 8 + type hints** for maintainability.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Tuple

# ---------------------------------------------------------------------------#
# 0. Constants & configuration                                                #
# ---------------------------------------------------------------------------#

DEFAULT_SRC_DIR = Path.cwd()
DEFAULT_DB_FILE = Path("iran_laws.db")
DEFAULT_JSONL_FILE = Path("iran_laws.jsonl")

PERSIAN_TO_LATIN = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
ARTICLE_RE = re.compile(r"ماده\s+(\d+)[\s\-—–.]*", re.MULTILINE)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"  # Fixed typo here
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
logger = logging.getLogger("build_law_db")

# ---------------------------------------------------------------------------#
# 1. Helpers                                                                  #
# ---------------------------------------------------------------------------#

def extract_articles(text: str) -> Iterator[Tuple[int, str]]:
    """Yield `(article_id, article_text)` pairs from full law *text*.

    Persian digits are converted to Latin before the regex split.
    """
    text = text.translate(PERSIAN_TO_LATIN)
    parts: list[str] = ARTICLE_RE.split(text)
    for idx in range(1, len(parts), 2):
        article_id_raw, body = parts[idx], parts[idx + 1]
        try:
            aid = int(article_id_raw)
        except ValueError:
            continue  # skip malformed numbers
        body = body.strip()
        if body:
            yield aid, body


def iter_text_files(directory: Path) -> Iterable[Path]:
    """Return sorted *.txt files (non-empty) in *directory*."""
    return sorted(p for p in directory.glob("*.txt") if p.stat().st_size > 0)

# ---------------------------------------------------------------------------#
# 2. Core build routine                                                       #
# ---------------------------------------------------------------------------#

def build_database(src_dir: Path, db_file: Path, jsonl_file: Path, *, overwrite: bool = False) -> None:
    start_ts = datetime.utcnow()

    txt_files = list(iter_text_files(src_dir))
    if not txt_files:
        logger.error("No .txt files found in %s", src_dir)
        sys.exit(1)
    logger.info("Found %d source files: %s", len(txt_files), ", ".join(f.name for f in txt_files))

    if overwrite and db_file.exists():
        db_file.unlink()

    with sqlite3.connect(db_file) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS articles (
                code TEXT,
                id INTEGER,
                text TEXT,
                PRIMARY KEY(code, id))"""
        )

        json_mode = "w" if overwrite else "a"
        jsonl_file.parent.mkdir(parents=True, exist_ok=True)

        with jsonl_file.open(json_mode, encoding="utf-8") as json_out:
            total_articles = 0
            for txt_path in txt_files:
                code = (
                    txt_path.stem.replace("_law", "")
                    .replace(" ", "_")
                    .lower()
                )
                seen_ids: set[int] = set()
                content = txt_path.read_text(encoding="utf-8")
                for aid, body in extract_articles(content):
                    if aid in seen_ids:
                        continue
                    seen_ids.add(aid)
                    conn.execute("INSERT OR IGNORE INTO articles VALUES (?,?,?)", (code, aid, body))
                    json_out.write(json.dumps({"code": code, "id": aid, "text": body}, ensure_ascii=False) + "\n")

                logger.info("%-20s → %4d ماده", code, len(seen_ids))
                total_articles += len(seen_ids)

    duration = (datetime.utcnow() - start_ts).total_seconds()
    logger.info("Done! %d articles saved in %s & %s (%.1fs)", total_articles, db_file, jsonl_file, duration)
    logger.info("UTC timestamp: %s", datetime.utcnow().isoformat(" ", timespec="seconds"))

# ---------------------------------------------------------------------------#
# 3. CLI                                                                      #
# ---------------------------------------------------------------------------#

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SQLite & JSONL from Iranian law text files.")
    parser.add_argument("-s", "--src-dir", type=Path, default=DEFAULT_SRC_DIR, help="Directory containing *.txt files")
    parser.add_argument("-d", "--db-file", type=Path, default=DEFAULT_DB_FILE, help="Output SQLite database path")
    parser.add_argument("-j", "--jsonl-file", type=Path, default=DEFAULT_JSONL_FILE, help="Output JSONL file path")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing outputs and rebuild from scratch")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)

# ---------------------------------------------------------------------------#
# 4. Main entry                                                               #
# ---------------------------------------------------------------------------#

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.debug:
        logger.setLevel(logging.DEBUG)

    build_database(
        src_dir=args.src_dir,
        db_file=args.db_file,
        jsonl_file=args.jsonl_file,
        overwrite=args.overwrite,
    )

if __name__ == "__main__":
    main()
