from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# مسیر فایل SQLite شما
DB_PATH = Path("users.db")
DB_PATH.touch(exist_ok=True)

# ---------------------------------------------------------------------------#
# ۱. اتصال با Row Factory                                                     #
# ---------------------------------------------------------------------------#
def _connect() -> sqlite3.Connection:
    """
    این تابع یک اتصال به فایل SQLite باز می‌کند، row_factory را تنظیم می‌کند
    و PRAGMA foreign_keys را فعال می‌نماید.
    """
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    # مهم: این خط باعث می‌شود خروجی cursor.fetchall() به صورت sqlite3.Row باشد
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

@contextmanager
def get_db() -> Iterator[sqlite3.Connection]:
    """
    Context-manager با commit/rollback خودکار و لاگ خطا.
    هر بار که با "with get_db() as conn:" وارد شویم، اینجا یک شیء sqlite3.Connection می‌گیریم.
    """
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        logger.exception("DB error – rolled back")
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------#
# ۲. ساخت اسکیما و ایندکس (در صورت نیاز)                                       #
# ---------------------------------------------------------------------------#
def init_db() -> None:
    with get_db() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id         INTEGER PRIMARY KEY,
            username        TEXT,
            first_name      TEXT,
            last_name       TEXT,
            status          TEXT NOT NULL DEFAULT 'pending',
            receipt_photo_id TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);
        """)
    logger.info("SQLite schema ready")


# ---------------------------------------------------------------------------#
# ۳. عملیات CRUD با حداقل اتصال                                               #
# ---------------------------------------------------------------------------#
def add_user(
    user_id: int,
    username: str | None,
    first_name: str | None,
    last_name: str | None
) -> None:
    with get_db() as db:
        db.execute("""
            INSERT OR IGNORE INTO users (user_id, username, first_name, last_name)
            VALUES (?, ?, ?, ?)
        """, (user_id, username, first_name, last_name))

def update_user_status(user_id: int, status: str) -> None:
    with get_db() as db:
        db.execute("UPDATE users SET status = ? WHERE user_id = ?", (status, user_id))

def update_receipt(user_id: int, photo_id: str) -> None:
    with get_db() as db:
        db.execute("""
            UPDATE users
               SET receipt_photo_id = ?, status = 'pending'
             WHERE user_id = ?
        """, (photo_id, user_id))

def get_user_status(user_id: int) -> Optional[str]:
    with get_db() as db:
        row = db.execute(
            "SELECT status FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
    return row["status"] if row else None

def list_pending_users() -> List[Tuple]:
    with get_db() as db:
        return db.execute("""
            SELECT user_id, username, first_name, last_name
              FROM users
             WHERE status = 'pending'
        """).fetchall()
