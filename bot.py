#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RebLawBot – Telegram bot that sells subscriptions and answers legal questions using OpenAI.
Version 2025-05-24 – Stable Edition
"""

from __future__ import annotations


import asyncio
import logging
import os
import sqlite3
import re
import json
import tempfile

DB_PATH = "data/reblaw.db"  # ← مسیر دیتابیس SQLite شما

from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Generator, Optional
from database import get_db

# External libraries
from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError, RateLimitError, AuthenticationError
from psycopg2.pool import SimpleConnectionPool
from telegram import Update, Message, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.constants import ChatAction, ParseMode
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from telegram import Update
from telegram.ext import ContextTypes

from texts import TEXTS  # assuming texts.py provides translation strings


from functools import wraps
from database import add_rlc_score  # مطمئن شو این تابع را در db.py ساختی
from database import create_score_table
create_score_table()


ADMIN_IDS = {1596461417}  # 👈 شناسه تلگرام خودتان را جایگزین کنید

def admin_only(func):
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        if user_id not in ADMIN_IDS:
            await update.message.reply_text("⛔ این دستور فقط برای مدیر مجاز است.")
            return
        return await func(update, context, *args, **kwargs)
    return wrapper


from datetime import datetime, date

def get_credits(user_id: int) -> int:
    """دریافت اعتبار باقی‌مانده برای کاربر و ریست روزانه در صورت نیاز"""
    today = date.today()

    if USE_PG:
        assert POOL is not None
        with POOL.getconn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT credits_left, last_reset FROM credits WHERE user_id = %s", (user_id,))
                row = cur.fetchone()

                if row:
                    credits_left, last_reset = row
                    if last_reset != today:
                        # ریست روزانه
                        cur.execute(
                            "UPDATE credits SET credits_left = 1, last_reset = %s WHERE user_id = %s",
                            (today, user_id)
                        )
                        conn.commit()
                        return 1
                    return credits_left
                else:
                    cur.execute(
                        "INSERT INTO credits (user_id, credits_left, last_reset) VALUES (%s, %s, %s)",
                        (user_id, 1, today)
                    )
                    conn.commit()
                    return 1

    else:
        with sqlite3.connect(SQLITE_FILE) as conn:
            cur = conn.cursor()
            cur.execute("SELECT credits_left, last_reset FROM credits WHERE user_id = ?", (user_id,))
            row = cur.fetchone()

            if row:
                credits_left, last_reset = row
                if last_reset != today.isoformat():
                    cur.execute(
                        "UPDATE credits SET credits_left = 1, last_reset = ? WHERE user_id = ?",
                        (today.isoformat(), user_id)
                    )
                    conn.commit()
                    return 1
                return credits_left
            else:
                cur.execute(
                    "INSERT INTO credits (user_id, credits_left, last_reset) VALUES (?, ?, ?)",
                    (user_id, 1, today.isoformat())
                )
                conn.commit()
                return 1


def decrement_credits(user_id: int) -> None:
    """کاهش یک واحد از اعتبار کاربر، فقط اگر اعتبار مثبت باشد"""
    if USE_PG:
        assert POOL is not None
        with POOL.getconn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE credits
                    SET credits_left = credits_left - 1
                    WHERE user_id = %s AND credits_left > 0
                """, (user_id,))
                conn.commit()
    else:
        with sqlite3.connect(SQLITE_FILE) as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE credits
                SET credits_left = credits_left - 1
                WHERE user_id = ? AND credits_left > 0
            """, (user_id,))
            conn.commit()


# ─── Global Environment and Logging ──────────────────────────────────────────
load_dotenv()  # Load environment variables from .env
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger("RebLawBot")

# Async OpenAI client (for answering questions)
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─── Utility: Voice to Text (Whisper model) ─────────────────────────────────
import whisper
import ffmpeg  # ensure ffmpeg is installed in environment
whisper_model = whisper.load_model("base")

def voice_to_text(file_path: str) -> str:
    """Convert an audio file to text using OpenAI Whisper."""
    result = whisper_model.transcribe(file_path)
    return result["text"]


# ─── Bot Menus and Language Helper ──────────────────────────────────────────
def get_main_menu(lang: str):
    menus = {
        "fa": [
            [KeyboardButton("🛒 خرید اشتراک"), KeyboardButton("📤 ارسال رسید")],
            [KeyboardButton("⚖️ سؤال حقوقی"), KeyboardButton("🎤 سؤال صوتی")],
            [KeyboardButton("📚 پرونده‌های مشهور"), KeyboardButton("ℹ️ درباره توکن")],
            [KeyboardButton("/lang")]
        ],
        "en": [
            [KeyboardButton("🛒 Buy Subscription"), KeyboardButton("📤 Send Receipt")],
            [KeyboardButton("⚖️ Legal Question"), KeyboardButton("🎤 Voice Question")],
            [KeyboardButton("📚 Famous Cases"), KeyboardButton("ℹ️ About Token")],
            [KeyboardButton("/lang")]
        ],
        "ku": [
            [KeyboardButton("🛒 کڕینی بەشداریکردن"), KeyboardButton("📤 ناردنی پسوڵە")],
            [KeyboardButton("⚖️ پرسیاری یاسایی"), KeyboardButton("🎤 پرسیاری دەنگی")],
            [KeyboardButton("📚 پرۆسەی ناودار"), KeyboardButton("ℹ️ دەربارەی تۆکێن")],
            [KeyboardButton("/lang")]
        ]
    }
    return ReplyKeyboardMarkup(menus.get(lang, menus["fa"]), resize_keyboard=True)



def tr(key: str, lang: str = "fa", **kwargs) -> str:
    """Translate text by key for the given language (fallback to Persian)."""
    base_text = TEXTS.get(key, {}).get(lang) or TEXTS.get(key, {}).get("fa") or ""
    return base_text.format(**kwargs)


def getenv_or_die(key: str) -> str:
    """Get an environment variable or raise an error if it's missing."""
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Environment variable {key!r} is missing")
    return value


def get_lang(context: ContextTypes.DEFAULT_TYPE) -> str:
    """Retrieve or initialize the user's language (defaults to 'fa')."""
    lang = context.user_data.get("lang")
    if lang not in ("fa", "en", "ku"):
        lang = "fa"
        context.user_data["lang"] = lang
    return lang


from datetime import date

def check_and_use_credit(user_id: int) -> bool:
    """
    Check if the user has already used their free credit today.
    If not, insert a new record and return True.
    If already used, return False.
    """
    today = date.today()
    if USE_PG:
        assert POOL is not None
        with POOL.getconn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM credits WHERE user_id = %s AND used_at = %s;",
                    (user_id, today)
                )
                if cur.fetchone():
                    return False
                cur.execute(
                    "INSERT INTO credits (user_id, used_at) VALUES (%s, %s);",
                    (user_id, today)
                )
            conn.commit()
        return True
    else:
        with sqlite3.connect(SQLITE_FILE) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT 1 FROM credits WHERE user_id = ? AND used_at = ?;",
                (user_id, today)
            )
            if cur.fetchone():
                return False
            cur.execute(
                "INSERT INTO credits (user_id, used_at) VALUES (?, ?);",
                (user_id, today)
            )
            conn.commit()
        return True

def log_question_answer(user_id: int, question: str, answer: str) -> None:
    """Log the asked question and its answer into the database."""
    
    from datetime import timezone
    now = datetime.now(timezone.utc)

    if USE_PG:
        assert POOL is not None
        with POOL.getconn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO questions (user_id, question, answer, asked_at)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (user_id, question, answer, now)
                )
            conn.commit()
    else:
        with sqlite3.connect(SQLITE_FILE) as conn:
            conn.execute(
                """
                INSERT INTO questions (user_id, question, answer, asked_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, question, answer, now)
            )
            conn.commit()


# ─── Database Setup (PostgreSQL with SQLite fallback) ───────────────────────
SQLITE_FILE = Path("users.db")
POOL: Optional[SimpleConnectionPool] = None
USE_PG = False  # Will be set to True if PostgreSQL is available
_sqlite_lock = asyncio.Lock()  # using asyncio Lock for async context if needed

def init_db() -> None:
    """
    Initialize the database connection.
    Tries PostgreSQL (if POSTGRES_URL is set and reachable), otherwise uses SQLite.
    Creates the necessary tables if they don't exist.
    """
    global POOL, USE_PG

    try:
        pg_url = os.getenv("POSTGRES_URL")  # e.g., postgres://user:pass@host:port/db
        if not pg_url:
            raise ValueError("POSTGRES_URL not set")

        # Attempt PostgreSQL connection pool
        POOL = SimpleConnectionPool(minconn=1, maxconn=5, dsn=pg_url, connect_timeout=10, sslmode="require")

        # Simple test query to verify connection
        with POOL.getconn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")

        USE_PG = True
        logger.info("✅ Connected to PostgreSQL")
        _setup_schema_pg()

    except Exception as exc:

        logger.warning("PostgreSQL unavailable (%s), switching to SQLite.", exc)
        USE_PG = False
        _setup_schema_sqlite()

    _update_placeholder()


def _setup_schema_pg() -> None:
    """Create tables in PostgreSQL if they don't exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS users (
        user_id          BIGINT PRIMARY KEY,
        username         TEXT,
        first_name       TEXT,
        last_name        TEXT,
        status           TEXT    DEFAULT 'pending',
        receipt_photo_id TEXT,
        expire_at        TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS questions (
        id         SERIAL PRIMARY KEY,
        user_id    BIGINT,
        question   TEXT,
        answer     TEXT,
        asked_at   TIMESTAMP DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS credits (
        user_id      BIGINT PRIMARY KEY,
        credits_left INTEGER NOT NULL DEFAULT 1,
        last_reset   DATE
    );
    """

    assert POOL is not None
    with POOL.getconn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()



def _setup_schema_sqlite() -> None:
    """Create tables in SQLite if they don't exist."""
    SQLITE_FILE.touch(exist_ok=True)
    ddl = """
    CREATE TABLE IF NOT EXISTS users (
        user_id          INTEGER PRIMARY KEY,
        username         TEXT,
        first_name       TEXT,
        last_name        TEXT,
        status           TEXT    DEFAULT 'pending',
        receipt_photo_id TEXT,
        expire_at        TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS questions (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    INTEGER,
        question   TEXT,
        answer     TEXT,
        asked_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS credits (
        user_id      INTEGER PRIMARY KEY,
        credits_left INTEGER NOT NULL DEFAULT 1,
        last_reset   DATE
    );
    """

    with sqlite3.connect(SQLITE_FILE) as conn:
        conn.executescript(ddl)
        conn.commit()


_PLACEHOLDER = "?"  # Will use SQLite placeholder by default

def _update_placeholder() -> None:
    """Update the SQL placeholder based on which DB is in use. Call after init_db()."""
    global _PLACEHOLDER
    _PLACEHOLDER = "%s" if USE_PG else "?"


@contextmanager
def get_db():
    """
    Context manager for database connection (returns a psycopg2 connection or sqlite3 connection).
    Usage:
        with get_db() as conn:
            ... # use conn.cursor() etc.
    """
    if USE_PG:
        # PostgreSQL connection from pool
        conn = POOL.getconn()
        try:
            yield conn
        finally:
            POOL.putconn(conn)
    else:
        # SQLite connection (not truly async-safe, hence protected by lock if used in async context)
        conn = sqlite3.connect(SQLITE_FILE)
        try:
            yield conn
        finally:
            conn.close()


def _exec(sql: str, params: tuple = ()) -> None:
    """Execute a write operation (INSERT/UPDATE/DELETE) on the database."""
    if USE_PG:
        # Use a connection from pool for executing
        with POOL.getconn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                conn.commit()
            POOL.putconn(conn)
    else:
        # SQLite execution (single thread assumed or external lock used)
        with sqlite3.connect(SQLITE_FILE) as conn:
            conn.execute(sql, params)
            conn.commit()


def _fetchone(sql: str, params: tuple = ()) -> Optional[tuple]:
    """Execute a read query (SELECT) and return the first row, or None."""
    if USE_PG:
        with POOL.getconn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
            POOL.putconn(conn)
            return row
    else:
        with sqlite3.connect(SQLITE_FILE) as conn:
            cur = conn.execute(sql, params)
            return cur.fetchone()


# ─── Database Helper Functions ──────────────────────────────────────────────
def upsert_user(user_id: int, username: Optional[str], first: Optional[str], last: Optional[str]) -> None:
    """Insert or update a user's profile in the DB. Initial status is 'pending' if new."""
    sql = (
        # PostgreSQL uses ON CONFLICT for upsert
        f"INSERT INTO users (user_id, username, first_name, last_name) "
        f"VALUES ({_PLACEHOLDER},{_PLACEHOLDER},{_PLACEHOLDER},{_PLACEHOLDER}) "
        f"ON CONFLICT (user_id) DO UPDATE SET "
        f"username=EXCLUDED.username, first_name=EXCLUDED.first_name, last_name=EXCLUDED.last_name"
        if USE_PG else
        # SQLite (ON CONFLICT requires a conflict clause on table definition; using REPLACE as simpler approach)
        f"INSERT OR REPLACE INTO users (user_id, username, first_name, last_name) "
        f"VALUES ({_PLACEHOLDER},{_PLACEHOLDER},{_PLACEHOLDER},{_PLACEHOLDER})"
    )
    _exec(sql, (user_id, username, first, last))


def save_receipt_request(user_id: int, receipt_data: str) -> None:
    """Save the receipt (photo file_id or text) and mark user status as 'awaiting' for admin review."""
    sql = (
        f"UPDATE users SET receipt_photo_id={_PLACEHOLDER}, status='awaiting' WHERE user_id={_PLACEHOLDER}"
    )
    _exec(sql, (receipt_data, user_id))


def set_user_status(user_id: int, status: str) -> None:
    """Update the user's status (pending/approved/rejected/awaiting)."""
    _exec(f"UPDATE users SET status={_PLACEHOLDER} WHERE user_id={_PLACEHOLDER}", (status, user_id))



def save_subscription(user_id: int, days: int = 30) -> None:
    """On approving a receipt, set subscription expiration (now + days) and status to 'approved'."""
    expire_at = datetime.utcnow() + timedelta(days=days)
    sql = (
        f"UPDATE users SET expire_at={_PLACEHOLDER}, status='approved' WHERE user_id={_PLACEHOLDER}"
    )
    _exec(sql, (expire_at, user_id))


from datetime import datetime, timezone

def has_active_subscription(user_id: int) -> bool:
    """Check if the user has an active subscription (expire_at in the future and status='approved')."""
    row = _fetchone(
        f"SELECT expire_at FROM users WHERE user_id={_PLACEHOLDER} AND status='approved'",
        (user_id,)
    )
    if not row or row[0] is None:
        return False

    expire_at = row[0]  # In PG this might be a datetime, in SQLite a string
    if isinstance(expire_at, str):
        expire_at = datetime.fromisoformat(expire_at)
    if expire_at.tzinfo is None:
        expire_at = expire_at.replace(tzinfo=timezone.utc)

    return expire_at >= datetime.now(timezone.utc)



# If there's an external "famous cases" database for /cases command:
def get_famous_cases() -> list[tuple[int, str]]:
    """
    Fetch a list of famous case (id, title) from a local database (e.g., laws.db).
    Returns a list of (case_id, title).
    """
    try:
        with sqlite3.connect("laws.db") as conn:
            rows = conn.execute("SELECT id, title FROM famous_cases ORDER BY id ASC").fetchall()
        return [(row[0], row[1]) for row in rows]
    except Exception as e:
        logger.error("Error fetching famous cases: %s", e)
        return []


def get_case_summary(case_id: int) -> Optional[str]:
    """Get summary text for a famous case by ID from the local database."""
    with sqlite3.connect("laws.db") as conn:
        row = conn.execute("SELECT summary FROM famous_cases WHERE id=?", (case_id,)).fetchone()
    return row[0] if row else None


def get_user_subscription_expiry(user_id: int) -> Optional[datetime]:
    """
    Retrieve the expiration datetime of the user's subscription from the database.
    Returns None if the user is not subscribed.
    """
    query = "SELECT expire_at FROM users WHERE user_id = ?"
    if USE_PG:
        query = "SELECT expire_at FROM users WHERE user_id = %s"

    try:
        if USE_PG:
            assert POOL is not None
            with POOL.getconn() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (user_id,))
                    row = cur.fetchone()
        else:
            with sqlite3.connect(SQLITE_FILE) as conn:
                row = conn.execute(query, (user_id,)).fetchone()
        if row and row[0]:
            return datetime.fromisoformat(str(row[0]))
    except Exception as e:
        logger.error("Error in get_user_subscription_expiry: %s", e)
    return None

def add_rlc_score(user_id: int, points: int):
    conn = sqlite3.connect("your_database_file.db")  # یا اتصال PostgreSQL
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()

    cursor.execute("""
        INSERT INTO rlc_scores (user_id, total_points, last_updated)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id)
        DO UPDATE SET total_points = total_points + ?, last_updated = ?;
    """, (user_id, points, now, points, now))

    conn.commit()
    conn.close()

# ─── Bot Command Handlers ───────────────────────────────────────────────────
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command: greets the user and shows main menu."""
    lang = get_lang(context)

    welcome_text = {
        "fa": (
            "سلام! 👋\n"
            "من <b>ربات حقوقی RebLawBot</b> هستم.\n\n"
            "📌 روزانه می‌توانید <b>۱ سؤال حقوقی رایگان</b> بپرسید.\n"
            "💳 با تهیه اشتراک، به امکانات نامحدود و خدمات کامل دسترسی خواهید داشت.\n\n"
            "برای شروع، یکی از گزینه‌های زیر را انتخاب کنید:"
        ),
        "en": (
            "Hello! 👋\n"
            "I am <b>RebLawBot</b>, your smart legal assistant.\n\n"
            "📌 You can ask <b>1 free legal question per day</b>.\n"
            "💳 Buy a subscription to unlock unlimited access and premium features.\n\n"
            "Please choose an option from the menu:"
        ),
        "ku": (
            "سڵاو! 👋\n"
            "من <b>ڕۆبۆتی یاسایی RebLawBot</b>م.\n\n"
            "📌 ڕۆژانە دەتوانیت <b>یەک پرسیاری بەخۆراو</b> بپرسیت.\n"
            "💳 بە بەشداریکردن، بە هەموو تایبەتمەندییەکان دەست دەکەویت.\n\n"
            "تکایە یەکێک لە هەڵبژاردەکان دیاری بکە:"
        )
    }

    await update.message.reply_text(
        welcome_text.get(lang, welcome_text["fa"]),
        reply_markup=get_main_menu(lang),
        parse_mode=ParseMode.HTML
    )


quiz_questions = [
    {
        "id": 1,
        "question": "اگر شخصی مال غیر را بفروشد، چه جرمی مرتکب شده است؟",
        "options": ["کلاهبرداری", "خیانت در امانت", "فروش مال غیر", "سرقت"],
        "answer_index": 2,
    },
    {
        "id": 2,
        "question": "ماده ۱۰ قانون مدنی به چه اصلی اشاره دارد؟",
        "options": ["لزوم قراردادها", "اثر قرارداد نسبت به اشخاص ثالث", "تعارض منافع", "فسخ عقد"],
        "answer_index": 0,
    }
]

# مسیر دیتابیس SQLite
DB_PATH = "data/reblaw.db"  # مسیر مناسب با پروژه شما

async def play_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    today = datetime.date.today().isoformat()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()


    # ایجاد جدول در صورت نبود
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_scores (
            telegram_id INTEGER PRIMARY KEY,
            score INTEGER DEFAULT 0,
            last_played TEXT
        )
    """)

    # اتصال به دیتابیس
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ایجاد جدول در صورت نبود
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_scores (
            telegram_id INTEGER PRIMARY KEY,
            score INTEGER DEFAULT 0,
            last_played TEXT
        )
    """)

    # گرفتن اطلاعات کاربر
    cursor.execute("SELECT score, last_played FROM user_scores WHERE telegram_id = ?", (user_id,))
    row = cursor.fetchone()

    if row:
        score, last_played = row
        if last_played == today:
            await update.message.reply_text("📌 شما امروز بازی کرده‌اید. لطفاً فردا دوباره امتحان کنید.")
            conn.close()
            return
    else:
        cursor.execute("INSERT INTO user_scores (telegram_id) VALUES (?)", (user_id,))
        conn.commit()

    # انتخاب یک سؤال تصادفی
    import random
    question = random.choice(quiz_questions)
    context.user_data["current_question"] = question  # ذخیره برای بررسی جواب

    # ساخت دکمه‌ها
    keyboard = [
        [InlineKeyboardButton(opt, callback_data=f"quiz:{i}")]
        for i, opt in enumerate(question["options"])
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(f"🧠 سوال:\n\n{question['question']}", reply_markup=reply_markup)
    conn.close()


async def quiz_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    if "current_question" not in context.user_data:
        await query.edit_message_text("❗ پرسشی برای شما فعال نیست.")
        return

    question = context.user_data["current_question"]
    selected = int(query.data.split(":")[1])
    correct = question["answer_index"]
    is_correct = selected == correct

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if is_correct:
        cursor.execute(
            "UPDATE user_scores SET score = score + 10, last_played = ? WHERE telegram_id = ?",
            (datetime.date.today().isoformat(), user_id)
        )
        response = "✅ پاسخ درست بود! ۱۰ امتیاز گرفتید."
    else:
        cursor.execute(
            "UPDATE user_scores SET last_played = ? WHERE telegram_id = ?",
            (datetime.date.today().isoformat(), user_id)
        )
        response = f"❌ پاسخ نادرست بود. جواب درست: {question['options'][correct]}"

    conn.commit()
    conn.close()

    await query.edit_message_text(response)

async def redeem_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # بررسی وجود جدول امتیازات
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_scores (
            telegram_id INTEGER PRIMARY KEY,
            score INTEGER DEFAULT 0,
            last_played TEXT
        )
    """)

    # بررسی وجود جدول اشتراک‌ها
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subscriptions (
            telegram_id INTEGER PRIMARY KEY,
            expires_at TEXT
        )
    """)

    # گرفتن امتیاز فعلی کاربر
    cursor.execute("SELECT score FROM user_scores WHERE telegram_id = ?", (user_id,))
    row = cursor.fetchone()

    if not row:
        await update.message.reply_text("❌ شما هنوز هیچ امتیازی ندارید.")
        conn.close()
        return

    score = row[0]

    if score < 100:
        await update.message.reply_text(f"📉 امتیاز شما {score} است. برای دریافت اشتراک حداقل ۱۰۰ امتیاز لازم است.")
        conn.close()
        return

    # کم کردن امتیاز و افزودن اشتراک ۷ روزه
    cursor.execute("UPDATE user_scores SET score = score - 100 WHERE telegram_id = ?", (user_id,))

    new_expiry = datetime.now() + timedelta(days=7)
    cursor.execute(
        "INSERT OR REPLACE INTO subscriptions (telegram_id, expires_at) VALUES (?, ?)",
        (user_id, new_expiry.isoformat())
    )

    conn.commit()
    conn.close()

    await update.message.reply_text("✅ تبریک! اشتراک ۷ روزه برای شما فعال شد و ۱۰۰ امتیاز کسر گردید.")


async def score_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # بررسی وجود جدول امتیاز
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_scores (
            telegram_id INTEGER PRIMARY KEY,
            score INTEGER DEFAULT 0,
            last_played TEXT
        )
    """)

    # خواندن امتیاز
    cursor.execute("SELECT score FROM user_scores WHERE telegram_id = ?", (user_id,))
    row = cursor.fetchone()

    conn.close()

    if not row:
        await update.message.reply_text("📉 شما هنوز هیچ امتیازی ندارید.")
    else:
        score = row[0]
        await update.message.reply_text(f"🎯 امتیاز فعلی شما: {score} امتیاز")


async def buy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /buy command: show subscription purchase information."""
    lang = get_lang(context)

    ton_wallet = getenv_or_die("TON_WALLET_ADDRESS")
    bank_card = getenv_or_die("BANK_CARD_NUMBER")
    rlc_wallet = os.getenv("RLC_WALLET_ADDRESS", "N/A")

    messages = {
        "fa": (
            "🛒 <b>خرید اشتراک</b>\n\n"
            "📌 شما روزانه <b>۱ سؤال حقوقی رایگان</b> در اختیار دارید.\n"
            "برای دسترسی نامحدود، لطفاً اشتراک تهیه فرمایید:\n\n"
            "💳 <b>کارت بانکی:</b> ۳۰۰,۰۰۰ تومان\n"
            f"🏦 شماره کارت: <code>{bank_card}</code>\n"
            f"👤 به‌نام: <b>ریبوار توفیقی</b>\n\n"
            "💎 <b>پرداخت با TON:</b> ۰٫۵ TON\n"
            f"👛 آدرس کیف پول TON: <code>{ton_wallet}</code>\n\n"
            "🚀 <b>توکن RLC:</b> ۱٬۰۰۰٬۰۰۰ RLC\n"
            f"🔗 آدرس کیف پول RLC: <code>{rlc_wallet}</code>\n\n"
            "✅ پس از پرداخت، از دستور /send_receipt برای ارسال رسید استفاده کنید."
        ),
        "en": (
            "🛒 <b>Buy Subscription</b>\n\n"
            "📌 You can ask <b>1 legal question for free each day</b>.\n"
            "To unlock unlimited access, please purchase a subscription:\n\n"
            "💳 <b>Bank Card (IRR):</b> 300,000 Toman\n"
            f"🏦 Card Number: <code>{bank_card}</code>\n"
            f"👤 Name: <b>Rebwar Tofiqi</b>\n\n"
            "💎 <b>TON Payment:</b> 0.5 TON\n"
            f"👛 Wallet: <code>{ton_wallet}</code>\n\n"
            "🚀 <b>RLC Token:</b> 1,000,000 RLC\n"
            f"🔗 Wallet Address: <code>{rlc_wallet}</code>\n\n"
            "✅ After payment, use /send_receipt to submit your receipt."
        ),
        "ku": (
            "🛒 <b>کڕینی بەشداریکردن</b>\n\n"
            "📌 ڕۆژانە دەتوانیت <b>یەک پرسیاری بەخۆراکەت</b> بپرسیت.\n"
            "بۆ بەدەستهێنانی دەستگیشتی بێ سنوور، تکایە بەشداریکردن بکە:\n\n"
            "💳 <b>کارتی بانکی:</b> ٣٠٠,٠٠٠ تومان\n"
            f"🏦 ژمارەی کارت: <code>{bank_card}</code>\n"
            f"👤 ناوی خاوەن کارت: <b>ریبوار توفیقی</b>\n\n"
            "💎 <b>پارەدان بە TON:</b> ٠.٥ TON\n"
            f"👛 ناونیشانی جزدان: <code>{ton_wallet}</code>\n\n"
            "🚀 <b>تۆکینی RLC:</b> ١,٠٠٠,٠٠٠ RLC\n"
            f"🔗 ناونیشانی RLC: <code>{rlc_wallet}</code>\n\n"
            "✅ دوای پارەدان، فەرمانی /send_receipt بەکاربێنە بۆ ناردنی پسوڵە."
        ),
    }

    await update.message.reply_text(
        messages[lang],
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True
    )



async def send_receipt_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /send_receipt command: prompt user to send a receipt (photo or text)."""
    lang = get_lang(context)
    # Mark that we are now expecting a receipt from the user
    context.user_data["awaiting_receipt"] = True
    await update.message.reply_text(tr("send_receipt_prompt", lang))  # send_receipt_prompt text from TEXTS



async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
 
    uid = update.effective_user.id
    lang = get_lang(context)

    sub_expiry = get_user_subscription_expiry(uid)
    credits = get_credits(uid)

    if sub_expiry and sub_expiry > datetime.utcnow():
        remaining = sub_expiry - datetime.utcnow()
        days = remaining.days
        msg = {
            "fa": f"✅ شما اشتراک فعال دارید.\n"
                  f"🗓️ اعتبار تا <b>{sub_expiry.strftime('%Y-%m-%d')}</b> (تقریباً {days} روز دیگر)\n"
                  f"📌 همچنین روزانه ۱ اعتبار رایگان نیز فعال است.",
            "en": f"✅ You have an active subscription.\n"
                  f"🗓️ Valid until <b>{sub_expiry.strftime('%Y-%m-%d')}</b> (~{days} days left)\n"
                  f"📌 You also receive 1 free daily credit.",
            "ku": f"✅ تۆ بەشداریکردنەکی چالاکت هەیە.\n"
                  f"🗓️ بەردەوامە تا <b>{sub_expiry.strftime('%Y-%m-%d')}</b> ({days} ڕۆژ باقیە)\n"
                  f"📌 هەروەها ڕۆژانە ١ کرێدیتت دەدرێت."
        }
    else:
        msg = {
            "fa": f"⚠️ شما اشتراک فعال ندارید.\n"
                  f"📊 اعتبار رایگان باقی‌ماندهٔ امروز: <b>{credits}</b> پرسش\n\n"
                  f"💡 برای دسترسی کامل می‌توانید اشتراک بخرید با /buy",
            "en": f"⚠️ You don't have an active subscription.\n"
                  f"📊 Your free credits left for today: <b>{credits}</b> question(s)\n\n"
                  f"💡 Use /buy to unlock unlimited access.",
            "ku": f"⚠️ بەشداریکردنەکی چالاکت نییە.\n"
                  f"📊 کرێدیتە بەخۆراکەت بۆ ئەمڕۆ: <b>{credits}</b> پرسیار\n\n"
                  f"💡 فەرمانی /buy بەکاربێنە بۆ بەدەستهێنانی دەستگیشتی."
        }

    await update.message.reply_text(msg[lang], parse_mode=ParseMode.HTML)

logger = logging.getLogger(__name__)

def find_law_article(article_number: int, law_name: str) -> str:
    """جستجو در پایگاه داده برای یافتن متن یک ماده خاص از یک قانون مشخص"""
    try:
        with sqlite3.connect("laws.db") as conn:
            row = conn.execute(
                "SELECT text FROM laws WHERE number=? AND law LIKE ? COLLATE NOCASE",
                (article_number, f"%{law_name}%")
            ).fetchone()
            return row[0] if row else ""
    except Exception as e:
        logger.error("Error in find_law_article: %s", e)
        return ""

async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /ask command: forward the question to OpenAI if user has credit or active subscription."""
    uid = update.effective_user.id
    lang = get_lang(context)
    question = " ".join(context.args).strip()

    # No question provided
    if not question:
        await update.message.reply_text({
            "fa": "❓ لطفاً سؤال را بعد از دستور بنویسید.",
            "en": "❓ Please write your legal question after the command.",
            "ku": "❓ تکایە پرسیارت لە دوای فەرمانەکە بنوسە."
        }.get(lang, "❓ لطفاً سؤال را بعد از دستور بنویسید."))
        return

    # Try to detect legal article request
    article_match = re.search(r"(?i)(ماده\s+(\d+)\s+قانون\s+([\w\s]+))", question)
    article_text = ""
    if article_match:
        article_number = article_match.group(2)
        law_name = article_match.group(3).strip()
        article_text = find_law_article(int(article_number), law_name)
        if article_text:
            await update.message.reply_text(f"📘 {article_match.group(1)}:\n\n{article_text}")

    # Check user access (subscription or credits)
    is_subscriber = has_active_subscription(uid)
    if not is_subscriber:
        credits = get_credits(uid)
        if credits <= 0:
            await update.message.reply_text({
                "fa": "⛔ شما اعتبار فعال برای پرسش ندارید.\n\n📌 روزانه فقط ۱ سؤال رایگان مجاز است.\nبرای مشاهده وضعیت از <b>/credits</b> استفاده کنید.",
                "en": "⛔ You don't have active credits to ask a question.\n\n📌 Only 1 free legal question is allowed per day.\nUse <b>/credits</b> to check your status.",
                "ku": "⛔ تۆ کرێدیتت نییە بۆ پرسیار.\n\n📌 ڕۆژانە تەنها یەک پرسیاری بەخۆراو دەکرێت.\nفەرمانی <b>/credits</b> بەکاربێنە.",
            }.get(lang),
            parse_mode=ParseMode.HTML)
            return

    # If question is only a request for article text, skip OpenAI
    if article_text and len(question.strip().split()) < 6:
        return  # don't send to OpenAI unless it's a full legal question



    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        answer_text = await client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {
                    "role": "system",
                    "content": "You are an experienced Iranian lawyer. Answer in formal Persian." if lang == "fa"
                    else "You are an experienced lawyer. Answer clearly."
                },
                {"role": "user", "content": question}
            ],
            temperature=0.6,
            max_tokens=1024
        )
        answer = answer_text.choices[0].message.content.strip()
    except (APIError, RateLimitError, AuthenticationError) as e:
        logger.error("OpenAI API error: %s", e)
        answer = tr("openai_error", lang) if "openai_error" in TEXTS else "❗️Service is unavailable. Please try again later."

    # Send answer in parts if long
    for part in [answer[i:i+4000] for i in range(0, len(answer), 4000)]:
        await update.message.reply_text(part)


    await update.message.reply_text({
        "fa": "✅ پاسخ ارسال شد.",
        "en": "✅ Answer sent.",
        "ku": "✅ وەڵام نێردرا."
    }.get(lang))


    if not is_subscriber:
        decrement_credits(uid)





async def about_token(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show information about the RLC token with image and purchase link."""
    lang = get_lang(context)
    message = update.effective_message

    # مسیر عکس: reblawcoin.png در همان پوشه فایل bot.py
    token_img = Path("reblawcoin.png")
    has_img = token_img.exists()

    # متن چندزبانه با لینک خرید
    token_texts = {
        "fa": (
            "🎉 <b>توکن RebLawCoin (RLC)</b> – اولین ارز دیجیتال با محوریت خدمات حقوقی.\n\n"
            "📌 اهداف پروژه:\n"
            "• سرمایه‌گذاری در نوآوری‌های حقوقی\n"
            "• نهادینه‌سازی عدالت روی بلاک‌چین\n"
            "• سودآوری پایدار برای سرمایه‌گذاران\n\n"
            "🛒 <a href='https://t.me/blum/app?startapp=memepadjetton_RLC_JpMH5-ref_1wgcKkl94N'>خرید توکن RLC در بلوُم</a>"
        ),
        "en": (
            "🎉 <b>RebLawCoin (RLC)</b> – the first crypto token focused on legal innovation.\n\n"
            "📌 Project goals:\n"
            "• Invest in legal tech\n"
            "• Decentralize justice\n"
            "• Enable sustainable value for holders\n\n"
            "🛒 <a href='https://t.me/blum/app?startapp=memepadjetton_RLC_JpMH5-ref_1wgcKkl94N'>Buy RLC Token on Bloom</a>"
        ),
        "ku": (
            "🎉 <b>تۆکینی RebLawCoin (RLC)</b> – یەکەم تۆکن بۆ نوێکاری یاسایی.\n\n"
            "📌 ئامانجی پرۆژە:\n"
            "• پانگە دان بە هەژماری یاسایی\n"
            "• دادپەروەرییەکی دەسەڵات‌ناوەندی\n"
            "• بەهای بەردەوام بۆ هەژمارگیران\n\n"
            "🛒 <a href='https://t.me/blum/app?startapp=memepadjetton_RLC_JpMH5-ref_1wgcKkl94N'>کڕینی تۆکین لە Bloom</a>"
        )
    }

    content = token_texts.get(lang, token_texts["fa"])

    # اگر عکس وجود داشت، اول عکس بفرست
    if has_img:
        await message.reply_photo(token_img.open("rb"), caption="📌 RebLawCoin (RLC)", parse_mode=ParseMode.HTML)

    # سپس متن اطلاعات و لینک را بفرست
    await message.reply_text(content, parse_mode=ParseMode.HTML, disable_web_page_preview=False)



async def lang_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /lang command: show language selection keyboard."""
    await update.message.reply_text(
        "لطفاً زبان مورد نظر را انتخاب کنید:\nPlease select your preferred language:\nتکایە زمانت هەلبژێرە:",
        reply_markup=ReplyKeyboardMarkup([[KeyboardButton("فارسی"), KeyboardButton("English"), KeyboardButton("کوردی")]], one_time_keyboard=True, resize_keyboard=True)
    )


async def case_page_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """جابجایی بین صفحات پرونده‌های مشهور"""
    query = update.callback_query
    await query.answer()
    try:
        page = int(query.data.split(":")[1])
    except Exception:
        page = 0
    await show_case_page(update, context, page=page)


async def cases_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /cases command: redirect to show first page of famous cases."""
    await show_case_page(update, context, page=0)

CASES_PER_PAGE = 5  # تعداد پرونده در هر صفحه


async def show_case_page(update_or_query, context: ContextTypes.DEFAULT_TYPE, page: int) -> None:
    lang = get_lang(context)
    all_cases = get_famous_cases()
    total = len(all_cases)
    pages = (total + CASES_PER_PAGE - 1) // CASES_PER_PAGE
    page = max(0, min(page, pages - 1))  # محدود کردن بازه صفحه

    # Slice پرونده‌ها برای صفحه فعلی
    start = page * CASES_PER_PAGE
    end = start + CASES_PER_PAGE
    cases = all_cases[start:end]

    # دکمه‌ها
    buttons = [
        [InlineKeyboardButton(title, callback_data=f"case:{cid}")]
        for cid, title in cases
    ]

    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton("⬅️ قبلی", callback_data=f"case_page:{page - 1}"))
    if page < pages - 1:
        nav_buttons.append(InlineKeyboardButton("➡️ بعدی", callback_data=f"case_page:{page + 1}"))
    if nav_buttons:
        buttons.append(nav_buttons)

    message = {
        "fa": f"📚 فهرست پرونده‌های مشهور (صفحه {page + 1} از {pages}):",
        "en": f"📚 Famous Cases (Page {page + 1} of {pages}):",
        "ku": f"📚 پرۆسەی ناودار (لاپەڕەی {page + 1} لە {pages}):"
    }.get(lang, "📚 پرونده‌ها:")

    if isinstance(update_or_query, Update) and update_or_query.message:
        await update_or_query.message.reply_text(message, reply_markup=InlineKeyboardMarkup(buttons))
    elif update_or_query.callback_query:
        await update_or_query.callback_query.edit_message_text(
            message, reply_markup=InlineKeyboardMarkup(buttons)
        )


# ─── Callback Query Handlers ────────────────────────────────────────────────
async def case_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process the inline button for a case (from /cases list) and send the case summary."""
    query = update.callback_query
    await query.answer()  # acknowledge the callback
    if not query.data or not query.data.startswith("case:"):
        return  # not a case query
    case_id = int(query.data.split(":")[1])
    summary = get_case_summary(case_id)
    if summary:
        # Send only up to 4000 chars to avoid Telegram message limit
        await query.message.reply_text(f"📝 خلاصه:\n\n{summary[:4000]}")
    else:
        await query.message.reply_text("❌ خطا در دریافت پرونده." if get_lang(context) == "fa" else "❌ Failed to retrieve case summary.")


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process inline buttons for approving or rejecting a subscription receipt (admin only)."""
    query = update.callback_query
    await query.answer()
    
    try:
        action, uid_str = query.data.split(":")
        target_uid = int(uid_str)
    except Exception:
        # Invalid data format in callback
        await query.answer("داده دکمه نامعتبر است." if get_lang(context) == "fa" else "Invalid button data.", show_alert=True)
        return
    # Ensure only the admin can approve/reject receipts
    admin_id = int(getenv_or_die("ADMIN_ID"))
    if update.effective_user.id != admin_id:
        await query.answer("فقط مدیر می‌تواند این کار را انجام دهد." if get_lang(context) == "fa" else "Only the admin can perform this action.", show_alert=True)
        return
    # Perform the requested action (approve or reject)
    if action == "approve":
        save_subscription(target_uid, days=int(os.getenv("SUBSCRIPTION_DAYS", "30") or 30))
        await context.bot.send_message(chat_id=target_uid, text="🎉 اشتراک شما تأیید شد." if get_lang(context) == "fa" else "🎉 Your subscription has been approved.")
        status_text = "✔️ تأیید شد"  # Approved (in Persian)
    else:  # "reject"
        set_user_status(target_uid, "rejected")
        await context.bot.send_message(chat_id=target_uid, text="❌ رسید شما رد شد." if get_lang(context) == "fa" else "❌ Your receipt was rejected.")
        status_text = "❌ رد شد"  # Rejected (in Persian)
    # Update the admin's message (the one with receipt and buttons) to reflect the decision
    try:
        updated_caption = (query.message.caption or query.message.text or "") + f"\n\n<b>وضعیت:</b> {status_text}"
        if query.message.photo:
            # If the message was a photo with a caption
            await query.message.edit_caption(updated_caption, parse_mode=ParseMode.HTML)
        else:
            await query.message.edit_text(updated_caption, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error("Failed to edit admin message status: %s", e)


# ─── Message Handlers (non-command messages) ────────────────────────────────
async def lang_text_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle language selection from the custom keyboard (text messages 'فارسی', 'English', 'کوردی')."""
    choice = (update.message.text or "").strip()
    # Set language based on user's choice
    if choice in ["فارسی", "Farsi", "Persian"]:
        context.user_data["lang"] = "fa"
        await update.message.reply_text("زبان شما فارسی تنظیم شد." if choice == "فارسی" else "Language set to Persian.")
    elif choice in ["English", "انگلیسی"]:
        context.user_data["lang"] = "en"
        await update.message.reply_text("Language changed to English.")
    elif choice in ["کوردی", "Kurdish"]:
        context.user_data["lang"] = "ku"
        await update.message.reply_text("زمانت کرا بە کوردی." if choice == "کوردی" else "Language set to Kurdish.")


async def handle_receipt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle an incoming receipt (photo or text message) after /send_receipt was used."""
    msg: Message = update.message
    uid = update.effective_user.id
    # Only proceed if we asked for a receipt (awaiting_receipt) or the message is a photo.
    if not context.user_data.get("awaiting_receipt") and not msg.photo:
        # If not expecting a receipt and it's not a photo, ignore this message.
        return
    # Once we handle this message, reset the flag
    context.user_data["awaiting_receipt"] = False
    # Upsert user info in DB (ensure user exists in DB)
    upsert_user(uid, msg.from_user.username, msg.from_user.first_name, msg.from_user.last_name)
    # Save the receipt data (photo file_id or text content)
    photo_id = msg.photo[-1].file_id if msg.photo else None
    receipt_data = photo_id if photo_id else (msg.text or "<empty>")
    save_receipt_request(uid, receipt_data)
    # Prepare inline buttons for admin approval
    approve_button = InlineKeyboardButton("✅ تأیید", callback_data=f"approve:{uid}")
    reject_button  = InlineKeyboardButton("❌ رد", callback_data=f"reject:{uid}")
    admin_kb = InlineKeyboardMarkup([[approve_button, reject_button]])
    # Prepare caption for the admin's message
    caption = (
        f"📩 رسید جدید از <a href='tg://user?id={uid}'>{msg.from_user.full_name}</a>\n"
        f"نام کاربری: @{msg.from_user.username or 'بدون'}\n\nبررسی مدیر:"
    )
    # Send the receipt to admin (photo or text)
    admin_chat_id = int(getenv_or_die("ADMIN_ID"))
    if photo_id:
        await context.bot.send_photo(chat_id=admin_chat_id, photo=photo_id, caption=caption, reply_markup=admin_kb, parse_mode=ParseMode.HTML)

    else:
        # If it's text receipt, include the text in the message body

        await context.bot.send_message(chat_id=admin_chat_id, text=f"{caption}\n\n{msg.text}", reply_markup=admin_kb, parse_mode=ParseMode.HTML)

    # Acknowledge to the user that their receipt was sent for review

    await msg.reply_text("✅ رسید شما ارسال شد. لطفاً منتظر تأیید مدیر بمانید." if get_lang(context) == "fa" else "✅ Your receipt has been sent. Please wait for admin approval.")




async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
 
    text = (update.message.text or "").strip()
    lang = get_lang(context)


    if lang == "fa":
        if text == "🛒 خرید اشتراک":
            await buy_cmd(update, context)
        elif text == "📤 ارسال رسید":
            await send_receipt_cmd(update, context)
        elif text == "⚖️ سؤال حقوقی":
            await update.message.reply_text(
                "💬 لطفاً سؤال خود را با دستور /ask ارسال نمایید.\n"
                "مثال:\n<code>/ask قرارداد مشارکت چه شرایطی دارد؟</code>\n\n"
                "📌 هر کاربر روزانه ۱ سؤال رایگان دارد.\n"
                "برای مشاهده اعتبار خود از دستور /credits استفاده کنید.",
                parse_mode=ParseMode.HTML
            )
      
        elif text == "🎤 سؤال صوتی":
            await update.message.reply_text(
                "🎙️ لطفاً سؤال خود را به صورت پیام صوتی (voice) ارسال نمایید.\n\n📌 فقط پیام صوتی تلگرام پشتیبانی می‌شود."
            )
        elif text == "ℹ️ درباره توکن":
            await about_token(update, context)
      
        elif text == "📚 پرونده‌های مشهور":
            await cases_cmd(update, context)

    elif lang == "en":
        if text == "🛒 Buy Subscription":
            await buy_cmd(update, context)
        elif text == "📤 Send Receipt":
            await send_receipt_cmd(update, context)
        elif text == "⚖️ Legal Question":
            await update.message.reply_text(
                "💬 Please send your legal question using the /ask command.\n"
                "Example:\n<code>/ask What are the conditions for a partnership contract?</code>\n\n"
                "📌 You have 1 free legal question per day.\n"
                "Use /credits to check your remaining credit.",
                parse_mode=ParseMode.HTML
            )
     
        elif text == "🎤 Voice Question":
            await update.message.reply_text(
                "🎙️ Please send your legal question as a Telegram voice message.\n\n📌 Only Telegram voice messages are supported."
            )
        elif text == "ℹ️ About Token":
            await about_token(update, context)
        elif text == "📚 Famous Cases":
            await cases_cmd(update, context)

    elif lang == "ku":
        if text == "🛒 کڕینی بەشداریکردن":
            await buy_cmd(update, context)
        elif text == "📤 ناردنی پسوڵە":
            await send_receipt_cmd(update, context)
        elif text == "⚖️ پرسیاری یاسایی":
            await update.message.reply_text(
                "💬 تکایە پرسیارەکەت بنێرە بە فەرمانی /ask.\n"
                "نموونە:\n<code>/ask پەیوەندی هاوبەش چییە؟</code>\n\n"
                "📌 ڕۆژانە ١ پرسیاری بەخۆراکەت هەیە.\n"
                "بۆ بینینی ماوەی کرێدیتت /credits بنووسە.",
                parse_mode=ParseMode.HTML
            )
        elif text == "🎤 پرسیاری دەنگی":
            await update.message.reply_text(
                "🎙️ تکایە پرسیارەکەت بە شێوەی پەیامی دەنگی بنێرە.\n\n📌 تەنها پەیامەکانی دەنگی تێلەگرام پشتیوانی دەکرێن."
            )
        elif text == "ℹ️ دەربارەی تۆکێن":
            await about_token(update, context)
        elif text == "📚 پرۆسەی ناسراو":
            await cases_cmd(update, context)

    else:
        await update.message.reply_text("❌ دستور نامعتبر است. لطفاً از منو استفاده کنید.")


   
    # If text doesn't match any known command or menu option, we could handle it (e.g., ask AI directly if subscribed).
    # For now, do nothing or send a default message:
    # else:
    #     await update.message.reply_text("I'm not sure how to respond to that. Use /help for commands.")


async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming voice messages (convert to text and answer if subscribed)."""
    uid = update.effective_user.id
    lang = get_lang(context)
    if not has_active_subscription(uid):

        # Only allow voice questions if subscribed

        await update.message.reply_text(tr("no_sub", lang))
        return

    # Download the voice file

    voice_file = await update.message.voice.get_file()
    temp_dir = tempfile.mkdtemp()
    ogg_path = os.path.join(temp_dir, "voice.ogg")
    wav_path = os.path.join(temp_dir, "voice.wav")
    await voice_file.download_to_drive(ogg_path)

    # Convert OGG (Telegram voice format) to WAV using ffmpeg

    try:
        ffmpeg.input(ogg_path).output(wav_path).run(quiet=True, overwrite_output=True)
    except Exception as e:
        logger.error("FFmpeg conversion error: %s", e)
        await update.message.reply_text("❌ خطا در پردازش پیام صوتی." if lang == "fa" else "❌ Error processing voice message.")
        return

    # Transcribe voice to text

    try:
        question_text = voice_to_text(wav_path)
    except Exception as e:
        logger.error("Whisper transcription error: %s", e)
        await update.message.reply_text("❌ خطا در تبدیل صدا به متن." if lang == "fa" else "❌ Could not transcribe the voice message.")
        return

    # Now answer the question using the same logic as /ask

    await update.message.reply_text("🎙️❓ " + question_text)  # Echo the transcribed question to user (optional)

    # Use the ask_cmd logic to get answer

    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        answer_text = await client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": "You are an experienced lawyer. Answer clearly." if lang != "fa" else "You are an experienced Iranian lawyer. Answer in formal Persian."},
                {"role": "user", "content": question_text}
            ],
            temperature=0.6,
            max_tokens=1024
        )
        answer = answer_text.choices[0].message.content.strip()
    except (APIError, RateLimitError, AuthenticationError) as e:
        logger.error("OpenAI API error (voice question): %s", e)
        answer = tr("openai_error", lang) if "openai_error" in TEXTS else "❗️Service is unavailable. Please try again later."

    # Send answer (split into parts if too long)

    parts = [answer[i:i+4000] for i in range(0, len(answer), 4000)]
    for part in parts:
        await update.message.reply_text(part)
    # Inform user they can ask another
    await update.message.reply_text({
        "fa": "✅ پاسخ ارسال شد. در صورت نیاز می‌توانید سؤال صوتی دیگری بفرستید.",
        "en": "✅ Answer sent. You may send another voice question if needed.",
        "ku": "✅ وەڵام نێردرا. دەتوانیت پرسیاری دەنگییەکی تر بنێریت."
    }[lang])


@admin_only
async def list_users_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Admin-only handler for the /users command.
    این تابع آخرین ۱۰ کاربر ثبت‌شده را بر اساس user_id نزولی نمایش می‌دهد.
    """
    try:
        with get_db() as conn:
            cursor = conn.cursor()

            # کوئریِ اصلی با ستون‌های user_id, username, first_name, last_name, status
            cursor.execute("""
                SELECT user_id, username, first_name, last_name, status
                  FROM users
                 ORDER BY user_id DESC
                 LIMIT 10
            """)
            rows = cursor.fetchall()  # لیستی از تاپل‌ها (tuple) برمی‌گردد

        # اگر هیچ کاربری ثبت نشده باشد
        if not rows:
            await update.message.reply_text("📭 هیچ کاربری در سیستم ثبت نشده است.")
            return

        # آماده‌سازی متن پاسخ با دسترسی به ستون‌ها از طریق ایندکس عددی
        text = "📋 <b>آخرین ۱۰ کاربر ثبت‌شده:</b>\n\n"
        for row in rows:
            # ترتیب ستون‌ها بر اساس کوئری:
            # row[0] = user_id
            # row[1] = username
            # row[2] = first_name
            # row[3] = last_name
            # row[4] = status
            uid   = row[0]
            uname = row[1] or "—"
            fname = row[2] or ""
            lname = row[3] or ""
            status = row[4] or "—"

            text += (
                f"👤 <code>{uid}</code> — @{uname}\n"
                f"   نام: {fname} {lname}\n"
                f"   وضعیت: <b>{status}</b>\n\n"
            )

        # ارسال پیام نهایی با parse_mode=HTML
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    except Exception:
        # ثبت استک‌تریس کامل در لاگ
        logger.exception("خطا در اجرای /users")
        # ارسال پیام عمومی خطا به مدیر
        await update.message.reply_text("❌ خطا در دریافت لیست کاربران. لطفاً بعداً دوباره تلاش کنید.")


async def credits_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /credits command: show user's remaining question credits and subscription status."""
    uid = update.effective_user.id
    lang = get_lang(context)

    credits = get_credits(uid)
    expire_at = get_user_subscription_expiry(uid)
    now = datetime.now()

    # آیا اشتراک فعال است؟
    is_subscribed = expire_at is not None and expire_at > now

    if credits > 0:
        msg = {
            "fa": (
                "✅ شما <b>۱ سؤال رایگان</b> برای امروز دارید.\n"
                f"{'📅 اشتراک شما فعال است تا تاریخ ' + expire_at.strftime('%Y-%m-%d') if is_subscribed else 'ℹ️ شما اشتراک فعال ندارید.'}"
            ),
            "en": (
                "✅ You have <b>1 free legal question</b> remaining today.\n"
                f"{'📅 Your subscription is active until ' + expire_at.strftime('%Y-%m-%d') if is_subscribed else 'ℹ️ You don’t have an active subscription.'}"
            ),
            "ku": (
                "✅ تۆ <b>یەک پرسیاری بەخۆراو</b>ت هەیە بۆ ئەمڕۆ.\n"
                f"{'📅 بەشداریکردنەکەت چالاکە تا ' + expire_at.strftime('%Y-%m-%d') if is_subscribed else 'ℹ️ بەشداریکردنی چالاک نییە.'}"
            ),
        }
    else:
        msg = {
            "fa": (
                "⛔ شما امروز از سؤال رایگان استفاده کرده‌اید.\n"
                f"{'📅 اشتراک شما فعال است تا تاریخ ' + expire_at.strftime('%Y-%m-%d') if is_subscribed else 'ℹ️ شما اشتراک فعال ندارید.'}\n\n"
                "📌 می‌توانید فردا دوباره سؤال رایگان بپرسید یا اشتراک تهیه کنید."
            ),
            "en": (
                "⛔ You’ve used your free legal question today.\n"
                f"{'📅 Your subscription is active until ' + expire_at.strftime('%Y-%m-%d') if is_subscribed else 'ℹ️ You don’t have an active subscription.'}\n\n"
                "📌 You can ask again tomorrow or purchase a subscription."
            ),
            "ku": (
                "⛔ تۆ پێشتر پرسیاری بەخۆراوی ئەمڕۆت بەکارهێنا.\n"
                f"{'📅 بەشداریکردنەکەت چالاکە تا ' + expire_at.strftime('%Y-%m-%d') if is_subscribed else 'ℹ️ بەشداریکردنی چالاک نییە.'}\n\n"
                "📌 دەتوانیت سبەی پرسیاری تر بکەیت یان بەشداریکردن بکەیت."
            ),
        }

    await update.message.reply_text(
        msg.get(lang, msg["fa"]),
        parse_mode=ParseMode.HTML
    )


async def handle_webapp_data(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """داده ارسال‌شده از WebApp (بازی) را پردازش و برای مدیر ارسال می‌کند."""
    if not update.effective_user or not update.effective_user.id:
        return

    uid = update.effective_user.id
    webapp_data = getattr(update.effective_message, "web_app_data", None)

    if not webapp_data or not webapp_data.data:
        logger.warning("⛔️ No web_app_data received.")
        return

    try:
        parsed = json.loads(webapp_data.data)
        if parsed.get("type") == "submit_argument":
            case_id = parsed.get("caseId")
            role = parsed.get("role")
            text = parsed.get("text")

            message = (
                f"🧠 <b>دفاعیه جدید از بازی</b>\n"
                f"👤 کاربر: <code>{uid}</code>\n"
                f"📂 پرونده: {case_id}\n"
                f"🎭 نقش: {role}\n"
                f"📝 متن:\n{text}"
            )

            for admin_id in ADMIN_IDS:
                await context.bot.send_message(chat_id=admin_id, text=message, parse_mode="HTML")

            await update.effective_message.reply_text("✅ دفاعیه شما با موفقیت ارسال شد. منتظر بررسی باشید.")
        else:
            logger.info(f"⚠️ Unknown WebApp data type: {parsed.get('type')}")
    except Exception as e:
        logger.error(f"❌ خطا در پردازش web_app_data: {e}")
        await update.effective_message.reply_text("⚠️ خطایی در پردازش داده‌ها رخ داد.")

    except Exception as e:
        await update.effective_message.reply_text("❌ خطا در پردازش داده ارسال‌شده.")
        print(f"Error parsing WebAppData: {e}")


async def handle_decision_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """پردازش تصمیم مدیر (پذیرش یا رد دفاعیه ارسال‌شده از WebApp)."""
    query = update.callback_query
    await query.answer()

    data = query.data
    uid_match = re.search(r"user_(\d+)", data)
    if not uid_match:
        await query.edit_message_text("❌ شناسه کاربر نامعتبر است.")
        return

    uid = int(uid_match.group(1))

    if data.startswith("approve_"):
        # ✅ افزودن امتیاز RLC
        add_rlc_score(user_id=uid, points=5)

        await context.bot.send_message(
            chat_id=uid,
            text="✅ دفاعیه شما توسط مدیر تأیید شد. ممنون از مشارکت شما!\n💎 شما ۵ امتیاز RLC دریافت کردید."
        )
        await query.edit_message_text("دفاعیه تأیید شد و ۵ امتیاز RLC به کاربر داده شد.")
        
    elif data.startswith("reject_"):
        await context.bot.send_message(
            chat_id=uid,
            text="❌ دفاعیه شما توسط مدیر رد شد. لطفاً در نوبت بعدی با دقت بیشتری تلاش کنید."
        )
        await query.edit_message_text("دفاعیه رد شد و به کاربر اطلاع داده شد.")
    else:
        await query.edit_message_text("❌ تصمیم ناشناخته.")


# ─── Register Handlers ──────────────────────────────────────────────────────
def register_handlers(app: Application) -> None:

    """Register all command and message handlers with the Application."""

    # Command handlers

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("buy", buy_cmd))
    app.add_handler(CommandHandler("send_receipt", send_receipt_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(CommandHandler("about_token", about_token))
    app.add_handler(CommandHandler("lang", lang_cmd))
    app.add_handler(CommandHandler("cases", cases_cmd))
    app.add_handler(CommandHandler("users", list_users_cmd))
    app.add_handler(CommandHandler("credits", credits_cmd))
    app.add_handler(CommandHandler("redeem", redeem_cmd))
    app.add_handler(CommandHandler("score", score_cmd))

    # Callback query handlers for inline buttons

    app.add_handler(CallbackQueryHandler(case_callback_handler, pattern=r"^case:\d+$"))
    app.add_handler(CallbackQueryHandler(callback_handler, pattern=r"^(approve|reject):\d+$"))
    app.add_handler(CallbackQueryHandler(case_page_callback_handler, pattern=r"^case_page:\d+$"))
    app.add_handler(CommandHandler("play", play_cmd))
    app.add_handler(CallbackQueryHandler(quiz_callback, pattern=r"^quiz:\d+$"))
    
    # Non-command message handlers (ordered by group to control priority)

    app.add_handler(MessageHandler(filters.Regex("^(فارسی|English|کوردی)$"), lang_text_router), group=0)
    app.add_handler(MessageHandler(filters.PHOTO | (filters.TEXT & ~filters.COMMAND), handle_receipt), group=1)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router), group=2)
    app.add_handler(MessageHandler(filters.VOICE, handle_voice_message), group=3)
    app.add_handler(MessageHandler(filters.StatusUpdate.WEB_APP_DATA, handle_webapp_data))
    app.add_handler(CallbackQueryHandler(handle_decision_callback, pattern="^(approve|reject)_user_\\d+$"))


# ─── Main Entrypoint ───────────────────────────────────────────────────────

def main() -> None:
    """Initialize the bot and start polling for updates."""
    bot_token = os.getenv("BOT_TOKEN")
    if not bot_token:
        raise RuntimeError("❌ BOT_TOKEN not found in environment.")
   
    # Initialize database (ensure tables are created before bot starts)
    init_db()  # ✅ Important: call before starting the bot
    # Build the Telegram Application
   
    application = Application.builder().token(bot_token).build()
   
    # Register all command and message handlers
   
    register_handlers(application)
   
    logger.info("🤖 RebLawBot started. Waiting for updates...")
    # Start the bot (polling Telegram for new updates)
    application.run_polling(allowed_updates=Update.ALL_TYPES)

# Run the bot if this script is executed directly
if __name__ == "__main__":
    main()
