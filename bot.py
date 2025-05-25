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

import tempfile

from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Generator, Optional

# External libraries
from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError, RateLimitError, AuthenticationError
from psycopg2.pool import SimpleConnectionPool
from telegram import Update, Message, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.constants import ChatAction, ParseMode
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters

from texts import TEXTS  # assuming texts.py provides translation strings

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
        # Fallback to SQLite if PostgreSQL fails
        logger.warning("PostgreSQL unavailable (%s), switching to SQLite.", exc)
        USE_PG = False
        _setup_schema_sqlite()
    # Update placeholder after determining DB type
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
    """
    assert POOL is not None  # pool should be set if USE_PG is True
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
    """
    # Acquire a lock to ensure thread-safety if accessed from multiple threads
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

def has_active_subscription(user_id: int) -> bool:
    """Check if the user has an active subscription (i.e., expire_at in the future and status='approved')."""
    row = _fetchone(
        f"SELECT expire_at FROM users WHERE user_id={_PLACEHOLDER} AND status='approved'",
        (user_id,)
    )
    if not row or row[0] is None:
        return False
    expire_at = row[0]  # In PG this might be a datetime, in SQLite a string
    if isinstance(expire_at, str):
        expire_at = datetime.fromisoformat(expire_at)
    return expire_at >= datetime.utcnow()

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

# ─── Bot Command Handlers ───────────────────────────────────────────────────
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command: greets the user and shows main menu."""
    lang = get_lang(context)
    welcome_text = {
        "fa": "سلام! 👋\nمن <b>ربات حقوقی RebLawBot</b> هستم.\nبا تهیه اشتراک می‌توانید سؤالات حقوقی خود را بپرسید.\nبرای شروع، یکی از گزینه‌های زیر را انتخاب کنید:",
        "en": "Hello! 👋\nI am <b>RebLawBot</b>, your legal assistant.\nPurchase a subscription to ask legal questions.\nPlease choose an option from the menu:",
        "ku": "سڵاو! 👋\nمن <b>ڕۆبۆتی یاسایی RebLawBot</b>م.\nبە بەشداربوون دەتوانیت پرسیاری یاساییت بکەیت.\nتکایە یەکێک لە هەڵبژاردەکان دیاری بکە:"
    }
    await update.message.reply_text(
        welcome_text.get(lang, welcome_text["fa"]),
        reply_markup=get_main_menu(lang),
        parse_mode=ParseMode.HTML
    )

async def buy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /buy command: show subscription purchase information."""
    lang = get_lang(context)
    # Fetch payment info from environment (wallet addresses, etc.)
    ton_wallet = getenv_or_die("TON_WALLET_ADDRESS")
    bank_card = getenv_or_die("BANK_CARD_NUMBER")
    rlc_wallet = os.getenv("RLC_WALLET_ADDRESS", "N/A")
    price_text = {
        "fa": (
            "🔸 قیمت اشتراک یک‌ماهه:\n\n"
            f"💳 کارت بانکی: 500,000 تومان\n🏦 شماره کارت: <code>{bank_card}</code>\n\n"
            f"💎 تون کوین (TON): 1\n👛 آدرس کیف پول: <code>{ton_wallet}</code>\n\n"
            f"🚀 توکن RLC: 1,800,000\n🔗 آدرس والت RLC: <code>{rlc_wallet}</code>\n"
        ),
        "en": (
            "🔸 One-month subscription price:\n\n"
            f"💳 Bank (IRR): 500,000 IRR\n🏦 Card Number: <code>{bank_card}</code>\n\n"
            f"💎 TON Coin (TON): 1\n👛 Wallet Address: <code>{ton_wallet}</code>\n\n"
            f"🚀 RLC Token: 1,800,000\n🔗 RLC Wallet Address: <code>{rlc_wallet}</code>\n"
        ),
        "ku": (
            "🔸 نرخی اشتراکی مانگانە:\n\n"
            f"💳 کارتی بانکی: 500,000 تومان\n🏦 ژمارەی کارت: <code>{bank_card}</code>\n\n"
            f"💎 تۆن کوین (TON): 1\n👛 ناونیشانی جزدان: <code>{ton_wallet}</code>\n\n"
            f"🚀 تۆکینی RLC: 1,800,000\n🔗 ناونیشانی RLC: <code>{rlc_wallet}</code>\n"
        ),
    }
    await update.message.reply_text(
        price_text.get(lang, price_text["fa"]),
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
    """Handle /status command: inform the user of their subscription status."""
    uid = update.effective_user.id
    lang = get_lang(context)
    if has_active_subscription(uid):
        # Fetch expiration date from DB
        row = _fetchone("SELECT expire_at FROM users WHERE user_id=" + _PLACEHOLDER, (uid,))
        expire_at = row[0] if row else None
        if isinstance(expire_at, str):  # if stored as text in SQLite
            expire_at = datetime.fromisoformat(expire_at)
        if expire_at:
            exp_date = expire_at.strftime("%Y-%m-%d")
            await update.message.reply_text(
                tr("status_active", lang).format(date=exp_date),
                parse_mode=ParseMode.HTML
            )
    else:
        await update.message.reply_text(tr("no_sub", lang))

async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /ask command: forward the question to OpenAI if user has an active subscription."""
    uid = update.effective_user.id
    lang = get_lang(context)
    if not has_active_subscription(uid):
        await update.message.reply_text(tr("no_sub", lang))
        return
    # Combine the command arguments into the question text
    question = " ".join(context.args).strip()
    if not question:
        # Prompt user to provide a question text after /ask
        await update.message.reply_text({
            "fa": "❓ لطفاً سؤال را بعد از دستور بنویسید.",
            "en": "❓ Please write your legal question after the command.",
            "ku": "❓ تکایە پرسیارت لە دوای فەرمانەکە بنوسە."
        }.get(lang, "❓ لطفاً سؤال را بعد از دستور بنویسید."))
        return
    # Send typing action and get answer from OpenAI
    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        answer_text = await client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": "You are an experienced lawyer. Answer clearly." if lang != "fa" else "You are an experienced Iranian lawyer. Answer in formal Persian."},
                {"role": "user", "content": question}
            ],
            temperature=0.6,
            max_tokens=1024
        )
        answer = answer_text.choices[0].message.content.strip()
    except (APIError, RateLimitError, AuthenticationError) as e:
        logger.error("OpenAI API error: %s", e)
        answer = tr("openai_error", lang) if "openai_error" in TEXTS else "❗️Service is unavailable. Please try again later."
    # Split answer into smaller parts if too long (to respect Telegram message limit)
    parts = [answer[i:i+4000] for i in range(0, len(answer), 4000)]
    for part in parts:
        await update.message.reply_text(part)
    # Acknowledge to user that the answer was sent (especially if voice query, see voice handler)
    await update.message.reply_text({
        "fa": "✅ پاسخ ارسال شد. در صورت نیاز می‌توانید سؤال دیگری بپرسید.",
        "en": "✅ Answer sent. You may ask another question if needed.",
        "ku": "✅ وەڵام نێردرا. دەتوانیت پرسیاری تر بکەیت."
    }[lang])

async def about_token(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /about_token command: provide information about the token system."""
    lang = get_lang(context)
    # Assuming TEXTS contains an entry for "about_token" in different languages
    await update.message.reply_text(tr("about_token_info", lang), parse_mode=ParseMode.HTML)

async def lang_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /lang command: show language selection keyboard."""
    await update.message.reply_text(
        "لطفاً زبان مورد نظر را انتخاب کنید:\nPlease select your preferred language:\nتکایە زمانت هەلبژێرە:",
        reply_markup=ReplyKeyboardMarkup([[KeyboardButton("فارسی"), KeyboardButton("English"), KeyboardButton("کوردی")]], one_time_keyboard=True, resize_keyboard=True)
    )

async def cases_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /cases command: display a list of famous cases with inline buttons."""
    cases = get_famous_cases()
    if not cases:
        await update.message.reply_text("❌ پرونده‌ای یافت نشد." if get_lang(context) == "fa" else "❌ No cases found.")
        return
    # Create inline button list
    keyboard = [[InlineKeyboardButton(title, callback_data=f"case:{cid}")] for cid, title in cases]
    await update.message.reply_text(
        "📚 فهرست پرونده‌های مشهور:\nبرای مشاهده خلاصه، روی یکی از موارد زیر کلیک کنید:" if get_lang(context) == "fa" else 
        "📚 Famous Cases:\nClick a case below to see its summary:",
        reply_markup=InlineKeyboardMarkup(keyboard)
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
    """Catch-all handler for general text messages (excluding commands and specific cases)."""
    text = (update.message.text or "").strip()
    lang = get_lang(context)
    # Route by content if it matches menu options
    if lang == "fa":
        if text == "🛒 خرید اشتراک":
            await buy_cmd(update, context)
        elif text == "📤 ارسال رسید":
            await send_receipt_cmd(update, context)
        elif text == "⚖️ سؤال حقوقی":
            await update.message.reply_text(
                "سؤال خود را بعد از /ask بفرستید.\nمثال:\n<code>/ask قانون کار چیست؟</code>",
                parse_mode=ParseMode.HTML
            )

        elif text == "🎤 سؤال صوتی":
            await update.message.reply_text("🎙️ لطفاً سؤال خود را به صورت پیام صوتی ارسال کنید.\n\n📌 فقط پیام صوتی تلگرام پشتیبانی می‌شود.")

        elif text == "ℹ️ درباره توکن":
            await about_token(update, context)

    elif lang == "en":
        if text == "🛒 Buy Subscription":
            await buy_cmd(update, context)
        elif text == "📤 Send Receipt":
            await send_receipt_cmd(update, context)
        elif text == "⚖️ Legal Question":
            await update.message.reply_text(
                "Send your question after /ask.\nExample:\n<code>/ask What is labor law?</code>",
                parse_mode=ParseMode.HTML
            )

        elif text == "🎤 Voice Question":
            await update.message.reply_text("🎙️ Please send your legal question as a voice message.\n\n📌 Only Telegram voice messages are supported.")

        elif text == "ℹ️ About Token":
            await about_token(update, context)

    elif lang == "ku":
        if text == "🛒 کڕینی بەشداریکردن":
            await buy_cmd(update, context)
        elif text == "📤 ناردنی پسوڵە":
            await send_receipt_cmd(update, context)
        elif text == "⚖️ پرسیاری یاسایی":
            await update.message.reply_text(
                "پرسیارت لە دوای /ask بنووسە.\nنمونة:\n<code>/ask یاسای کار چیە؟</code>",
                parse_mode=ParseMode.HTML
            )
        elif text == "🎤 پرسیاری دەنگی":
            await update.message.reply_text("🎙️ تکایە پرسیاری یاساییەکەت وەکوو نامەی دەنگی بنێرە.\n\n📌 تەنها نامەی دەنگیی تەلەگرام پشتیوانی دەکرێت.")
        elif text == "ℹ️ دەربارەی تۆکێن":
            await about_token(update, context)

        elif text == "📚 پرونده‌های مشهور":
            await cases_cmd(update, context)
        elif text == "📚 Famous Cases":
            await cases_cmd(update, context)

        elif text == "📚 پرۆسەی ناودار":
           await cases_cmd(update, context)

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

    # Callback query handlers for inline buttons

    app.add_handler(CallbackQueryHandler(case_callback_handler, pattern=r"^case:\d+$"))
    app.add_handler(CallbackQueryHandler(callback_handler, pattern=r"^(approve|reject):\d+$"))

    # Non-command message handlers (ordered by group to control priority)

    app.add_handler(MessageHandler(filters.Regex("^(فارسی|English|کوردی)$"), lang_text_router), group=0)
    app.add_handler(MessageHandler(filters.PHOTO | (filters.TEXT & ~filters.COMMAND), handle_receipt), group=1)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router), group=2)
    app.add_handler(MessageHandler(filters.VOICE, handle_voice_message), group=3)

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
