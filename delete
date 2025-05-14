#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RebLawBot – Telegram bot that sells subscriptions and answers legal questions using OpenAI.
Version 2025-05-14 (compat OpenAI 1.x)
"""

from __future__ import annotations
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator, Optional, Tuple

# External libraries
from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError, RateLimitError, AuthenticationError
from psycopg2.pool import SimpleConnectionPool
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    Update,
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from enum import Enum

# ---------------------------------------------------------------------------#
# 1️⃣ Environment & Global Configuration                                      #
# ---------------------------------------------------------------------------#

load_dotenv()  # Load environment variables from .env
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger("RebLawBot")

# Helper to ensure environment variables are set
def getenv_or_die(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Environment variable {key!r} is missing")
    return value

# Load essential configuration
BOT_TOKEN = getenv_or_die("BOT_TOKEN")
ADMIN_ID = int(getenv_or_die("ADMIN_ID"))
OPENAI_API_KEY = getenv_or_die("OPENAI_API_KEY")
DATABASE_URL = getenv_or_die("DATABASE_URL")
TON_WALLET_ADDR = getenv_or_die("TON_WALLET_ADDRESS")
BANK_CARD_NUMBER = getenv_or_die("BANK_CARD_NUMBER")

# Initialize OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# SQLite fallback file
SQLITE_PATH = Path("users.db")
POOL: Optional[SimpleConnectionPool] = None
DB_TYPE = ""  # "postgres" or "sqlite"

# ---------------------------------------------------------------------------#
# 2️⃣ Database Layer                                                          #
# ---------------------------------------------------------------------------#

def _ensure_schema(conn):
    """
    Ensure the database schema exists for both PostgreSQL and SQLite.
    """
    cur = conn.cursor()
    # Subscriptions table
    cur.execute(
        """CREATE TABLE IF NOT EXISTS subscriptions (
               user_id    BIGINT PRIMARY KEY,
               username   TEXT,
               expires_at TIMESTAMP
           );"""
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_subscriptions_expires ON subscriptions (expires_at);"
    )
    # Questions table
    cur.execute(
        """CREATE TABLE IF NOT EXISTS questions (
               id        BIGSERIAL PRIMARY KEY,
               user_id   BIGINT,
               question  TEXT,
               answer    TEXT,
               timestamp TIMESTAMP
           );"""
    )
    conn.commit()
    logger.info("✅ Database schema ensured.")

def init_db():
    """
    Initialize the database connection (PostgreSQL → fallback SQLite).
    """
    global DB_TYPE, POOL
    try:
        # Try connecting to PostgreSQL
        POOL = SimpleConnectionPool(
            1, 10, dsn=DATABASE_URL, sslmode="require", connect_timeout=10
        )
        with POOL.getconn() as conn:
            _ensure_schema(conn)
        DB_TYPE = "postgres"
        logger.info("✅ Connected to PostgreSQL")
    except Exception as exc:
        # Fallback to SQLite
        logger.warning("PostgreSQL unavailable: %r → switching to SQLite.", exc)
        SQLITE_PATH.touch(exist_ok=True)
        DB_TYPE = "sqlite"
        with sqlite3.connect(
            SQLITE_PATH,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            check_same_thread=False,
        ) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            _ensure_schema(conn)
        logger.info("✅ Using local SQLite: %s", SQLITE_PATH)

@contextmanager
def get_conn():
    """
    Context manager for database connections.
    """
    if DB_TYPE == "postgres":
        assert POOL is not None
        conn = POOL.getconn()
        try:
            yield conn
        finally:
            conn.commit()
            POOL.putconn(conn)
    else:
        conn = sqlite3.connect(
            SQLITE_PATH,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            check_same_thread=False,
        )
        try:
            yield conn
        finally:
            conn.commit()
            conn.close()

# ---------------------------------------------------------------------------#
# 3️⃣ Data Helpers                                                            #
# ---------------------------------------------------------------------------#

def save_subscription(uid: int, username: Optional[str], days: int = 60):
    """
    Save or update a user's subscription in the database.
    """
    exp = datetime.utcnow() + timedelta(days=days)
    with get_conn() as conn:
        cur = conn.cursor()
        if DB_TYPE == "postgres":
            cur.execute(
                """
                INSERT INTO subscriptions (user_id, username, expires_at)
                  VALUES (%s, %s, %s)
                  ON CONFLICT (user_id) DO UPDATE
                  SET username = COALESCE(EXCLUDED.username, subscriptions.username),
                      expires_at = EXCLUDED.expires_at
                """,
                (uid, username, exp),
            )
        else:
            cur.execute(
                "INSERT OR REPLACE INTO subscriptions VALUES (?,?,?)",
                (uid, username, exp),
            )
    logger.info("🔒 Subscription set: %s → %s", uid, exp)

def has_active_subscription(uid: int) -> bool:
    """
    Check if a user has an active subscription.
    """
    with get_conn() as conn:
        cur = conn.cursor()
        sql = (
            "SELECT expires_at FROM subscriptions WHERE user_id=%s"
            if DB_TYPE == "postgres"
            else "SELECT expires_at FROM subscriptions WHERE user_id=?"
        )
        cur.execute(sql, (uid,))
        row = cur.fetchone()
    return bool(row and datetime.utcnow() < row[0])

def save_question(uid: int, q: str, a: str):
    """
    Save a legal question and its answer to the database.
    """
    with get_conn() as conn:
        cur = conn.cursor()
        sql = (
            "INSERT INTO questions (user_id, question, answer, timestamp) VALUES (%s,%s,%s,%s)"
            if DB_TYPE == "postgres"
            else "INSERT INTO questions VALUES (NULL,?,?,?,?)"
        )
        cur.execute(sql, (uid, q, a, datetime.utcnow()))
    logger.debug("💾 Q saved for %s", uid)

# ---------------------------------------------------------------------------#
# 4️⃣ Utility Functions                                                       #
# ---------------------------------------------------------------------------#

async def send_long(update: Update, text: str, **kwargs):
    """
    Send long messages in chunks to avoid Telegram's message length limit.
    """
    msg, _ = get_reply(update)
    for part in split_message(text):
        await msg.reply_text(part, **kwargs)

def split_message(text: str, limit: int = 4096) -> List[str]:
    """
    Split long messages into smaller chunks.
    """
    if len(text) <= limit:
        return [text]
    parts = []
    while len(text) > limit:
        cutoff = text.rfind("\n", 0, limit)
        if cutoff == -1:
            cutoff = limit
        parts.append(text[:cutoff].strip())
        text = text[cutoff:].strip()
    if text:
        parts.append(text)
    return parts

# ---------------------------------------------------------------------------#
# 5️⃣ Handlers                                                                #
# ---------------------------------------------------------------------------#

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle the /start command.
    """
    msg, _ = get_reply(update)
    await msg.reply_text("👋 به RebLawBot خوش آمدید!", reply_markup=main_menu())

async def buy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle the /buy command.
    """
    msg, _ = get_reply(update)
    await msg.reply_text(
        (
            "<b>📌 روش خرید اشتراک:</b>\n"
            f"• کارت‌به‌کارت ۳۰۰٬۰۰۰ تومان → <code>{BANK_CARD_NUMBER}</code>\n"
            "\nپس از پرداخت، روی «ارسال رسید» بزنید یا دستور /send_receipt را ارسال کنید."
        ),
        parse_mode=ParseMode.HTML,
    )

async def send_receipt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle the /send_receipt command.
    """
    await update.message.reply_text("لطفاً رسید را به صورت عکس یا متن ارسال کنید.")
    context.user_data["awaiting_receipt"] = True

async def handle_receipt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle receipt submission (photo or text).
    """
    if not context.user_data.get("awaiting_receipt"):
        return
    user = update.effective_user
    if not (update.message.photo or update.message.text):
        await update.message.reply_text("❌ فقط عکس یا متن قابل قبول است.")
        return
    caption = (
        "📥 رسید پرداخت\n"
        f"ID: <code>{user.id}</code>\n"
        f"👤 @{user.username or '—'}"
    )
    markup = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ تأیید", callback_data=f"approve:{user.id}"),
        InlineKeyboardButton("❌ رد", callback_data=f"reject:{user.id}")
    ]])
    try:
        if update.message.photo:
            await context.bot.send_photo(
                chat_id=ADMIN_ID,
                photo=update.message.photo[-1].file_id,
                caption=caption,
                reply_markup=markup,
                parse_mode=ParseMode.HTML,
            )
        else:
            await context.bot.send_message(
                chat_id=ADMIN_ID,
                text=f"{caption}\n📝 {update.message.text}",
                reply_markup=markup,
                parse_mode=ParseMode.HTML,
            )
        await update.message.reply_text("✅ رسید شما ثبت شد، منتظر بررسی مدیر باشید.")
    except Exception as e:
        logger.error("Receipt forwarding error: %s", e)
        await update.message.reply_text("❌ خطا در ارسال رسید به مدیر.")
    finally:
        context.user_data["awaiting_receipt"] = False

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle admin approval/rejection of receipts.
    """
    query = update.callback_query
    await query.answer()
    action, user_id_str = query.data.split(":")
    user_id = int(user_id_str)
    try:
        if action == "approve":
            save_subscription(user_id, days=180)
            await context.bot.send_message(user_id, "✅ اشتراک شما تایید شد.")
            await query.edit_message_caption("✅ رسید تایید شد.")
        else:
            await context.bot.send_message(user_id, "❌ متاسفیم، رسید شما رد شد.")
            await query.edit_message_caption("❌ رسید رد شد.")
    except Exception as e:
        logger.error("Callback handler error: %s", e)
        await query.edit_message_reply_markup(reply_markup=None)

# ---------------------------------------------------------------------------#
# 6️⃣ Menu & Static Texts                                                     #
# ---------------------------------------------------------------------------#

class Menu(Enum):
    BUY = "menu_buy"
    SEND_RECEIPT = "menu_send_receipt"
    STATUS = "menu_status"
    ASK = "menu_ask"
    RESOURCES = "menu_resources"
    TOKEN = "menu_token"

def main_menu() -> InlineKeyboardMarkup:
    """
    Generate the main menu keyboard.
    """
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🔐 خرید اشتراک", callback_data=Menu.BUY.value)],
            [InlineKeyboardButton("📎 ارسال رسید", callback_data=Menu.SEND_RECEIPT.value)],
            [InlineKeyboardButton("📅 وضعیت اشتراک", callback_data=Menu.STATUS.value)],
            [InlineKeyboardButton("⚖️ سؤال حقوقی", callback_data=Menu.ASK.value)],
            [InlineKeyboardButton("📘 منابع حقوقی", callback_data=Menu.RESOURCES.value)],
            [InlineKeyboardButton("💎 توکن RebLawCoin", callback_data=Menu.TOKEN.value)],
        ]
    )

# ---------------------------------------------------------------------------#
# 7️⃣ Dispatcher Registration & Main Function                                 #
# ---------------------------------------------------------------------------#

def register_handlers(app: Application):
    """
    Register all handlers for the bot.
    """
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("buy", buy))
    app.add_handler(CommandHandler("send_receipt", send_receipt))
    app.add_handler(MessageHandler(filters.PHOTO | (filters.TEXT & ~filters.COMMAND), handle_receipt))
    app.add_handler(CallbackQueryHandler(callback_handler))

def main() -> None:
    """
    Main entry point for the bot.
    """
    init_db()
    application = Application.builder().token(BOT_TOKEN).build()
    register_handlers(application)
    logger.info("🤖 RebLawBot started successfully …")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
