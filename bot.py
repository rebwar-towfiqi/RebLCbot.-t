#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RebLawBot – Telegram bot that sells subscriptions and answers legal questions using OpenAI.
Version 2025-05-12
"""

from __future__ import annotations

# ---------------------------------------------------------------------------#
# 0. Imports                                                                 #
# ---------------------------------------------------------------------------#
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import openai
from dotenv import load_dotenv
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

# ---------------------------------------------------------------------------#
# 1. Environment & global configuration                                      #
# ---------------------------------------------------------------------------#
load_dotenv()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("RebLawBot")


def getenv_or_die(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"Environment variable '{key}' is missing")
    return val


CFG = {
    "BOT_TOKEN": getenv_or_die("BOT_TOKEN"),
    "ADMIN_ID": int(getenv_or_die("ADMIN_ID")),
    "OPENAI_API_KEY": getenv_or_die("OPENAI_API_KEY"),
    "DATABASE_URL": getenv_or_die("DATABASE_URL"),
    "BANK_CARD_NUMBER": getenv_or_die("BANK_CARD_NUMBER"),
}

client = openai.OpenAI(api_key=CFG["OPENAI_API_KEY"])

# ---------------------------------------------------------------------------#
# 2. Application DB (PostgreSQL → fallback SQLite)                           #
# ---------------------------------------------------------------------------#
SQLITE_PATH = Path("users.db")
POOL: Optional[SimpleConnectionPool] = None
DB_TYPE = ""  # "postgres" or "sqlite"


def _ensure_schema(conn):
    cur = conn.cursor()
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


def init_db():
    """Initialise database and detect backend."""
    global DB_TYPE, POOL
    dsn = CFG["DATABASE_URL"].strip()
    try:
        POOL = SimpleConnectionPool(
            1, 10, dsn=dsn, sslmode="require", connect_timeout=10
        )
        with POOL.getconn() as conn:
            _ensure_schema(conn)
        DB_TYPE = "postgres"
        logger.info("✅ Connected to PostgreSQL")
    except Exception as exc:
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
def get_conn() -> Generator:
    """Context manager returning a DB connection with auto-commit."""
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
# 3. Laws database (read-only SQLite)                                        #
# ---------------------------------------------------------------------------#
LAWS_DB = sqlite3.connect("iran_laws.db", check_same_thread=False)


def lookup(code: str, art_id: int) -> Optional[str]:
    cur = LAWS_DB.execute(
        "SELECT text FROM articles WHERE code=? AND id=?", (code.lower(), art_id)
    )
    row = cur.fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------#
# 4. Data helpers                                                            #
# ---------------------------------------------------------------------------#
def _dt(val):
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(val)
    except Exception:
        return datetime.strptime(val, "%Y-%m-%d %H:%M:%S")


def save_subscription(uid: int, username: Optional[str], days: int = 60) -> None:
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
    with get_conn() as conn:
        cur = conn.cursor()
        sql = (
            "SELECT expires_at FROM subscriptions WHERE user_id=%s"
            if DB_TYPE == "postgres"
            else "SELECT expires_at FROM subscriptions WHERE user_id=?"
        )
        cur.execute(sql, (uid,))
        row = cur.fetchone()
    return bool(row and datetime.utcnow() < _dt(row[0]))


def get_subscription_expiry(uid: int) -> Optional[datetime]:
    with get_conn() as conn:
        cur = conn.cursor()
        sql = (
            "SELECT expires_at FROM subscriptions WHERE user_id=%s"
            if DB_TYPE == "postgres"
            else "SELECT expires_at FROM subscriptions WHERE user_id=?"
        )
        cur.execute(sql, (uid,))
        row = cur.fetchone()
    return _dt(row[0]) if row else None


def save_question(uid: int, q: str, a: str) -> None:
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
# 5. Utility helpers                                                         #
# ---------------------------------------------------------------------------#
def get_reply(update: Update) -> Tuple[Message, bool]:
    return (
        (update.callback_query.message, True)
        if update.callback_query
        else (update.message, False)
    )


def chunks(text: str, limit: int = 4096) -> List[str]:
    import textwrap

    if len(text) <= limit:
        return [text]
    return textwrap.wrap(text, limit - 20, break_long_words=False)


async def send_long(update: Update, text: str, **kw):
    msg, _ = get_reply(update)
    for part in chunks(text):
        await msg.reply_text(part, **kw)


# ---------------------------------------------------------------------------#
# 6. Menu & static texts                                                     #
# ---------------------------------------------------------------------------#
class Menu(Enum):
    BUY = "menu_buy"
    SEND_RECEIPT = "menu_send_receipt"
    STATUS = "menu_status"
    ASK = "menu_ask"
    RESOURCES = "menu_resources"
    TOKEN = "menu_token"


def main_menu() -> InlineKeyboardMarkup:
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
# 7. Handlers – commands & callbacks                                         #
# ---------------------------------------------------------------------------#
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg, is_cb = get_reply(update)
    if is_cb:
        await update.callback_query.answer()
    await msg.reply_text("👋 به RebLawBot خوش آمدید!", reply_markup=main_menu())


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("از دستور /start یا دکمه‌های منو استفاده کنید.")


async def buy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg, is_cb = get_reply(update)
    if is_cb:
        await update.callback_query.answer()
    await msg.reply_text(
        (
            "<b>📌 روش خرید اشتراک:</b>\n"
            f"• کارت‌به‌کارت ۳۰۰٬۰۰۰ تومان → <code>{CFG['BANK_CARD_NUMBER']}</code>\n"
            "\nپس از پرداخت، روی «ارسال رسید» بزنید یا دستور /send_receipt را ارسال کنید."
        ),
        parse_mode=ParseMode.HTML,
    )


async def send_receipt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """یادآوری به کاربر برای ارسال تصویر/متن رسید."""
    msg, is_cb = get_reply(update)
    if is_cb:
        await update.callback_query.answer()
    context.user_data["awaiting_receipt"] = True
    await msg.reply_text(
        "🖼️ لطفاً تصویر یا متن رسید را همین‌جا ارسال کنید.\n"
        "پس از تأیید مدیر، اشتراک شما فعال خواهد شد."
    )


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    exp = get_subscription_expiry(uid)
    if exp and datetime.utcnow() < exp:
        await update.message.reply_text(
            f"✅ اشتراک شما تا تاریخ {exp:%Y-%m-%d} معتبر است."
        )
    else:
        await update.message.reply_text("❌ اشتراک فعالی ثبت نشده است.")


LEGAL_DOCS_PATH = Path("legal_documents")


async def resources_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    docs = sorted(d.stem for d in LEGAL_DOCS_PATH.glob("*.txt"))
    if not docs:
        await update.message.reply_text("هیچ سند حقوقی بارگذاری نشده است.")
        return
    await send_long(
        update,
        "📚 فهرست منابع حقوقی موجود:\n" + "\n".join(f"• {name}" for name in docs),
    )


async def law_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "📌 دستور را این‌گونه بنویسید:\n"
            "/law <کدقانون> <شماره‌ماده>\n"
            "مثال: /law civil 300"
        )
        return
    code_key = context.args[0].lower()
    if len(context.args) < 2 or not context.args[1].isdigit():
        await update.message.reply_text("شماره ماده نامعتبر است.")
        return
    article_id = int(context.args[1])
    text = lookup(code_key, article_id)
    if text:
        await send_long(
            update, f"📜 ماده {article_id} ({code_key.upper()})\n\n{text}"
        )
    else:
        await update.message.reply_text("ماده پیدا نشد.")


TOKEN_IMG = Path(__file__).with_name("reblawcoin.png")


async def about_token(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg, is_cb = get_reply(update)
    if is_cb:
        await update.callback_query.answer()
    if TOKEN_IMG.exists():
        await msg.reply_photo(TOKEN_IMG.open("rb"))
    await msg.reply_text(
        (
            "🎉 <b>توکن RebLawCoin (RLC)</b> – اولین ارز دیجیتال با محوریت خدمات حقوقی.\n\n"
            "<b>اهداف پروژه:</b>\n"
            "• سرمایه‌گذاری در نوآوری‌های حقوقی\n"
            "• نهادینه‌سازی عدالت روی بلاک‌چین\n"
            "• سودآوری پایدار برای سرمایه‌گذاران"
        ),
        parse_mode=ParseMode.HTML,
    )


async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """‎/ask [question]"""
    uid = update.effective_user.id
    if not has_active_subscription(uid):
        await update.message.reply_text("❌ ابتدا اشتراک تهیه کنید.")
        return

    question = " ".join(context.args) if context.args else None
    if question:
        await answer_question(update, context, question)
    else:
        context.user_data["awaiting_question"] = True
        await update.message.reply_text("✍🏻 سؤال خود را ارسال کنید…")


async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Catch-all text handler:
       • رسید پرداخت
       • سؤال حقوقی (در حالت انتظار)
    """
    if context.user_data.get("awaiting_receipt"):
        context.user_data.pop("awaiting_receipt", None)
        # اینجا می‌توانید رسید را به مدیر فوروارد و در DB ذخیره کنید
        await update.message.forward(CFG["ADMIN_ID"])
        await update.message.reply_text("✅ رسید دریافت شد؛ پس از بررسی تأیید می‌شود.")
        return

    if context.user_data.get("awaiting_question"):
        context.user_data.pop("awaiting_question", None)
        await answer_question(update, context, update.message.text)


async def answer_question(
    update: Update, context: ContextTypes.DEFAULT_TYPE, question: str
):
    """Send question to ChatGPT, save answer, and reply."""
    uid = update.effective_user.id
    await update.message.chat.send_action(ChatAction.TYPING)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        temperature=0.2,
        max_tokens=1024,
    )
    answer = resp.choices[0].message.content.strip()
    save_question(uid, question, answer)
    await send_long(update, answer)


# ----------------- CallbackQuery router ----------------- #
async def menu_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = update.callback_query.data
    await update.callback_query.answer()

    if data == Menu.BUY.value:
        await buy(update, context)
    elif data == Menu.SEND_RECEIPT.value:
        await send_receipt(update, context)
    elif data == Menu.STATUS.value:
        await status_cmd(update, context)
    elif data == Menu.ASK.value:
        context.user_data["awaiting_question"] = True
        msg, _ = get_reply(update)
        await msg.reply_text("✍🏻 سؤال خود را ارسال کنید…")
    elif data == Menu.RESOURCES.value:
        await resources_cmd(update, context)
    elif data == Menu.TOKEN.value:
        await about_token(update, context)


# ---------------------------------------------------------------------------#
# 8. Dispatcher registration                                                 #
# ---------------------------------------------------------------------------#
def register_handlers(app: Application):
    # commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("buy", buy))
    app.add_handler(CommandHandler("send_receipt", send_receipt))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("resources", resources_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(CommandHandler("law", law_document))

    # callback queries
    app.add_handler(CallbackQueryHandler(menu_router))

    # text (generic)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))


# ---------------------------------------------------------------------------#
# 9. Main                                                                    #
# ---------------------------------------------------------------------------#
def main() -> None:
    init_db()
    application = Application.builder().token(CFG["BOT_TOKEN"]).build()
    register_handlers(application)

    logger.info("🤖 Bot started …")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

