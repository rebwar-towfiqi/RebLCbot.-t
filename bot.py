#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RebLawBot – Telegram bot that sells subscriptions and answers legal questions using OpenAI.
Updated: 2025-05-12 (Fixed version)
"""

from __future__ import annotations
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, time as dtime
from enum import Enum
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import openai
from dotenv import load_dotenv
from psycopg2 import OperationalError
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
    JobQueue,
    MessageHandler,
    filters,
)

load_dotenv()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger("RebLawBot")


def getenv_or_die(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"Environment variable '{key}' is missing")
    return val


CFG = {
    "BOT_TOKEN": getenv_or_die("BOT_TOKEN"),
    "ADMIN_ID": getenv_or_die("ADMIN_ID"),
    "OPENAI_API_KEY": getenv_or_die("OPENAI_API_KEY"),
    "DATABASE_URL": getenv_or_die("DATABASE_URL"),
    "BANK_CARD_NUMBER": getenv_or_die("BANK_CARD_NUMBER"),
    "USE_WEBHOOK": os.getenv("USE_WEBHOOK", "false").lower() == "true",
    "WEBHOOK_URL": os.getenv("WEBHOOK_URL", ""),
}

ADMIN_ID_INT = int(CFG["ADMIN_ID"])
client = openai.OpenAI(api_key=CFG["OPENAI_API_KEY"])

SQLITE_PATH = Path("users.db")
POOL: Optional[SimpleConnectionPool] = None
DB_TYPE = ""
_SQLITE_CONN: Optional[sqlite3.Connection] = None
_SQLITE_LOCK = threading.RLock()


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
               id        INTEGER PRIMARY KEY AUTOINCREMENT,
               user_id   BIGINT,
               question  TEXT,
               answer    TEXT,
               timestamp TIMESTAMP
           );"""
    )
    conn.commit()


def init_db():
    global DB_TYPE, POOL, _SQLITE_CONN
    dsn = CFG["DATABASE_URL"].strip()
    try:
        POOL = SimpleConnectionPool(1, 10, dsn=dsn, sslmode="require", connect_timeout=10)
        with POOL.getconn() as conn:
            _ensure_schema(conn)
        DB_TYPE = "postgres"
        logger.info("✅ Successfully connected to PostgreSQL.")
    except OperationalError as exc:
        logger.warning("PostgreSQL unavailable (%s), switching to SQLite.", exc)
        SQLITE_PATH.touch(exist_ok=True)
        DB_TYPE = "sqlite"
        _SQLITE_CONN = sqlite3.connect(
            SQLITE_PATH,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            check_same_thread=False,
        )
        _SQLITE_CONN.execute("PRAGMA foreign_keys=ON")
        _ensure_schema(_SQLITE_CONN)
        logger.info("✅ Using SQLite database at %s", SQLITE_PATH)


@contextmanager
def get_conn() -> Generator:
    if DB_TYPE == "postgres":
        conn = POOL.getconn()
        try:
            yield conn
        finally:
            conn.commit()
            POOL.putconn(conn)
    else:
        with _SQLITE_LOCK:
            yield _SQLITE_CONN
            _SQLITE_CONN.commit()


# ---------------------------------------------------------------------------#
# 3. Laws database (read-only SQLite)
# ---------------------------------------------------------------------------#

LAWS_DB = sqlite3.connect("iran_laws.db", check_same_thread=False)


def lookup(code: str, art_id: int) -> Optional[str]:
    cur = LAWS_DB.execute(
        "SELECT text FROM articles WHERE code=? AND id=?", (code.lower(), art_id)
    )
    row = cur.fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------#
# 4. i18n helpers
# ---------------------------------------------------------------------------#

TEXTS = {
    "fa": {
        "welcome": "👋 به RebLawBot خوش آمدید!",
        "help": "از منوی اصلی یا دستورات زیر استفاده کنید:",
        "buy": "<b>📌 روش خرید اشتراک:</b>\n• کارت‌به‌کارت ۳۰۰٬۰۰۰ تومان → <code>{card}</code>\nپس از پرداخت، روی «ارسال رسید» بزنید یا دستور /send_receipt را ارسال کنید.",
        "enter_receipt": "🖼️ لطفاً تصویر یا متن رسید را همین‌جا ارسال کنید.\nپس از تأیید مدیر، اشتراک شما فعال خواهد شد.",
        "receipt_ok": "✅ رسید دریافت شد؛ پس از بررسی تأیید می‌شود.",
        "no_subscription": "❌ ابتدا اشتراک تهیه کنید.",
        "ask_prompt": "✍🏻 سؤال خود را ارسال کنید…",
        "subscription_ok": "✅ اشتراک شما تا تاریخ {date} معتبر است.",
        "subscription_none": "❌ اشتراک فعالی ثبت نشده است.",
        "resources_empty": "هیچ سند حقوقی بارگذاری نشده است.",
        "resources_list": "📚 فهرست منابع حقوقی موجود:\n{list}",
        "invalid_article": "ماده پیدا نشد.",
        "invalid_law_usage": "📌 دستور را این‌گونه بنویسید:\n/law <کدقانون> <شماره‌ماده>\nمثال: /law civil 300",
    },
    "en": {
        "welcome": "👋 Welcome to RebLawBot!",
        "help": "Use the inline buttons or commands.",
        "buy": "<b>📌 How to buy subscription:</b>\n• Bank transfer – 300,000 Toman → <code>{card}</code>\nAfter payment click “Send Receipt” or /send_receipt.",
        "enter_receipt": "🖼️ Please send the receipt image or text here.\nSubscription activates after admin approval.",
        "receipt_ok": "✅ Receipt received. Awaiting approval.",
        "no_subscription": "❌ Please purchase a subscription first.",
        "ask_prompt": "✍🏻 Please type your legal question…",
        "subscription_ok": "✅ Your subscription is valid until {date}.",
        "subscription_none": "❌ No active subscription found.",
        "resources_empty": "No legal documents uploaded yet.",
        "resources_list": "📚 Available legal documents:\n{list}",
        "invalid_article": "Article not found.",
        "invalid_law_usage": "Usage: /law <code> <article_id> e.g. /law civil 300",
    },
}


def get_lang(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    return context.user_data.get("lang") or (
        "en" if (update.effective_user.language_code or "").startswith("en") else "fa"
    )


def tr(key: str, lang: str, **fmt) -> str:
    return TEXTS.get(lang, TEXTS["fa"]).get(key, key).format(**fmt)


# ---------------------------------------------------------------------------#
# 5. Data helpers
# ---------------------------------------------------------------------------#

def _dt(val):
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(val)
    except Exception:
        try:
            return datetime.strptime(val, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return datetime.utcnow()


def save_subscription(uid: int, username: Optional[str], days: int = 60) -> None:
    exp = datetime.utcnow() + timedelta(days=days)
    with get_conn() as conn:
        cur = conn.cursor()
        if DB_TYPE == "postgres":
            cur.execute(
                """INSERT INTO subscriptions (user_id, username, expires_at)
                       VALUES (%s,%s,%s)
                       ON CONFLICT (user_id) DO UPDATE
                       SET username=COALESCE(EXCLUDED.username,subscriptions.username),
                           expires_at=EXCLUDED.expires_at""",
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
# 6. Utility helpers
# ---------------------------------------------------------------------------#

def get_reply(update: Update) -> Tuple[Message, bool]:
    return (
        (update.callback_query.message, True)
        if update.callback_query
        else (update.message, False)
    )


def reply_target(update: Update) -> Tuple[Message, bool]:
    return get_reply(update)


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
# 7. Menu & static texts
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
# 8. Handlers – commands & callbacks
# ---------------------------------------------------------------------------#

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg, is_cb = get_reply(update)
    if is_cb:
        await update.callback_query.answer()
    lang = get_lang(update, context)
    context.user_data["lang"] = lang
    await msg.reply_text(tr("welcome", lang), reply_markup=main_menu())


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = get_lang(update, context)
    await update.message.reply_text(tr("help", lang))


async def buy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg, is_cb = get_reply(update)
    if is_cb:
        await update.callback_query.answer()
    lang = get_lang(update, context)
    await msg.reply_text(
        tr("buy", lang, card=CFG["BANK_CARD_NUMBER"]),
        parse_mode=ParseMode.HTML,
    )


async def send_receipt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg, is_cb = get_reply(update)
    if is_cb:
        await update.callback_query.answer()
    context.user_data["awaiting_receipt"] = True
    lang = get_lang(update, context)
    await msg.reply_text(tr("enter_receipt", lang))


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    lang = get_lang(update, context)
    exp = get_subscription_expiry(uid)
    if exp and datetime.utcnow() < exp:
        await update.message.reply_text(tr("subscription_ok", lang, date=exp.strftime("%Y-%m-%d")))
    else:
        await update.message.reply_text(tr("subscription_none", lang))


LEGAL_DOCS_PATH = Path("legal_documents")


async def resources_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    docs = sorted(d.stem for d in LEGAL_DOCS_PATH.glob("*.txt"))
    lang = get_lang(update, context)
    if not docs:
        await update.message.reply_text(tr("resources_empty", lang))
        return
    await send_long(
        update,
        tr("resources_list", lang, list="\n".join(f"• {name}" for name in docs)),
    )


async def law_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = get_lang(update, context)
    if not context.args:
        await update.message.reply_text(tr("invalid_law_usage", lang))
        return
    code_key = context.args[0].lower()
    if len(context.args) < 2 or not context.args[1].isdigit():
        await update.message.reply_text(tr("invalid_article", lang))
        return
    article_id = int(context.args[1])
    text = lookup(code_key, article_id)
    if text:
        await send_long(
            update, f"📜 ماده {article_id} ({code_key.upper()})\n{text}"
        )
    else:
        await update.message.reply_text(tr("invalid_article", lang))


TOKEN_IMG = Path(__file__).with_name("reblawcoin.png")
LINK_BUY = "https://t.me/blum/app?startapp=memepadjetton_RLC_JpMH5-ref_1wgcKkl94N "


async def about_token(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg, is_cb = reply_target(update)
    if is_cb:
        await update.callback_query.answer()
    buy_markup = InlineKeyboardMarkup(
        [[InlineKeyboardButton("🛒 خرید / Buy RLC", url=LINK_BUY)]]
    )
    if TOKEN_IMG.exists():
        await msg.reply_photo(TOKEN_IMG.open("rb"), reply_markup=buy_markup)
    await msg.reply_text(
        (
            "🎉 <b>توکن RebLawCoin (RLC)</b> – اولین ارز دیجیتال با محوریت خدمات حقوقی.\n"
            "<b>اهداف پروژه:</b>\n"
            "• سرمایه‌گذاری در نوآوری‌های حقوقی\n"
            "• نهادینه‌سازی عدالت روی بلاک‌چین\n"
            "• سودآوری پایدار برای سرمایه‌گذاران\n"
            "برای خرید مستقیم روی دکمهٔ زیر بزنید."
        ),
        parse_mode=ParseMode.HTML,
        reply_markup=buy_markup,
    )


async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    lang = get_lang(update, context)
    if not has_active_subscription(uid):
        await update.message.reply_text(tr("no_subscription", lang))
        return
    question = " ".join(context.args) if context.args else None
    if question:
        await answer_question(update, context, question)
    else:
        context.user_data["awaiting_question"] = True
        await update.message.reply_text(tr("ask_prompt", lang))


async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get("awaiting_receipt"):
        context.user_data.pop("awaiting_receipt", None)
        if update.message.photo or update.message.text:
            await update.message.forward(ADMIN_ID_INT)
            lang = get_lang(update, context)
            await update.message.reply_text(tr("receipt_ok", lang))
        else:
            await update.message.reply_text("⚠️ لطفاً یک تصویر یا متن رسید ارسال کنید.")
        return

    if context.user_data.get("awaiting_question"):
        context.user_data.pop("awaiting_question", None)
        await answer_question(update, context, update.message.text)


async def answer_question(
    update: Update, context: ContextTypes.DEFAULT_TYPE, question: str
):
    uid = update.effective_user.id
    lang = get_lang(update, context)
    if not has_active_subscription(uid):
        await update.message.reply_text(tr("no_subscription", lang))
        return

    try:
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
    except Exception as e:
        logger.error(f"OpenAI Error: {e}")
        await update.message.reply_text("❌ خطایی در پردازش سوال شما رخ داده است.")


async def daily_cleanup(context: ContextTypes.DEFAULT_TYPE):
    now = datetime.utcnow()
    with get_conn() as conn:
        cur = conn.cursor()
        sql = (
            "DELETE FROM subscriptions WHERE expires_at < %s"
            if DB_TYPE == "postgres"
            else "DELETE FROM subscriptions WHERE expires_at < ?"
        )
        cur.execute(sql, (now,))
    logger.info("🧹 Daily cleanup completed: removed expired subscriptions")


async def approve_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID_INT:
        await update.message.reply_text("🚫 دسترسی غیرمجاز.")
        return
    if len(context.args) < 1 or not context.args[0].isdigit():
        await update.message.reply_text("📌 دستور: /approve <user_id>")
        return
    uid = int(context.args[0])
    save_subscription(uid, None, days=60)
    await update.message.reply_text(f"✅ اشتراک کاربر {uid} فعال شد.")


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
        msg.reply_text("✍🏻 سؤال خود را ارسال کنید…")
    elif data == Menu.RESOURCES.value:
        await resources_cmd(update, context)
    elif data == Menu.TOKEN.value:
        await about_token(update, context)


# ---------------------------------------------------------------------------#
# 9. Dispatcher registration
# ---------------------------------------------------------------------------#

def register_handlers(app: Application):
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("buy", buy))
    app.add_handler(CommandHandler("send_receipt", send_receipt))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("resources", resources_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(CommandHandler("law", law_document))
    app.add_handler(CommandHandler("approve", approve_cmd))  # New admin command
    app.add_handler(CallbackQueryHandler(menu_router))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    job_queue = app.job_queue
    if job_queue:
        job_queue.run_daily(daily_cleanup, time=dtime(hour=3, minute=0))


# ---------------------------------------------------------------------------#
# 10. Main
# ---------------------------------------------------------------------------#

def main() -> None:
    init_db()
    application = Application.builder().token(CFG["BOT_TOKEN"]).build()
    register_handlers(application)
    logger.info("🤖 Bot started …")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
