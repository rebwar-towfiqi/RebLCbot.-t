"""RebLCbot – Telegram bot that sells subscriptions and answers legal questions using OpenAI.
Copyright © 2025 Rebwar Lawyer
"""
from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator, Optional

import openai
from dotenv import load_dotenv
from psycopg2.pool import SimpleConnectionPool
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ──────────────────────────────────────────
# 1. Environment variables & configuration
# ──────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("RebLawCoin_bot")


def getenv_or_die(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Environment variable '{key}' is missing")
    return value


config = {
    "BOT_TOKEN": getenv_or_die("BOT_TOKEN"),
    "ADMIN_ID": int(getenv_or_die("ADMIN_ID")),
    "OPENAI_API_KEY": getenv_or_die("OPENAI_API_KEY"),
    "DATABASE_URL": getenv_or_die("DATABASE_URL"),
    "BANK_CARD_NUMBER": getenv_or_die("BANK_CARD_NUMBER"),
}

client = openai.OpenAI(api_key=config["OPENAI_API_KEY"])

# ──────────────────────────────────────────
# 2. Database initialisation
# ──────────────────────────────────────────
SQLITE_PATH = Path("users.db")
DB_TYPE = ""
POOL: Optional[SimpleConnectionPool] = None


@contextmanager
def get_conn() -> Generator:
    global DB_TYPE, POOL
    if DB_TYPE == "postgres":
        conn = POOL.getconn()
        try:
            yield conn
        finally:
            POOL.putconn(conn)
    else:
        conn = sqlite3.connect(SQLITE_PATH)
        try:
            yield conn
        finally:
            conn.close()


def init_db() -> None:
    """Try Postgres, fall back to SQLite."""
    global DB_TYPE, POOL
    db_url = config["DATABASE_URL"]
    if db_url:
        try:
            POOL = SimpleConnectionPool(
                1, 10, dsn=db_url, sslmode="require", connect_timeout=10
            )
            POOL.putconn(POOL.getconn())
            DB_TYPE = "postgres"
            logger.info("Connected to PostgreSQL 🎉")
            return
        except Exception as e:  # noqa: BLE001
            logger.warning("Postgres init failed (%r), using SQLite.", e)

    DB_TYPE = "sqlite"
    SQLITE_PATH.touch(exist_ok=True)
    with sqlite3.connect(SQLITE_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS subscriptions (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                expires_at TIMESTAMP
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                question TEXT,
                answer TEXT,
                timestamp TIMESTAMP
            );
            """
        )
    logger.info("Using SQLite database (%s) 🎉", SQLITE_PATH)


# ──────────────────────────────────────────
# 3. DB helpers
# ──────────────────────────────────────────
def save_subscription(user_id: int, username: str | None, days: int) -> None:
    expires_at = datetime.utcnow() + timedelta(days=days)
    with get_conn() as conn:
        cur = conn.cursor()
        if DB_TYPE == "postgres":
            cur.execute(
                """
                INSERT INTO subscriptions (user_id, username, expires_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    username = COALESCE(EXCLUDED.username, subscriptions.username),
                    expires_at = EXCLUDED.expires_at
                """,
                (user_id, username, expires_at),
            )
        else:
            cur.execute(
                """
                INSERT OR REPLACE INTO subscriptions (user_id, username, expires_at)
                VALUES (?, ?, ?)
                """,
                (user_id, username, expires_at),
            )
    logger.info("Subscription saved for %s until %s", user_id, expires_at)


def has_active_subscription(user_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.cursor()
        query = (
            "SELECT expires_at FROM subscriptions WHERE user_id = %s"
            if DB_TYPE == "postgres"
            else "SELECT expires_at FROM subscriptions WHERE user_id = ?"
        )
        cur.execute(query, (user_id,))
        row = cur.fetchone()
    if not row:
        return False
    expires = row[0]
    if isinstance(expires, str):
        expires = datetime.fromisoformat(expires)
    return datetime.utcnow() < expires


# ──────────────────────────────────────────
# 4. Bot UI & handlers
# ──────────────────────────────────────────
def english_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🔐 Buy Subscription", callback_data="menu_buy")],
            [InlineKeyboardButton("📎 Send Receipt", callback_data="menu_send_receipt")],
            [InlineKeyboardButton("📅 Status", callback_data="menu_status")],
            [InlineKeyboardButton("⚖️ Ask a Question", callback_data="menu_ask")],
            [
                InlineKeyboardButton(
                    "💎 RebLawCoin Token", callback_data="menu_token_info"
                )
            ],
        ]
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 به RebLawBot خوش آمدید!\n"
        "از منوی زیر گزینهٔ موردنظر را انتخاب کنید.\n\n"
        "📖 <b>دستورات:</b>\n"
        "/buy — خرید اشتراک\n"
        "/send_receipt — ارسال رسید پرداخت\n"
        "/status — وضعیت اشتراک\n"
        "/ask &lt;سؤال&gt; — پرسش حقوقی\n"
        "/token — معرفی و اهداف توکن RebLawCoin",
        parse_mode=ParseMode.HTML,
        reply_markup=english_menu(),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "ℹ️ برای مشاهدهٔ منو و دستورات، /start را ارسال کنید."
    )


async def buy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "<b>📌 روش خرید اشتراک:</b>\n"
        f"• کارت‌به‌کارت ۵۰۰٬۰۰۰ تومان → <code>{config['BANK_CARD_NUMBER']}</code>\n\n"
        "🔸 سپس دستور /send_receipt را ارسال کنید.\n"
        "✅ پرداخت کارت‌بانکی = اشتراک ۱ ماهه",
        parse_mode=ParseMode.HTML,
    )


async def send_receipt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("لطفاً رسید را به صورت عکس یا متن ارسال کنید.")
    context.user_data["awaiting_receipt"] = True
    context.user_data["payment_type"] = "bank"


async def handle_receipt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.user_data.get("awaiting_receipt"):
        return

    user = update.effective_user
    caption = (
        "📥 رسید پرداخت از:\n"
        f"🆔 <code>{user.id}</code>\n"
        f"👤 @{user.username or '—'}"
    )
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✅ تأیید", callback_data=f"approve:{user.id}"),
                InlineKeyboardButton("❌ رد", callback_data=f"reject:{user.id}"),
            ]
        ]
    )

    admin_id = config["ADMIN_ID"]
    if update.message.photo:
        await context.bot.send_photo(
            admin_id,
            update.message.photo[-1].file_id,
            caption=caption,
            reply_markup=keyboard,
            parse_mode=ParseMode.HTML,
        )
    else:
        await context.bot.send_message(
            admin_id,
            f"{caption}\n📝 {update.message.text}",
            reply_markup=keyboard,
            parse_mode=ParseMode.HTML,
        )

    await update.message.reply_text("✅ رسید شما ثبت شد، منتظر تأیید مدیر باشید.")
    context.user_data["awaiting_receipt"] = False


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    action, user_id_str = query.data.split(":", 1)
    user_id = int(user_id_str)

    if action == "approve":
        save_subscription(user_id, "-", 30)  # یک ماه
        await context.bot.send_message(
            user_id, "✅ اشتراک شما تأیید شد. از خدمات لذت ببرید."
        )
        await query.edit_message_caption("✅ رسید تأیید شد.")
    else:
        await context.bot.send_message(user_id, "❌ متأسفیم، رسید شما تأیید نشد.")
        await query.edit_message_caption("❌ رسید رد شد.")


def save_question(user_id: int, question: str, answer: str) -> None:
    ts = datetime.utcnow()
    with get_conn() as conn:
        cur = conn.cursor()
        if DB_TYPE == "postgres":
            cur.execute(
                "INSERT INTO questions (user_id, question, answer, timestamp) "
                "VALUES (%s, %s, %s, %s)",
                (user_id, question, answer, ts),
            )
        else:
            cur.execute(
                "INSERT INTO questions (user_id, question, answer, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (user_id, question, answer, ts),
            )
    logger.info("Q/A stored for %s", user_id)


async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not has_active_subscription(user_id):
        await update.message.reply_text("❌ شما اشتراک فعالی ندارید.")
        return

    if not context.args:
        await update.message.reply_text("❓ لطفاً سؤال را پس از /ask بنویسید.")
        return

    question = " ".join(context.args)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a legal expert assistant. Answer in Persian.",
                },
                {"role": "user", "content": question},
            ],
            max_tokens=500,
            temperature=0.7,
        )
        answer = response.choices[0].message.content.strip()
        await update.message.reply_text(answer)
        save_question(user_id, question, answer)
    except Exception as e:  # noqa: BLE001
        logger.error("OpenAI error: %s", e)
        await update.message.reply_text("⚠️ خطا در پردازش سؤال. بعداً دوباره تلاش کنید.")


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    with get_conn() as conn:
        cur = conn.cursor()
        query = (
            "SELECT expires_at FROM subscriptions WHERE user_id = %s"
            if DB_TYPE == "postgres"
            else "SELECT expires_at FROM subscriptions WHERE user_id = ?"
        )
        cur.execute(query, (user_id,))
        row = cur.fetchone()
    if not row:
        await update.message.reply_text("❌ اشتراک فعالی یافت نشد.")
        return

    expires = row[0]
    if isinstance(expires, str):
        expires = datetime.fromisoformat(expires)
    remaining = (expires - datetime.utcnow()).days
    await update.message.reply_text(
        f"✅ اشتراک شما تا {expires:%Y-%m-%d} معتبر است ({remaining} روز)."
    )


TOKEN_IMG = Path(__file__).with_name("reblawcoin.png")


async def about_token(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if TOKEN_IMG.exists():
        await update.message.reply_photo(TOKEN_IMG.open("rb"))
    await update.message.reply_text(
        "🎉 <b>توکن RebLawCoin (RLC)</b> اولین ارز دیجیتال حقوقی است.\n\n"
        "<b>اهداف پروژه:</b>\n"
        "• سرمایه‌گذاری در طرح‌های حقوقی\n"
        "• نهادینه‌سازی عدالت با بلاک‌چین\n"
        "• سودآوری پایدار برای سرمایه‌گذاران",
        parse_mode=ParseMode.HTML,
    )


# ──────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────
async def remove_webhook(app: Application) -> None:
    await app.bot.delete_webhook(drop_pending_updates=True)
    logger.info("Webhook removed.")


def main() -> None:
    init_db()

    app = (
        Application.builder()
        .token(config["BOT_TOKEN"])
        .post_init(remove_webhook)
        .build()
    )

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("buy", buy))
    app.add_handler(CommandHandler("send_receipt", send_receipt))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("ask", ask))
    app.add_handler(CommandHandler("token", about_token))

    # Menu callbacks
    app.add_handler(CallbackQueryHandler(buy, pattern="^menu_buy$"))
    app.add_handler(CallbackQueryHandler(send_receipt, pattern="^menu_send_receipt$"))
    app.add_handler(CallbackQueryHandler(status, pattern="^menu_status$"))
    app.add_handler(CallbackQueryHandler(ask, pattern="^menu_ask$"))
    app.add_handler(CallbackQueryHandler(about_token, pattern="^menu_token_info$"))

    # Receipt handler
    app.add_handler(
        MessageHandler((filters.PHOTO | filters.TEXT) & ~filters.COMMAND, handle_receipt)
    )

    # Admin approve / reject
    app.add_handler(CallbackQueryHandler(callback_handler, pattern="^(approve|reject):"))

    logger.info("🤖 RebLawBot started successfully.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
