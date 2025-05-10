"""RebLCbot - Telegram bot that sells subscriptions and answers legal questions using OpenAI.
Copyright © 2025 Rebwar Lawyer
"""
from __future__ import annotations
import os
import logging
import openai
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
from typing import Generator, Optional
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from psycopg2.pool import SimpleConnectionPool

# ---------------------------------------------------------------------------#
# 1. Environment variables & Configuration                                   #
# ---------------------------------------------------------------------------#

load_dotenv()
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
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
    "TON_WALLET_ADDR": getenv_or_die("TON_WALLET_ADDRESS"),
    "BANK_CARD_NUMBER": getenv_or_die("BANK_CARD_NUMBER"),
}

# Initialize OpenAI client
client = openai.OpenAI(api_key=config["OPENAI_API_KEY"])

# ---------------------------------------------------------------------------#
# 2. Database initialization & connection                                    #
# ---------------------------------------------------------------------------#

SQLITE_PATH = Path("users.db")
DB_TYPE: str = ""
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
    global DB_TYPE, POOL
    database_url = config["DATABASE_URL"]
    if database_url:
        try:
            POOL = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=database_url,
                sslmode="require",
                connect_timeout=10,
            )
            conn = POOL.getconn()
            POOL.putconn(conn)
            DB_TYPE = "postgres"
            logger.info("Connected to PostgreSQL 🎉")
            return
        except Exception as e:
            logger.error(f"Postgres init failed ({e!r})")
    else:
        logger.error("DATABASE_URL not set")

def initialize_database() -> None:
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            if DB_TYPE == "postgres":
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS subscriptions (user_id BIGINT PRIMARY KEY, username TEXT, expires_at TIMESTAMP);"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_subscriptions_expires ON subscriptions (expires_at);"
                )
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS questions (id SERIAL PRIMARY KEY, user_id BIGINT, question TEXT, answer TEXT, timestamp TIMESTAMP);"
                )
            else:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS subscriptions (
                        user_id INTEGER PRIMARY KEY,
                        username TEXT,
                        expires_at TIMESTAMP
                    );
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_subscriptions_expires ON subscriptions (expires_at);"
                )
                cur.execute(
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
            conn.commit()
        logger.info("Database schema ensured (%s) 🎉", DB_TYPE)
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

# ---------------------------------------------------------------------------#
# 3. Database DDL / DML helpers                                              #
# ---------------------------------------------------------------------------#

def save_subscription(user_id: int, username: Optional[str], days: int) -> None:
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
                "INSERT OR REPLACE INTO subscriptions (user_id, username, expires_at) VALUES (?, ?, ?)",
                (user_id, username, expires_at),
            )
    logger.info("Subscription saved for %s (until %s) 🎉", user_id, expires_at)

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
        try:
            expires = datetime.fromisoformat(expires)
        except ValueError:
            expires = datetime.strptime(expires, "%Y-%m-%d %H:%M:%S")
    return datetime.utcnow() < expires

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
    if row:
        expires = row[0]
        if isinstance(expires, str):
            expires = datetime.fromisoformat(expires)
        remaining = max((expires - datetime.utcnow()).days, 0)
        await update.message.reply_text(
            f"✅ اشتراک شما تا {expires.strftime('%Y-%m-%d')} معتبر است.\n({remaining} روز باقی‌مانده)"
        )
        return
    await update.message.reply_text("❌ اشتراک فعالی یافت نشد.")

# ---------------------------------------------------------------------------#
# 4. Handlers                                                                #
# ---------------------------------------------------------------------------#

TOKEN_IMG = Path(__file__).with_name("reblawcoin.png")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 به ربات RebLawCoin خوش آمدید!\n"
        "از دستورات زیر استفاده کنید:\n"
        "/buy — خرید اشتراک\n"
        "/status — وضعیت اشتراک\n"
        "/ask — سوال حقوقی\n"
        "/send_receipt — ارسال رسید\n"
        "/help — راهنما"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "دستورات:\n"
        "/start — منوی اصلی\n"
        "/help — این راهنما\n"
        "/send_receipt — ارسال رسید پرداخت\n"
        "/status — وضعیت اشتراک\n"
        "/ask <سوال> — پرسش و پاسخ حقوقی\n"
        "\n"
        "دستورات فارسی:\n"
        "خرید اشتراک — نحوه خرید اشتراک\n"
        "ارسال رسید — ارسال رسید پرداخت\n"
        "وضعیت اشتراک — بررسی وضعیت اشتراک\n"
        "سوال حقوقی — پرسش و پاسخ حقوقی\n"
        "درباره توکن — معرفی توکن RebLawCoin"
    )

async def buy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "<b>📌 روش‌های خرید اشتراک:</b>\n"
        f"1️⃣ واریز 2 TON → <code>{config['TON_WALLET_ADDR']}</code>\n"
        f"2️⃣ کارت‌به‌کارت ۵۰۰٬۰۰۰ تومان → <code>{config['BANK_CARD_NUMBER']}</code>\n"
        "🔸 سپس /send_receipt را ارسال کنید.\n"
        "✅ TON = اشتراک ۶ ماهه\n"
        "✅ کارت‌بانکی = اشتراک ۱ ماهه",
        parse_mode=ParseMode.HTML,
    )
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("📢 کانال تلگرام", url="https://t.me/RebLawCoin ")],
        [InlineKeyboardButton("📸 اینستاگرام", url="https://www.instagram.com/reblawcoin/ ")]
    ])
    await update.message.reply_text("🎉 RebLawCoin — اولین توکن حقوقی جهان", reply_markup=keyboard)

async def about_token(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logo = Path("reblawcoin.png")
    if logo.exists():
        await update.message.reply_photo(logo.open("rb"))
    text = (
        "🎉 به RebLawCoin خوش آمدید!\n"
        "💎 **توکن RebLawCoin (RLC)** اولین ارز دیجیتال حقوقی‌ست.\n"
        "**اهداف پروژه:**\n"
        "• سرمایه‌گذاری در طرح‌های حقوقی محلی و بین‌المللی\n"
        "• نهادینه‌سازی عدالت با بلاک‌چین\n"
        "• سوددهی پایدار برای سرمایه‌گذاران"
    )
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("📢 کانال تلگرام", url="https://t.me/RebLawCoin ")],
        [InlineKeyboardButton("📸 اینستاگرام", url="https://www.instagram.com/reblawcoin/ ")]
    ])
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=keyboard)

async def send_receipt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("لطفاً رسید را به صورت عکس یا متن ارسال کنید.")
    context.user_data["awaiting_receipt"] = True

async def handle_receipt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.user_data.get("awaiting_receipt"):
        return
    user = update.effective_user
    caption = (
        f"📥 رسید پرداخت از:\n"
        f"🆔 <code>{user.id}</code>\n"
        f"👤 @{user.username or '—'}"
    )
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ تأیید", callback_data=f"approve:{user.id}"),
        InlineKeyboardButton("❌ رد", callback_data=f"reject:{user.id}")
    ]])
    admin_id = config["ADMIN_ID"]
    if update.message.photo:
        await context.bot.send_photo(
            chat_id=admin_id,
            photo=update.message.photo[-1].file_id,
            caption=caption,
            reply_markup=keyboard,
            parse_mode=ParseMode.HTML
        )
    else:
        await context.bot.send_message(
            chat_id=admin_id,
            text=f"{caption}\n📝 {update.message.text}",
            reply_markup=keyboard,
            parse_mode=ParseMode.HTML
        )
    await update.message.reply_text("✅ رسید شما ثبت شد، منتظر بررسی مدیر باشید.")
    context.user_data["awaiting_receipt"] = False
    context.user_data["payment_type"] = "bank" if "کارت" in update.message.text else "ton"

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    action, user_id_str = query.data.split(":", 1)
    user_id = int(user_id_str)
    payment_type = context.user_data.get("payment_type", "ton")
    if action == "approve":
        if payment_type == "bank":
            save_subscription(user_id, "-", 30)  # یک ماه
        else:
            save_subscription(user_id, "-", 180)  # شش ماه
        await context.bot.send_message(user_id, "✅ اشتراک شما تایید شد. از خدمات لذت ببرید.")
        await query.edit_message_caption("✅ رسید تأیید شد.")
    else:
        await context.bot.send_message(user_id, "❌ متأسفیم، رسید شما تأیید نشد.")
        await query.edit_message_caption("❌ رسید رد شد.")

def save_question(user_id: int, question: str, answer: str) -> None:
    timestamp = datetime.utcnow()
    with get_conn() as conn:
        cur = conn.cursor()
        if DB_TYPE == "postgres":
            cur.execute(
                "INSERT INTO questions (user_id, question, answer, timestamp) VALUES (%s, %s, %s, %s)",
                (user_id, question, answer, timestamp),
            )
        else:
            cur.execute(
                "INSERT INTO questions (user_id, question, answer, timestamp) VALUES (?, ?, ?, ?)",
                (user_id, question, answer, timestamp),
            )
    logger.info("Q/A stored for %s at %s", user_id, timestamp)

async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not has_active_subscription(user_id):
        await update.message.reply_text("❌ شما اشتراک فعالی ندارید. ابتدا اشتراک خود را تمدید کنید.")
        return
    if len(context.args) == 0:
        await update.message.reply_text("❓ لطفاً سوال حقوقی خود را بعد از دستور `/ask` بنویسید.")
        return
    question = " ".join(context.args)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a legal expert assistant. Answer in Persian."},
                {"role": "user", "content": question}
            ],
            max_tokens=500,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        await update.message.reply_text(answer)
        save_question(user_id, question, answer)
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        await update.message.reply_text("⚠️ خطایی در پردازش سوال شما رخ داد. لطفاً دوباره امتحان کنید.")

# ---------------------------------------------------------------------------#
# 5. Main + remove_webhook                                                   #
# ---------------------------------------------------------------------------#

async def remove_webhook(app: Application) -> None:
    await app.bot.delete_webhook(drop_pending_updates=True)
    logger.info("وب‌هوک حذف و صف آپدیت‌ها ریست شد.")

def main() -> None:
    init_db()
    initialize_database()
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
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("ask", ask))
    app.add_handler(CommandHandler("send_receipt", send_receipt))

    # Persian commands
    from telegram.ext.filters import Regex
    import re
    app.add_handler(MessageHandler(filters.Regex(r"^[^\w]*(خرید اشتراک)[^\w]*$", re.IGNORECASE), buy))
    app.add_handler(MessageHandler(filters.Regex(r"^[^\w]*(ارسال رسید)[^\w]*$", re.IGNORECASE), send_receipt))
    app.add_handler(MessageHandler(filters.Regex(r"^[^\w]*(وضعیت اشتراک)[^\w]*$", re.IGNORECASE), status))
    app.add_handler(MessageHandler(filters.Regex(r"^[^\w]*(سوال حقوقی)[^\w]*$", re.IGNORECASE), ask))
    app.add_handler(MessageHandler(filters.Regex(r"^[^\w]*(درباره توکن|معرفی توکن|درباره ریبلوکوین)[^\w]*$", re.IGNORECASE), about_token))

    # Receipt handler
    receipt_filter = (filters.PHOTO | filters.TEXT) & ~filters.COMMAND
    app.add_handler(MessageHandler(receipt_filter, handle_receipt))

    # Admin callbacks
    app.add_handler(CallbackQueryHandler(callback_handler, pattern=r"^(approve|reject):"))

    logger.info("🤖 RebLawCoin_bot started successfully 🎉")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
