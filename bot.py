from __future__ import annotations
"""RebLawCoin_bot – Telegram bot that sells subscriptions and answers legal questions using OpenAI.
Copyright © 2025 Rebwar Lawyer
"""

import os
import logging
import sqlite3
import asyncio
import openai
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
from typing import Generator, Optional
from psycopg2.pool import SimpleConnectionPool
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup
from telegram.constants import ParseMode, ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from dotenv import load_dotenv

# ---------------------------------------------------------------------------#
# 1. Environment variables & Configuration                                   #
# ---------------------------------------------------------------------------#
load_dotenv()
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
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
    "DATABASE_URL": os.getenv("DATABASE_URL", ""),
    "TON_WALLET_ADDR": getenv_or_die("TON_WALLET_ADDRESS"),
    "RLC_WALLET_ADDR": getenv_or_die("RLC_WALLET_ADDRESS"),
    "BANK_CARD_NUMBER": getenv_or_die("BANK_CARD_NUMBER"),
    "MEMEPAD_LINK": getenv_or_die("MEMEPAD_LINK"),
}

# Initialize OpenAI client
client = openai.OpenAI(api_key=config["OPENAI_API_KEY"])

# ---------------------------------------------------------------------------#
# 2. Database initialization & connection                                    #
# ---------------------------------------------------------------------------#
SQLITE_PATH = Path("users.db")
DB_TYPE: str = ""
POOL: Optional[SimpleConnectionPool] = None


def init_db() -> None:
    """
    Try PostgreSQL via DATABASE_URL; on failure or missing, fallback to SQLite.
    """
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
            logger.warning(f"Postgres init failed ({e!r}), falling back to SQLite.")
    # SQLite fallback
    DB_TYPE = "sqlite"
    SQLITE_PATH.touch(exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH)
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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_subscriptions_expires ON subscriptions (expires_at);")
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
    conn.commit()
    conn.close()
    logger.info("Using SQLite database (%s) 🎉", SQLITE_PATH)


@contextmanager
def get_conn() -> Generator:
    if DB_TYPE == "postgres" and POOL:
        conn = POOL.getconn()
        try:
            yield conn
            conn.commit()
        finally:
            POOL.putconn(conn)
    else:
        conn = sqlite3.connect(SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

# ---------------------------------------------------------------------------#
# 3. Database DDL / DML helpers                                              #
# ---------------------------------------------------------------------------#
def initialize_database() -> None:
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            if DB_TYPE == "postgres":
                cur.execute("CREATE TABLE IF NOT EXISTS subscriptions (user_id BIGINT PRIMARY KEY, username TEXT, expires_at TIMESTAMP);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_subscriptions_expires ON subscriptions (expires_at);")
                cur.execute("CREATE TABLE IF NOT EXISTS questions (id SERIAL PRIMARY KEY, user_id BIGINT, question TEXT, answer TEXT, timestamp TIMESTAMP);")
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
                cur.execute("CREATE INDEX IF NOT EXISTS idx_subscriptions_expires ON subscriptions (expires_at);")
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


def save_subscription(user_id: int, username: Optional[str], days: int) -> None:
    expires_at = datetime.utcnow() + timedelta(days=days)
    with get_conn() as conn:
        cur = conn.cursor()
        if DB_TYPE == "postgres":
            cur.execute(
                """
                INSERT INTO subscriptions (user_id, username, expires_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET username = COALESCE(EXCLUDED.username, subscriptions.username), expires_at = EXCLUDED.expires_at
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
        query = "SELECT expires_at FROM subscriptions WHERE user_id = %s" if DB_TYPE == "postgres" else "SELECT expires_at FROM subscriptions WHERE user_id = ?"
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

async def main_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup([["/buy","/send_receipt"],["/status","/help"]], resize_keyboard=True)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "سلام! به RebLawCoin_bot خوش آمدید.\nبرای راهنما /help را ارسال کنید.",
        reply_markup=await main_menu(),
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "دستورات موجود:\n/start\n/help\n/buy\n/send_receipt\n/status\n/ask <سوال>"
    )

TOKEN_IMG = Path(__file__).with_name("reblawcoin.png")

async def buy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if TOKEN_IMG.exists():
        try:
            await update.message.reply_photo(
                photo=TOKEN_IMG.read_bytes(),
                caption=(
                    "<b>📌 روش خرید RLC:</b>\n"
                    "1️⃣ ورود به Blum: \n"
                    f"https://t.me/BlumCryptoBot?start={config['MEMEPAD_LINK']}\n"
                    "2️⃣ اتصال کیف پول.\n"
                    "3️⃣ جستجوی RLC در Memepad.\n"
                    "4️⃣ خرید با TON."
                ),
                parse_mode=ParseMode.HTML,
            )
        except Exception as exc:
            logger.warning(f"Cannot send RLC image: {exc}")
            await update.message.reply_text("⚠️ تصویر توکن قابل نمایش نیست.")
    else:
        await update.message.reply_text("❌ تصویر توکن در دسترس نیست.")

    await update.message.reply_text(
        "<b>📌 روش‌های خرید اشتراک:</b>\n" +
        f"1️⃣ واریز 2 TON → <code>{config['TON_WALLET_ADDR']}</code>\n" +
        f"2️⃣ واریز 1,800,000 RLC → <code>{config['RLC_WALLET_ADDR']}</code>\n" +
        f"3️⃣ کارت‌به‌کارت ۵۰٬۰۰۰ تومان → <code>{config['BANK_CARD_NUMBER']}</code>\n" +
        "🔸 سپس /send_receipt را ارسال کنید.\n" +
        "✅ TON/RLC = اشتراک ۶ ماهه\n" +
        "✅ کارت‌بانکی = اشتراک ۷ روزه",
        parse_mode=ParseMode.HTML,
    )
    
def save_question(user_id: int, question: str, answer: str) -> None:
    """
    Save a user's question and the AI's answer to the database with timestamp.
    """
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

# ---------------------------------------------------------------------------#
# 4. Handlers                                                                #
# ---------------------------------------------------------------------------#
def main_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup([["/buy","/send_receipt"],["/status","/help"]], resize_keyboard=True)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "سلام! به RebLawCoin_bot خوش آمدید.\nبرای راهنما /help را ارسال کنید.",
        reply_markup=main_menu(),
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "دستورات موجود:\n"
        "/start\n"
        "/help\n"
        "/buy\n"
        "/send_receipt\n"
        "/status\n"
        "/ask <سوال>"
    )

async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user

    # لاگ برای اطمینان از ورود به handler
    logger.info("📝 ask handler called by user %s with args: %s", user.id, context.args)

    if not has_active_subscription(user.id):
        await update.message.reply_text("⛔ اشتراک ندارید یا منقضی شده است.")
        return
    if not context.args:
        await update.message.reply_text("❓ لطفاً سوال را بعد از /ask بنویسید.")
        return

    question = " ".join(context.args)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        # اجرای synchronous call در executor تا بلوک نشود
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0,
                messages=[{"role": "user", "content": question}],
            )
        )
        answer = response.choices[0].message.content.strip()
        save_question(user.id, question, answer)
        await update.message.reply_text(answer)
    except openai.AuthenticationError:
        logger.error("Invalid OpenAI API key")
        await update.message.reply_text("❌ کلید OpenAI نامعتبر است.")
    except openai.RateLimitError:
        logger.warning("OpenAI rate limit exceeded")
        await update.message.reply_text("❌ سقف درخواست موقتاً پر شده است.")
    except openai.OpenAIError as e:
        logger.error("OpenAI error: %s", e)
        await update.message.reply_text("❌ خطا در دریافت پاسخ از سرویس OpenAI.")
    except Exception as e:
        logger.error("Unexpected error in ask: %s", e)
        await update.message.reply_text("❌ خطای داخلی رخ داده است.")

TOKEN_IMG = Path(__file__).with_name("RebLCbot.png")

async def buy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if TOKEN_IMG.exists():
        try:
            await update.message.reply_photo(
                photo=TOKEN_IMG.read_bytes(),
                caption=(
                    "<b>📌 روش خرید RLC:</b>\n"
                    "1️⃣ ورود به Blum: \n"
                    f"https://t.me/BlumCryptoBot?start={config['MEMEPAD_LINK']}\n"
                    "2️⃣ اتصال کیف پول.\n"
                    "3️⃣ جستجوی RLC در Memepad.\n"
                    "4️⃣ خرید با TON."
                ),
                parse_mode=ParseMode.HTML,
            )
        except Exception as exc:
            logger.warning(f"Cannot send RLC image: {exc}")
            await update.message.reply_text("⚠️ تصویر توکن قابل نمایش نیست.")
    else:
        await update.message.reply_text("❌ تصویر توکن در دسترس نیست.")

    await update.message.reply_text(
        "<b>📌 روش‌های خرید اشتراک:</b>\n"
        f"1️⃣ واریز 2 TON → <code>{config['TON_WALLET_ADDR']}</code>\n"
        f"2️⃣ واریز 1,800,000 RLC → <code>{config['RLC_WALLET_ADDR']}</code>\n"
        f"3️⃣ کارت‌به‌کارت ۵۰٬۰۰۰ تومان → <code>{config['BANK_CARD_NUMBER']}</code>\n"
        "🔸 سپس /send_receipt را ارسال کنید.\n"
        "✅ TON/RLC = اشتراک ۶ ماهه\n"
        "✅ کارت‌بانکی = اشتراک ۷ روزه",
        parse_mode=ParseMode.HTML,
    )

async def send_receipt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("لطفاً رسید را به صورت عکس یا متن ارسال کنید.")
    context.user_data["awaiting_receipt"] = True

async def handle_receipt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.user_data.get("awaiting_receipt"):
        return

    user = update.effective_user
    if not (update.message.photo or update.message.text):
        await update.message.reply_text("❌ فقط عکس یا متن قابل قبول است.")
        return

    caption = (
        f"📥 رسید پرداخت\n"
        f"ID: <code>{user.id}</code>\n"
        f"👤 @{user.username or '—'}"
    )
    markup = InlineKeyboardMarkup([[InlineKeyboardButton("✅ تأیید", callback_data=f"approve:{user.id}"), InlineKeyboardButton("❌ رد", callback_data=f"reject:{user.id}")]])

    try:
        if update.message.photo:
            await context.bot.send_photo(chat_id=config["ADMIN_ID"], photo=update.message.photo[-1].file_id, caption=caption, reply_markup=markup, parse_mode=ParseMode.HTML)
        else:
            await context.bot.send_message(chat_id=config["ADMIN_ID"], text=f"{caption}\n📝 {update.message.text}", reply_markup=markup, parse_mode=ParseMode.HTML)
        await update.message.reply_text("✅ رسید شما ثبت شد، منتظر بررسی مدیر باشید.")
    except Exception as e:
        logger.error("Receipt forwarding error: %s", e)
        await update.message.reply_text("❌ خطا در ارسال رسید به مدیر.")
    finally:
        context.user_data["awaiting_receipt"] = False

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    action, uid_str = query.data.split(":")
    uid = int(uid_str)
    if action == "approve":
        chat = await context.bot.get_chat(uid)
        save_subscription(uid, chat.username, days=180)
        await context.bot.send_message(uid, "✅ اشتراک شما تایید شد.")
        await query.edit_message_caption("✅ رسید تایید شد.")
    else:
        await context.bot.send_message(uid, "❌ متاسفیم، رسید شما رد شد.")
        await query.edit_message_caption("❌ رسید رد شد.")

async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not has_active_subscription(user.id):
        await update.message.reply_text("⛔ اشتراک ندارید یا منقضی شده است.")
        return
    if not context.args:
        await update.message.reply_text("❓ لطفاً سوال را بعد از /ask بنویسید.")
        return
    question = " ".join(context.args)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    response = client.chat.completions.create(model="gpt-3.5-turbo", temperature=0, messages=[{"role": "user", "content": question}])
    answer = response.choices[0].message.content.strip()
    save_question(user.id, question, answer)
    await update.message.reply_text(answer)

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    with get_conn() as conn:
        cur = conn.cursor()
        query = "SELECT expires_at FROM subscriptions WHERE user_id = %s" if DB_TYPE == "postgres" else "SELECT expires_at FROM subscriptions WHERE user_id = ?"
        cur.execute(query, (user_id,))
        row = cur.fetchone()
    if row:
        expires = row[0]
        remaining = max((expires - datetime.utcnow()).days, 0)
        await update.message.reply_text(f"✅ اشتراک شما تا {expires:%Y-%m-%d} معتبر است.\n({remaining} روز باقی‌مانده)")
    else:
        await update.message.reply_text("❌ اشتراک فعالی یافت نشد.")

# ---------------------------------------------------------------------------#
# 5. Main                                                                    #
# ---------------------------------------------------------------------------#

def main() -> None:
    init_db()
    initialize_database()
    app = Application.builder().token(config["BOT_TOKEN"]).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("buy", buy))
    app.add_handler(CommandHandler("send_receipt", handle_receipt))
    app.add_handler(CommandHandler("ask", ask))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CallbackQueryHandler(callback_handler))
    receipt_filter = (filters.TEXT | filters.PHOTO) & (~filters.COMMAND)
    app.add_handler(MessageHandler(receipt_filter, handle_receipt))
    logger.info("🤖 RebLawCoin_bot started successfully 🎉")
    logger.info("🚀 About to start polling…")
    print("🚀 Polling starting…")
    app.run_polling()

if __name__ == "__main__":
    main()
