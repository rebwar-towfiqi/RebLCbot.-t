import os
import logging
import sqlite3
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
import re

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

# Define Persian commands with regex patterns
BUY_REGEX = re.compile(r"^[^\w]*(خرید اشتراک)[^\w]*$", re.IGNORECASE)
SEND_RECEIPT_REGEX = re.compile(r"^[^\w]*(ارسال رسید)[^\w]*$", re.IGNORECASE)
STATUS_REGEX = re.compile(r"^[^\w]*(وضعیت اشتراک)[^\w]*$", re.IGNORECASE)
ASK_REGEX = re.compile(r"^[^\w]*(سوال حقوقی)[^\w]*$", re.IGNORECASE)
ABOUT_TOKEN_REGEX = re.compile(r"^[^\w]*(درباره توکن|معرفی توکن|درباره ریبلوکوین)[^\w]*$", re.IGNORECASE)

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
# 3. Main + remove_webhook                                                   #
# ---------------------------------------------------------------------------

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
    app.add_handler(MessageHandler(BUY_REGEX, buy))
    app.add_handler(MessageHandler(SEND_RECEIPT_REGEX, send_receipt))
    app.add_handler(MessageHandler(STATUS_REGEX, status))
    app.add_handler(MessageHandler(ASK_REGEX, ask))
    app.add_handler(MessageHandler(ABOUT_TOKEN_REGEX, about_token))

    # Receipt handler
    receipt_filter = (filters.PHOTO | filters.TEXT) & ~filters.COMMAND
    app.add_handler(MessageHandler(receipt_filter, handle_receipt))

    # Admin callbacks
    app.add_handler(CallbackQueryHandler(callback_handler, pattern=r"^(approve|reject):"))

    logger.info("🤖 RebLawCoin_bot started successfully 🎉")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
