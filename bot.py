from __future__ import annotations
"""
RebLawBot – Telegram bot that sells subscriptions and answers legal questions using OpenAI.
Refactored 2025-05-12

Key changes
===========
* PEP 8 compliance & type-hinting throughout
* Central `Config` dataclass (env loading & validation)
* `Database` class encapsulating PostgreSQL ↔ SQLite dual-mode logic
* Removed duplicated blocks (e.g. duplicate LAWS_DB)
* Added missing handler implementations (`send_receipt`, `about_token`)
* Wrapped long texts in helpers & constants
* Added `register_handlers()` and `main()` entry point
* Improved logging & graceful shutdown
"""

# ---------------------------------------------------------------------------#
# 0. Imports                                                                  #
# ---------------------------------------------------------------------------#
import asyncio
import logging
import os
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import AsyncGenerator, Generator, List, Optional, Tuple

import openai  # type: ignore
from dotenv import load_dotenv
from psycopg2.pool import SimpleConnectionPool  # type: ignore
from telegram import (InlineKeyboardButton, InlineKeyboardMarkup, Message, Update)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler,
                          ContextTypes, MessageHandler, filters)

__all__ = [
    "main",
]

# ---------------------------------------------------------------------------#
# 1. Environment & global configuration                                       #
# ---------------------------------------------------------------------------#

load_dotenv()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("RebLawBot")


def _env_or_die(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"Environment variable '{key}' is missing")
    return val


@dataclass(frozen=True, slots=True)
class Config:
    bot_token: str = _env_or_die("BOT_TOKEN")
    admin_id: int = int(_env_or_die("ADMIN_ID"))
    openai_api_key: str = _env_or_die("OPENAI_API_KEY")
    database_url: str = _env_or_die("DATABASE_URL")
    bank_card_number: str = _env_or_die("BANK_CARD_NUMBER")


CONFIG = Config()
client = openai.OpenAI(api_key=CONFIG.openai_api_key)

# ---------------------------------------------------------------------------#
# 2. Database layer                                                           #
# ---------------------------------------------------------------------------#

SQLITE_PATH = Path("users.db")
LAWS_DB_PATH = Path("iran_laws.db")


class Database:
    """Dual-backing store: PostgreSQL (preferred) ⇄ SQLite (fallback)."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn.strip()
        self._db_type: str = ""  # "postgres" | "sqlite"
        self._pool: Optional[SimpleConnectionPool] = None
        self._init_primary()
        self._ensure_schema()

    # ------------------------ public helpers ------------------------ #

    def save_subscription(self, uid: int, username: Optional[str], *, days: int = 60) -> None:
        exp = datetime.utcnow() + timedelta(days=days)
        with self._conn() as conn:
            cur = conn.cursor()
            if self._db_type == "postgres":
                cur.execute(
                    """INSERT INTO subscriptions (user_id, username, expires_at)
                           VALUES (%s, %s, %s)
                           ON CONFLICT (user_id) DO UPDATE
                             SET username = COALESCE(EXCLUDED.username, subscriptions.username),
                                 expires_at = EXCLUDED.expires_at""",
                    (uid, username, exp),
                )
            else:  # sqlite
                cur.execute(
                    "INSERT OR REPLACE INTO subscriptions VALUES (?,?,?)",
                    (uid, username, exp),
                )
        logger.info("Subscription %s → %s", uid, exp)

    def has_active_subscription(self, uid: int) -> bool:
        with self._conn() as conn:
            cur = conn.cursor()
            sql = (
                "SELECT expires_at FROM subscriptions WHERE user_id = %s"
                if self._db_type == "postgres"
                else "SELECT expires_at FROM subscriptions WHERE user_id = ?"
            )
            cur.execute(sql, (uid,))
            row = cur.fetchone()
        return bool(row and datetime.utcnow() < self._as_datetime(row[0]))

    def save_question(self, uid: int, question: str, answer: str) -> None:
        with self._conn() as conn:
            cur = conn.cursor()
            sql = (
                "INSERT INTO questions (user_id, question, answer, timestamp) VALUES (%s,%s,%s,%s)"
                if self._db_type == "postgres"
                else "INSERT INTO questions VALUES (NULL,?,?,?,?)"
            )
            cur.execute(sql, (uid, question, answer, datetime.utcnow()))
        logger.debug("Stored Q&A for user %s", uid)

    # ---------------------- internal plumbing ---------------------- #

    def _init_primary(self) -> None:
        try:
            self._pool = SimpleConnectionPool(1, 10, dsn=self._dsn, sslmode="require", connect_timeout=10)
            self._db_type = "postgres"
            logger.info("Connected to PostgreSQL ✨")
        except Exception as exc:  # pragma: no cover
            logger.warning("PostgreSQL init failed (%r); falling back to SQLite.", exc)
            self._db_type = "sqlite"
            SQLITE_PATH.touch(exist_ok=True)

    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """CREATE TABLE IF NOT EXISTS subscriptions (
                        user_id   BIGINT PRIMARY KEY,
                        username  TEXT,
                        expires_at TIMESTAMP
                    );""",
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_subscriptions_expires ON subscriptions (expires_at);")
            cur.execute(
                """CREATE TABLE IF NOT EXISTS questions (
                        id        BIGSERIAL PRIMARY KEY,
                        user_id   BIGINT,
                        question  TEXT,
                        answer    TEXT,
                        timestamp TIMESTAMP
                    );""",
            )
            conn.commit()
            logger.debug("DB schema ensured (%s)", self._db_type)

    @contextmanager
    def _conn(self):  # type: ignore[override]
        if self._db_type == "postgres":
            assert self._pool is not None  # for mypy
            conn = self._pool.getconn()
            try:
                yield conn
            finally:
                conn.commit()
                self._pool.putconn(conn)
        else:  # sqlite
            conn = sqlite3.connect(
                SQLITE_PATH,
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            )
            try:
                yield conn
            finally:
                conn.commit()
                conn.close()

    # -------------------------------------------------------------------------

    @staticmethod
    def _as_datetime(val):  # type: ignore[return-value]
        if isinstance(val, datetime):
            return val
        try:
            return datetime.fromisoformat(val)  # type: ignore[arg-type]
        except ValueError:
            return datetime.strptime(val, "%Y-%m-%d %H:%M:%S")


DB = Database(CONFIG.database_url)

# ---------------------------------------------------------------------------#
# 3. Law articles DB (read-only SQLite)                                       #
# ---------------------------------------------------------------------------#

LAWS_DB = sqlite3.connect(LAWS_DB_PATH, check_same_thread=False)


def lookup(code: str, art_id: int) -> Optional[str]:
    """Return the law article text, or *None* if not found."""
    cur = LAWS_DB.execute("SELECT text FROM articles WHERE code=? AND id=?", (code.lower(), art_id))
    row = cur.fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------#
# 4. Telegram helpers                                                         #
# ---------------------------------------------------------------------------#


def _reply_target(update: Update) -> Tuple[Message, bool]:
    """Return (message_to_reply_to, is_callback_query)."""
    return (
        (update.callback_query.message, True)
        if update.callback_query
        else (update.message, False)  # type: ignore[arg-type]
    )


def _split_long(txt: str, *, limit: int = 4096) -> List[str]:
    import textwrap

    return [txt] if len(txt) <= limit else textwrap.wrap(txt, limit - 32, break_long_words=False)


async def _send_long(update: Update, text: str, **kwargs):
    msg, _ = _reply_target(update)
    for part in _split_long(text):
        await msg.reply_text(part, **kwargs)


async def _typing(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    """Utility to show *typing…* for long operations."""
    await context.bot.send_chat_action(chat_id, ChatAction.TYPING)


# ---------------------------------------------------------------------------#
# 5. Menu & static texts                                                      #
# ---------------------------------------------------------------------------#

MAIN_MENU = InlineKeyboardMarkup(
    [
        [InlineKeyboardButton("🔐 خرید اشتراک", callback_data="menu_buy")],
        [InlineKeyboardButton("📎 ارسال رسید", callback_data="menu_send_receipt")],
        [InlineKeyboardButton("📅 وضعیت اشتراک", callback_data="menu_status")],
        [InlineKeyboardButton("⚖️ سؤال حقوقی", callback_data="menu_ask")],
        [InlineKeyboardButton("📘 منابع حقوقی", callback_data="menu_resources")],
        [InlineKeyboardButton("💎 توکن RebLawCoin", callback_data="menu_token")],
    ]
)

# ---------------------------------------------------------------------------#
# 6. Handlers                                                                 #
# ---------------------------------------------------------------------------#


async def h_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # /start
    msg, is_cb = _reply_target(update)
    if is_cb:
        await update.callback_query.answer()
    await msg.reply_text("👋 به RebLawBot خوش آمدید!", reply_markup=MAIN_MENU)


async def h_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # /help
    await update.message.reply_text("/start را بزنید و از منوی اینلاین استفاده کنید.")


async def h_buy(update: Update, context: ContextTypes.DEFAULT_TYPE):  # button: buy
    msg, is_cb = _reply_target(update)
    if is_cb:
        await update.callback_query.answer()
    await msg.reply_text(
        (
            "<b>📌 روش خرید اشتراک:</b>\n"
            "• کارت‌به‌کارت ۳۰۰٬۰۰۰ تومان → <code>{card}</code>\n\n"
            "پس از پرداخت، /send_receipt را ارسال کنید."
        ).format(card=CONFIG.bank_card_number),
        parse_mode=ParseMode.HTML,
    )


async def h_send_receipt(update: Update, context: ContextTypes.DEFAULT_TYPE):  # /send_receipt or button
    msg, is_cb = _reply_target(update)
    if is_cb:
        await update.callback_query.answer()

    if not (update.message and update.message.photo):
        await msg.reply_text("لطفاً عکس رسید را ارسال کنید.")
        return

    # Save the receipt image locally (simple proof-of-concept)
    photo = update.message.photo[-1]
    file_path = Path("receipts") / f"{update.effective_user.id}_{photo.file_id}.jpg"
    file_path.parent.mkdir(exist_ok=True)
    await photo.get_file().download_to_drive(str(file_path))

    # Grant subscription
    DB.save_subscription(update.effective_user.id, update.effective_user.username)

    await msg.reply_text("✅ رسید دریافت شد؛ اشتراک شما فعال شد.")
    logger.info("Receipt stored → %s", file_path)


TOKEN_IMG = Path(__file__).with_name("reblawcoin.png")


async def h_about_token(update: Update, context: ContextTypes.DEFAULT_TYPE):  # button: token
    msg, is_cb = _reply_target(update)
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


async def h_law(update: Update, context: ContextTypes.DEFAULT_TYPE):  # /law <code> <art>
    if len(context.args) != 2 or not context.args[1].isdigit():
        await update.message.reply_text(
            "📌 لطفاً دستور را به‌صورت /law <کدقانون> <شماره‌ماده> بنویسید.\nمثال: /law civil 300",
        )
        return

    code_key, art_id_str = context.args
    article = lookup(code_key, int(art_id_str))
    if article:
        await _send_long(update, f"📜 ماده {art_id_str} ({code_key})\n\n{article}")
    else:
        await update.message.reply_text("ماده پیدا نشد.")


# ---------------------------- Q&A workflow ---------------------------- #

async def _ask_openai(question: str) -> str:
    """Call OpenAI and return the answer text (simple chat-completion)."""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are an Iranian legal expert."}, {"role": "user", "content": question}],
        max_tokens=512,
        temperature=0.2,
    )
    return completion.choices[0].message.content.strip()


async def h_ask(update: Update, context: ContextTypes.DEFAULT_TYPE):  # button: ask
    msg, is_cb = _reply_target(update)
    if is_cb:
        await update.callback_query.answer()

    if not DB.has_active_subscription(update.effective_user.id):
        await msg.reply_text("❌ ابتدا اشتراک خریداری کنید.")
        return

    await msg.reply_text("✍️ سؤال خود را ارسال کنید.")

    # Wait for next user message (simple conversation-like flow)
    try:
        user_msg: Message = await context.application.wait_for_message(
            filters=filters.TEXT & filters.Chat(update.effective_chat.id),
            timeout=120,
        )
    except asyncio.TimeoutError:
        await msg.reply_text("⏱ زمان شما به پایان رسید.")
        return

    question = user_msg.text.strip()
    await _typing(context, update.effective_chat.id)

    try:
        answer = await _ask_openai(question)
    except Exception as exc:  # pragma: no cover
        logger.exception("OpenAI failure: %s", exc)
        await user_msg.reply_text("خطا در پردازش پاسخ. بعداً تلاش کنید.")
        return

    DB.save_question(update.effective_user.id, question, answer)
    await _send_long(user_msg, answer, parse_mode=ParseMode.MARKDOWN)


# ---------------------------------------------------------------------------#
# 7. Application bootstrap                                                    #
# ---------------------------------------------------------------------------#


def register_handlers(app: Application) -> None:
    """Register all command & callback handlers on the application."""

    # Commands
    app.add_handler(CommandHandler("start", h_start))
    app.add_handler(CommandHandler("help", h_help))
    app.add_handler(CommandHandler("law", h_law))
    app.add_handler(CommandHandler("send_receipt", h_send_receipt))

    # Callback buttons (menu_*)
    app.add_handler(CallbackQueryHandler(h_buy, pattern="^menu_buy$"))
    app.add_handler(CallbackQueryHandler(h_send_receipt, pattern="^menu_send_receipt$"))
    app.add_handler(CallbackQueryHandler(h_about_token, pattern="^menu_token$"))
    app.add_handler(CallbackQueryHandler(h_ask, pattern="^menu_ask$"))


async def _on_startup(app: Application):
    logger.info("Bot started ➜ @%s", (await app.bot.get_me()).username)


async def _on_shutdown(app: Application):
    logger.info("Bot shutting down…")


def main() -> None:
    application = (
        Application.builder()
        .token(CONFIG.bot_token)
        .post_init(_on_startup)
        .post_shutdown(_on_shutdown)
        .build()
    )

    register_handlers(application)

    logger.info("Polling…")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
