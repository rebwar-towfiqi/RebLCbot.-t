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

# ─────────────────── 1. Env & config ───────────────────
load_dotenv()
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    level=logging.INFO)
logger = logging.getLogger("RebLawCoin_bot")


def getenv_or_die(key: str) -> str:
    v = os.getenv(key)
    if not v:
        raise RuntimeError(f"Environment variable '{key}' is missing")
    return v


config = {
    "BOT_TOKEN": getenv_or_die("BOT_TOKEN"),
    "ADMIN_ID": int(getenv_or_die("ADMIN_ID")),
    "OPENAI_API_KEY": getenv_or_die("OPENAI_API_KEY"),
    "DATABASE_URL": getenv_or_die("DATABASE_URL"),
    "BANK_CARD_NUMBER": getenv_or_die("BANK_CARD_NUMBER"),
}

client = openai.OpenAI(api_key=config["OPENAI_API_KEY"])

# ─────────────────── 2. Database init ───────────────────
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
    global DB_TYPE, POOL
    try:
        POOL = SimpleConnectionPool(
            1, 10, dsn=config["DATABASE_URL"], sslmode="require", connect_timeout=10
        )
        POOL.putconn(POOL.getconn())
        DB_TYPE = "postgres"
        logger.info("Connected to PostgreSQL 🎉")
        return
    except Exception as e:  # noqa: BLE001
        logger.warning("Postgres init failed (%r); using SQLite.", e)

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
    logger.info("Using SQLite (%s) 🎉", SQLITE_PATH)

# ─────────────────── 3. DB helpers ───────────────────
def save_subscription(user_id: int, username: str | None, days: int) -> None:
    exp = datetime.utcnow() + timedelta(days=days)
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
                (user_id, username, exp),
            )
        else:
            cur.execute(
                "INSERT OR REPLACE INTO subscriptions (user_id, username, expires_at) "
                "VALUES (?, ?, ?)",
                (user_id, username, exp),
            )
        conn.commit()
    logger.info("Subscription saved for %s until %s", user_id, exp)


def has_active_subscription(user_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.cursor()
        q = ("SELECT expires_at FROM subscriptions WHERE user_id = %s"
             if DB_TYPE == "postgres"
             else "SELECT expires_at FROM subscriptions WHERE user_id = ?")
        cur.execute(q, (user_id,))
        row = cur.fetchone()
    if not row:
        return False
    exp = row[0]
    if isinstance(exp, str):
        exp = datetime.fromisoformat(exp)
    return datetime.utcnow() < exp


def save_question(user_id: int, qst: str, ans: str) -> None:
    ts = datetime.utcnow()
    with get_conn() as conn:
        cur = conn.cursor()
        if DB_TYPE == "postgres":
            cur.execute(
                "INSERT INTO questions (user_id, question, answer, timestamp) "
                "VALUES (%s, %s, %s, %s)",
                (user_id, qst, ans, ts),
            )
        else:
            cur.execute(
                "INSERT INTO questions (user_id, question, answer, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (user_id, qst, ans, ts),
            )
        conn.commit()
    logger.info("Q/A stored for %s", user_id)

# ─────────────────── 4. Utils ───────────────────
def get_reply_target(update: Update):
    """Return (message object, is_callback) for unified replies."""
    if update.message:
        return update.message, False
    if update.callback_query:
        return update.callback_query.message, True
    raise RuntimeError("No message in update")


async def send_long(update: Update, text: str, **kwargs) -> None:
    """Split long messages (Telegram limit 4096) and send sequentially."""
    msg, _ = get_reply_target(update)
    chunk = 4096
    while text:
        part = text[:chunk]
        # سعی می‌کنیم در محل خط جدید جدا کنیم
        if len(text) > chunk and "\n" in part:
            part = part.rsplit("\n", 1)[0]
        await msg.reply_text(part, **kwargs)
        text = text[len(part):]

# ─────────────────── 5. UI & handlers ───────────────────
def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🔐 Buy Subscription", callback_data="menu_buy")],
            [InlineKeyboardButton("📎 Send Receipt", callback_data="menu_send_receipt")],
            [InlineKeyboardButton("📅 Status", callback_data="menu_status")],
            [InlineKeyboardButton("⚖️ Ask a Question", callback_data="menu_ask")],
            [InlineKeyboardButton("💎 RebLawCoin Token", callback_data="menu_token_info")],
        ]
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg, is_cb = get_reply_target(update)
    if is_cb:
        await update.callback_query.answer()
    await msg.reply_text(
        "👋 به RebLawBot خوش آمدید!\n"
        "از منوی زیر گزینهٔ موردنظر را انتخاب کنید.\n\n"
        "📖 <b>دستورات:</b>\n"
        "/buy — خرید اشتراک\n"
        "/send_receipt — ارسال رسید پرداخت\n"
        "/status — وضعیت اشتراک\n"
        "/ask &lt;سؤال&gt; — پرسش حقوقی\n"
        "/token — معرفی و اهداف توکن RebLawCoin",
        parse_mode=ParseMode.HTML,
        reply_markup=main_menu(),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg, _ = get_reply_target(update)
    await msg.reply_text("ℹ️ برای مشاهدهٔ منو و دستورات، /start را ارسال کنید.")


async def buy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg, is_cb = get_reply_target(update)
    if is_cb:
        await update.callback_query.answer()
    await msg.reply_text(
        "<b>📌 روش خرید اشتراک:</b>\n"
        f"• کارت‌به‌کارت 300٬000 تومان → <code>{config['BANK_CARD_NUMBER']}</code>\n\n"
        "🔸 سپس دستور /send_receipt را ارسال کنید.\n"
        "✅ پرداخت کارت‌بانکی = اشتراک 2 ماهه",
        parse_mode=ParseMode.HTML,
    )


async def send_receipt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg, is_cb = get_reply_target(update)
    if is_cb:
        await update.callback_query.answer()
    await msg.reply_text("لطفاً رسید را به صورت عکس یا متن ارسال کنید.")
    context.user_data["awaiting_receipt"] = True


async def handle_receipt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.user_data.get("awaiting_receipt"):
        return
    usr = update.effective_user
    caption = (
        "📥 رسید پرداخت از:\n"
        f"🆔 <code>{usr.id}</code>\n"
        f"👤 @{usr.username or '—'}"
    )
    kb = InlineKeyboardMarkup(
        [[InlineKeyboardButton("✅ تأیید", callback_data=f"approve:{usr.id}"),
          InlineKeyboardButton("❌ رد", callback_data=f"reject:{usr.id}")]]
    )
    admin = config["ADMIN_ID"]
    if update.message.photo:
        await context.bot.send_photo(admin, update.message.photo[-1].file_id,
                                     caption=caption, reply_markup=kb,
                                     parse_mode=ParseMode.HTML)
    else:
        await context.bot.send_message(admin, f"{caption}\n📝 {update.message.text}",
                                       reply_markup=kb, parse_mode=ParseMode.HTML)
    await update.message.reply_text("✅ رسید شما ثبت شد، منتظر تأیید مدیر باشید.")
    context.user_data["awaiting_receipt"] = False


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    await q.answer()
    act, uid = q.data.split(":", 1)
    uid = int(uid)
    if act == "approve":
        save_subscription(uid, "-", 30)
        await context.bot.send_message(uid, "✅ اشتراک شما تأیید شد.")
        await q.edit_message_caption("✅ رسید تأیید شد.")
    else:
        await context.bot.send_message(uid, "❌ متأسفیم، رسید شما تأیید نشد.")
        await q.edit_message_caption("❌ رسید رد شد.")


async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg, _ = get_reply_target(update)
    uid = update.effective_user.id
    if not has_active_subscription(uid):
        await msg.reply_text("❌ شما اشتراک فعّالی ندارید.")
        return
    if not context.args:
        await msg.reply_text("❓ لطفاً سؤال را پس از /ask بنویسید.")
        return
    question = " ".join(context.args)
    try:
        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a legal expert assistant. Answer in Persian."},
                {"role": "user", "content": question},
            ],
            max_tokens=800,
            temperature=0.7,
        )
        ans = res.choices[0].message.content.strip()
        await send_long(update, ans)
        save_question(uid, question, ans)
    except Exception as e:  # noqa: BLE001
        logger.error("OpenAI error: %s", e)
        await msg.reply_text("⚠️ خطا در پردازش سؤال. بعداً دوباره تلاش کنید.")


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg, is_cb = get_reply_target(update)
    if is_cb:
        await update.callback_query.answer()
    uid = update.effective_user.id
    with get_conn() as conn:
        cur = conn.cursor()
        q = ("SELECT expires_at FROM subscriptions WHERE user_id = %s"
             if DB_TYPE == "postgres"
             else "SELECT expires_at FROM subscriptions WHERE user_id = ?")
        cur.execute(q, (uid,))
        row = cur.fetchone()
    if not row:
        await msg.reply_text("❌ اشتراک فعّالی یافت نشد.")
        return
    exp = row[0]
    if isinstance(exp, str):
        exp = datetime.fromisoformat(exp)
    remain = (exp - datetime.utcnow()).days
    await msg.reply_text(
        f"✅ اشتراک شما تا {exp:%Y-%m-%d} معتبر است ({remain} روز)."
    )


TOKEN_IMG = Path(__file__).with_name("reblawcoin.png")


async def about_token(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg, is_cb = get_reply_target(update)
    if is_cb:
        await update.callback_query.answer()
    if TOKEN_IMG.exists():
        await msg.reply_photo(TOKEN_IMG.open("rb"))
    await msg.reply_text(
        "🎉 <b>توکن RebLawCoin (RLC)</b> اولین ارز دیجیتال حقوقی است.\n\n"
        "<b>اهداف پروژه:</b>\n"
        "• سرمایه‌گذاری در طرح‌های حقوقی\n"
        "• نهادینه‌سازی عدالت با بلاک‌چین\n"
        "• سودآوری پایدار برای سرمایه‌گذاران",
        parse_mode=ParseMode.HTML,
    )

# ─────────────────── 6. Main ───────────────────
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

    # Slash commands
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

    # Receipt messages
    app.add_handler(MessageHandler((filters.PHOTO | filters.TEXT) & ~filters.COMMAND,
                                   handle_receipt))
    # Admin approve / reject
    app.add_handler(CallbackQueryHandler(callback_handler, pattern="^(approve|reject):"))

    logger.info("🤖 RebLawBot started successfully.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
