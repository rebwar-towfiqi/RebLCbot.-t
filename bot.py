#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RebLawBot – Telegram bot that sells subscriptions and answers legal questions using OpenAI.
Version 2025‑05‑12 (compat‑OpenAI 1.14)
"""
from __future__ import annotations
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
# 1️⃣ Environment & Global Configuration                                      #
# ---------------------------------------------------------------------------#
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger("RebLawBot")

def getenv_or_die(key: str) -> str:
    """
    دریافت مقدار متغیر محیطی؛ در صورت عدم وجود، خطا ایجاد می‌کند.
    """
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"Environment variable '{key}' is missing")
    return val

# Load environment variables
BOT_TOKEN = getenv_or_die("BOT_TOKEN")
ADMIN_ID = int(getenv_or_die("ADMIN_ID"))
OPENAI_API_KEY = getenv_or_die("OPENAI_API_KEY")
DATABASE_URL = getenv_or_die("DATABASE_URL")  # PostgreSQL connection string
BANK_CARD_NUMBER = getenv_or_die("BANK_CARD_NUMBER")

# Initialize OpenAI client
client = openai.ChatCompletion.create

# ---------------------------------------------------------------------------#
# 2️⃣ Application DB (PostgreSQL → fallback SQLite)                           #
# ---------------------------------------------------------------------------#
SQLITE_PATH = Path("users.db")
POOL: Optional[SimpleConnectionPool] = None
DB_TYPE = ""  # "postgres" or "sqlite"

def _ensure_schema(conn):
    """
    ایجاد جداول مورد نیاز در پایگاه داده.
    """
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
    logger.info("✅ Database schema ensured.")

def init_db():
    """
    راه‌اندازی پایگاه داده و انتخاب بین PostgreSQL یا SQLite.
    """
    global DB_TYPE, POOL
    dsn = DATABASE_URL.strip()
    try:
        # Try connecting to PostgreSQL
        POOL = SimpleConnectionPool(
            1, 10, dsn=dsn, sslmode="require", connect_timeout=10
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
def get_conn() -> Generator:
    """
    مدیریت اتصال به پایگاه داده با پشتیبانی از auto-commit.
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
# 3️⃣ Laws Database (Read-only SQLite)                                        #
# ---------------------------------------------------------------------------#
LAWS_DB = sqlite3.connect("iran_laws.db", check_same_thread=False)

def lookup(code: str, art_id: int) -> Optional[str]:
    """
    جستجو در پایگاه داده قوانین بر اساس کلید و شماره ماده.
    """
    try:
        cur = LAWS_DB.execute(
            "SELECT text FROM articles WHERE code=? AND id=?", 
            (code.lower(), art_id)
        )
        row = cur.fetchone()
        return row[0] if row else None
    except Exception as e:
        logger.error(f"❌ خطا در جستجوی ماده: {e}")
        return None

# ---------------------------------------------------------------------------#
# 4️⃣ Data Helpers                                                            #
# ---------------------------------------------------------------------------#
def _dt(val):
    """
    تبدیل مقدار دریافتی به نوع datetime.
    """
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(val)
    except Exception:
        return datetime.strptime(val, "%Y-%m-%d %H:%M:%S")

def save_subscription(uid: int, username: Optional[str], days: int = 60) -> None:
    """
    ذخیره اشتراک کاربر در دیتابیس.
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
    بررسی وضعیت اشتراک فعال برای کاربر.
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
    return bool(row and datetime.utcnow() < _dt(row[0]))

def get_subscription_expiry(uid: int) -> Optional[datetime]:
    """
    دریافت تاریخ انقضای اشتراک کاربر.
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
    return _dt(row[0]) if row else None

def save_question(uid: int, q: str, a: str) -> None:
    """
    ذخیره سؤال حقوقی و پاسخ آن در دیتابیس.
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
# 5️⃣ Utility Helpers                                                         #
# ---------------------------------------------------------------------------#
def get_reply(update: Update) -> Tuple[Message, bool]:
    """
    برگرداندن شیء Message مناسب و فلگ is_callback.
    """
    return ((update.callback_query.message, True)
            if update.callback_query else (update.message, False))

def chunks(text: str, limit: int = 4096) -> List[str]:
    """
    تقسیم متن‌های طولانی به بخش‌های کوچکتر برای ارسال در تلگرام.
    """
    import textwrap
    if len(text) <= limit:
        return [text]
    return textwrap.wrap(text, limit - 20, break_long_words=False)

async def send_long(update: Update, text: str, **kw):
    """
    ارسال متن‌های طولانی به کاربر در قالب چند پیام جداگانه.
    """
    msg, _ = get_reply(update)
    for part in chunks(text):
        await msg.reply_text(part, **kw)

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
    ساخت منوی اصلی بات تلگرام.
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
# 7️⃣ Handlers – Commands & Callbacks                                        #
# ---------------------------------------------------------------------------#
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    هندلر مربوط به دستور `/start`.
    """
    msg, is_cb = get_reply(update)
    if is_cb:
        await update.callback_query.answer()
    await msg.reply_text("👋 به RebLawBot خوش آمدید!", reply_markup=main_menu())

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    هندلر مربوط به دستور `/help`.
    """
    await update.message.reply_text("از دستور /start یا دکمه‌های منو استفاده کنید.")

async def buy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    هندلر مربوط به خرید اشتراک.
    """
    msg, is_cb = get_reply(update)
    if is_cb:
        await update.callback_query.answer()
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
    هندلر مربوط به ارسال رسید.
    """
    await update.message.reply_text("لطفاً رسید را به صورت عکس یا متن ارسال کنید.")
    context.user_data["awaiting_receipt"] = True

async def handle_receipt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    دریافت عکس/متن رسید – فقط وقتی در حال انتظار هستیم.
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
                photo=update.message.photo[-1].file_id,  # Largest size
                caption=caption,
                reply_markup=markup,
                parse_mode=ParseMode.HTML
            )
        else:
            await context.bot.send_message(
                chat_id=ADMIN_ID,
                text=f"{caption}\n📝 {update.message.text}",
                reply_markup=markup,
                parse_mode=ParseMode.HTML
            )
        await update.message.reply_text("✅ رسید شما ثبت شد، منتظر بررسی مدیر باشید.")
    except Exception as e:
        logger.error("Receipt forwarding error: %s", e)
        await update.message.reply_text("❌ خطا در ارسال رسید به مدیر.")
    finally:
        context.user_data["awaiting_receipt"] = False

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    هندلر مربوط به تأیید یا رد رسید.
    """
    query = update.callback_query
    await query.answer()
    action, user_id_str = query.data.split(":")
    user_id = int(user_id_str)
    is_approve = action == "approve"
    try:
        if is_approve:
            chat = await context.bot.get_chat(user_id)
            save_subscription(user_id, chat.username, days=180)
            await context.bot.send_message(user_id, "✅ اشتراک شما تایید شد.")
            await query.edit_message_caption("✅ رسید تایید شد.")
        else:
            await context.bot.send_message(user_id, "❌ متاسفیم، رسید شما رد شد.")
            await query.edit_message_caption("❌ رسید رد شد.")
    except Exception as e:
        logger.error("Callback handler error: %s", e)
        await query.edit_message_reply_markup(reply_markup=None)

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    هندلر مربوط به وضعیت اشتراک.
    """
    msg, _ = get_reply(update)
    uid = update.effective_user.id
    exp = get_subscription_expiry(uid)
    if exp and datetime.utcnow() < exp:
        await msg.reply_text(f"✅ اشتراک شما تا {exp:%Y-%m-%d} معتبر است.")
    else:
        await msg.reply_text("❌ اشتراک فعالی ثبت نشده است.")

LEGAL_DOCS_PATH = Path("legal_documents")
async def resources_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    هندلر مربوط به نمایش منابع حقوقی.
    """
    msg, _ = get_reply(update)
    docs = sorted(d.stem for d in LEGAL_DOCS_PATH.glob("*.txt"))
    if not docs:
        await msg.reply_text("هیچ سند حقوقی بارگذاری نشده است.")
        return
    await send_long(
        update,
        "📚 فهرست منابع حقوقی موجود:\n" + "\n".join(f"• {name}" for name in docs),
    )

async def legale_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    هندلر مربوط به جستجو در مواد قانونی.
    """
    if not context.args:
        await update.message.reply_text(
            "📌 دستور را این‌گونه بنویسید:\n"
            "/law <کلید> <شماره‌ماده>\n"
            "مثال: /law civil 300"
        )
        return
    code_key = context.args[0].lower()
    if len(context.args) < 2 or not context.args[1].isdigit():
        await update.message.reply_text("❌ شماره ماده نامعتبر است.")
        return
    article_id = int(context.args[1])
    text = lookup(code_key, article_id)
    if text:
        await send_long(
            update, f"📜 ماده {article_id} ({code_key.upper()})\n{text}"
        )
    else:
        await update.message.reply_text("❌ ماده پیدا نشد.")

TOKEN_IMG = Path(__file__).with_name("reblawcoin.png")
async def about_token(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    هندلر مربوط به نمایش اطلاعات توکن RebLawCoin.
    """
    msg, _ = get_reply(update)
    if TOKEN_IMG.exists():
        await msg.reply_photo(TOKEN_IMG.open("rb"))
    await msg.reply_text(
        "🎉 <b>توکن RebLawCoin (RLC)</b> – اولین ارز دیجیتال با محوریت خدمات حقوقی.\n"
        "<b>اهداف پروژه:</b>\n"
        "• سرمایه‌گذاری در نوآوری‌های حقوقی\n"
        "• نهادینه‌سازی عدالت روی بلاک‌چین\n"
        "• سودآوری پایدار برای سرمایه‌گذاران\n"
        "<a href=\"https://t.me/blum/app?startapp=memepadjetton_RLC_JpMH5-ref_1wgcKkl94N \">‏خرید RLC در Blum MemePad ↗</a>",
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )

async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    هندلر مربوط به سؤال حقوقی از طریق OpenAI.
    """
    uid = update.effective_user.id
    if not has_active_subscription(uid):
        await update.message.reply_text("❌ ابتدا اشتراک تهیه کنید.")
        return
    question = " ".join(context.args)
    if not question:
        await update.message.reply_text("❓ لطفاً سؤال حقوقی خود را بعد از دستور ارسال کنید.")
        return
    await update.message.reply_text("🧠 در حال پردازش...")
    try:
        response = client(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[{"role": "user", "content": question}]
        )
        answer = response.choices[0].message.content.strip()
        save_question(uid, question, answer)
        await send_long(update, answer)
    except openai.error.AuthenticationError:
        logger.error("Invalid OpenAI API key")
        await update.message.reply_text("❌ کلید OpenAI نامعتبر است.")
    except openai.error.RateLimitError:
        logger.warning("OpenAI rate limit")
        await update.message.reply_text("❌ سقف درخواست موقتاً پر شده است.")
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        await update.message.reply_text("❌ خطا در دریافت پاسخ.")

async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    گیرندهٔ عمومی متون.
    """
    message = update.message.text
    user_id = update.effective_user.id
    if context.user_data.get("awaiting_receipt"):
        context.user_data.pop("awaiting_receipt", None)
        await update.message.forward(ADMIN_ID)
        await update.message.reply_text("✅ رسید دریافت شد؛ پس از بررسی تأیید می‌شود.")
        logger.info(f"Receipt received from user {user_id}")
        return
    if context.user_data.get("awaiting_question"):
        context.user_data.pop("awaiting_question", None)
        await answer_question(update, context, message)
        return
    await update.message.reply_text(
        "❓ دستور نامشخص است. لطفاً از دکمه‌های منو استفاده کنید یا دستور مورد نظر خود را وارد کنید."
    )

async def answer_question(update: Update, context: ContextTypes.DEFAULT_TYPE, question: str):
    """
    پردازش و پاسخ به سؤال حقوقی کاربر.
    """
    uid = update.effective_user.id
    msg = update.effective_message
    await msg.chat.send_action(ChatAction.TYPING)
    try:
        response = client(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[{"role": "user", "content": question}]
        )
        answer = response.choices[0].message.content.strip()
        save_question(uid, question, answer)
        await send_long(update, answer)
        logger.info(f"✅ سؤال کاربر {uid} پردازش شد.")
    except openai.error.AuthenticationError:
        logger.error("Invalid OpenAI API key")
        await msg.reply_text("❌ کلید OpenAI نامعتبر است.")
    except openai.error.RateLimitError:
        logger.warning("OpenAI rate limit")
        await msg.reply_text("❌ سقف درخواست موقتاً پر شده است.")
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        await msg.reply_text("❌ خطا در دریافت پاسخ.")

async def menu_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    مسیریاب منو: هدایت به هندلر مرتبط بر اساس دکمه فشرده شده.
    """
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

def register_handlers(app: Application):
    """
    ثبت تمام هندلرها در بات.
    """
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("buy", buy))
    app.add_handler(CommandHandler("send_receipt", send_receipt))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("resources", resources_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(CommandHandler("law", legale_document))
    app.add_handler(CommandHandler("token", about_token))
    app.add_handler(CallbackQueryHandler(menu_router))
    app.add_handler(
        MessageHandler(
            filters.PHOTO | (filters.TEXT & ~filters.COMMAND),
            handle_receipt
        ),
        group=1,   # اول اجرا شود
    )

    # هندلر عمومی بعد از آن
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, text_router),
        group=2,
    )

def main() -> None:
    """
    تابع اصلی برای راه‌اندازی بات تلگرام.
    """
    init_db()
    application = Application.builder().token(BOT_TOKEN).build()
    register_handlers(application)
    logger.info("🤖 RebLawBot started successfully …")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
