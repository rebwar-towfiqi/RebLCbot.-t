#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RebLawBot – Telegram bot that sells subscriptions and answers legal questions using OpenAI.
Version 2025-05-13 (compat OpenAI 1.x)
"""

from __future__ import annotations

# ─── استاندارد کتابخانه ───────────────────────────────────────────────────────
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Generator, List, Optional, Tuple

# ─── کتابخانه‌های خارجی ───────────────────────────────────────────────────────
from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError, RateLimitError, AuthenticationError
from psycopg2.pool import SimpleConnectionPool
from telegram import (
    ReplyKeyboardMarkup, KeyboardButton,
    InlineKeyboardButton, InlineKeyboardMarkup,
    Message, Update,
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application, CallbackQueryHandler, CommandHandler,
    ContextTypes, MessageHandler, filters,
)

# ─── محیط و تنظیمات جهانی ─────────────────────────────────────────────────────
load_dotenv()  # بارگذاری فایل .env

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,  # سطح لاگ را از DEBUG به INFO کاهش دادم برای عملکرد Production
)
logger = logging.getLogger("RebLawBot")

# کلاینت OpenAI
client = AsyncOpenAI()

# ─── ابزارهای کمکی ─────────────────────────────────────────────────────────────
def getenv_or_die(key: str) -> str:
    """
    مقدار متغیر محیطی را می‌گیرد یا در صورت عدم وجود، خطا می‌دهد.
    """
    value = os.getenv(key)
    if not value:
        logger.critical(f"Missing required environment variable: {key}")
        raise RuntimeError(f"Environment variable {key!r} is missing")
    return value

# ---------------------------------------------------------------------------#
# 1. Database layer – PostgreSQL → SQLite fallback                           #
# ---------------------------------------------------------------------------#
import threading

# مسیر فایل SQLite برای استفاده محلی
SQLITE_FILE = Path("users.db")
POOL: Optional[SimpleConnectionPool] = None
USE_PG = False  # بعد از init_db تعیین می‌شود
_sqlite_lock = threading.RLock()  # برای ایمنی دسترسی همزمان در SQLite

def init_db() -> None:
    """
    تلاش می‌کند به PostgreSQL متصل شود؛ اگر شکست خورد، SQLite جایگزین می‌شود.
    """
    global POOL, USE_PG

    try:
        pg_url = os.getenv("POSTGRES_URL")
        if not pg_url:
            raise ValueError("POSTGRES_URL not set")

        POOL = SimpleConnectionPool(
            minconn=1,
            maxconn=5,
            dsn=pg_url,
            connect_timeout=10,
            sslmode="require",
        )

        with POOL.getconn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
        USE_PG = True
        logger.info("✅ Connected to PostgreSQL")
        _setup_schema_pg()

    except Exception as exc:
        logger.warning("⚠️ PostgreSQL unavailable (%s), falling back to SQLite.", exc)
        USE_PG = False
        _setup_schema_sqlite()


def _setup_schema_pg() -> None:
    """
    ایجاد جداول users و questions در PostgreSQL.
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS users (
        user_id          BIGINT PRIMARY KEY,
        username         TEXT,
        first_name       TEXT,
        last_name        TEXT,
        status           TEXT    DEFAULT 'pending',
        receipt_photo_id TEXT,
        expire_at        TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS questions (
        id         SERIAL PRIMARY KEY,
        user_id    BIGINT,
        question   TEXT,
        answer     TEXT,
        asked_at   TIMESTAMP DEFAULT NOW()
    );
    """
    with POOL.getconn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()


def _setup_schema_sqlite() -> None:
    """
    ایجاد جداول users و questions در SQLite.
    """
    SQLITE_FILE.touch(exist_ok=True)
    ddl = """
    CREATE TABLE IF NOT EXISTS users (
        user_id          INTEGER PRIMARY KEY,
        username         TEXT,
        first_name       TEXT,
        last_name        TEXT,
        status           TEXT    DEFAULT 'pending',
        receipt_photo_id TEXT,
        expire_at        TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS questions (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    INTEGER,
        question   TEXT,
        answer     TEXT,
        asked_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    with _sqlite_lock, sqlite3.connect(SQLITE_FILE) as conn:
        conn.executescript(ddl)
        conn.commit()


@contextmanager
def get_db() -> Generator[sqlite3.Connection | "psycopg2.extensions.connection", None, None]:
    """
    مدیریت اتصال پایگاه‌داده با پشتیبانی از PostgreSQL یا SQLite.
    """
    if USE_PG and POOL:
        conn = POOL.getconn()
        try:
            yield conn
        finally:
            POOL.putconn(conn)
    else:
        with _sqlite_lock, sqlite3.connect(SQLITE_FILE) as conn:
            conn.row_factory = sqlite3.Row
            yield conn

# ---------------------------------------------------------------------------#
# 2. Data helpers (users, receipts, questions)                               #
# ---------------------------------------------------------------------------#

# مقدار جایگزین پارامتر SQL در SQLite یا PostgreSQL
_PLACEHOLDER = "%s" if USE_PG else "?"

def _update_placeholder() -> None:
    global _PLACEHOLDER
    _PLACEHOLDER = "%s" if USE_PG else "?"


def _exec(sql: str, params: Tuple = ()) -> None:
    """
    اجرای دستورات INSERT / UPDATE / DELETE با پشتیبانی از PostgreSQL و SQLite.
    """
    with get_db() as conn:
        if USE_PG:
            with conn.cursor() as cur:
                cur.execute(sql, params)
        else:
            cur = conn.cursor()
            try:
                cur.execute(sql, params)
            finally:
                cur.close()
        conn.commit()


def _fetchone(sql: str, params: Tuple = ()) -> Optional[Tuple]:
    """
    اجرای SELECT و بازگرداندن تنها یک ردیف نتیجه.
    """
    with get_db() as conn:
        if USE_PG:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchone()
        else:
            cur = conn.cursor()
            try:
                cur.execute(sql, params)
                return cur.fetchone()
            finally:
                cur.close()


def upsert_user(user_id: int, username: Optional[str], first: Optional[str], last: Optional[str]) -> None:
    """
    درج یا به‌روزرسانی اطلاعات کاربر با استفاده از ON CONFLICT.
    """
    if USE_PG:
        sql = (
            f"INSERT INTO users (user_id, username, first_name, last_name) "
            f"VALUES ({_PLACEHOLDER},{_PLACEHOLDER},{_PLACEHOLDER},{_PLACEHOLDER}) "
            f"ON CONFLICT (user_id) DO UPDATE SET "
            f"username=EXCLUDED.username, first_name=EXCLUDED.first_name, last_name=EXCLUDED.last_name"
        )
    else:
        sql = (
            f"INSERT INTO users (user_id, username, first_name, last_name) "
            f"VALUES ({_PLACEHOLDER},{_PLACEHOLDER},{_PLACEHOLDER},{_PLACEHOLDER}) "
            f"ON CONFLICT(user_id) DO UPDATE SET "
            f"username=excluded.username, first_name=excluded.first_name, last_name=excluded.last_name"
        )
    _exec(sql, (user_id, username, first, last))


def save_receipt_request(user_id: int, photo_id: str) -> None:
    """
    ذخیره رسید و تعیین وضعیت کاربر به 'awaiting'.
    """
    sql = f"UPDATE users SET receipt_photo_id={_PLACEHOLDER}, status='awaiting' WHERE user_id={_PLACEHOLDER}"
    _exec(sql, (photo_id, user_id))


def set_user_status(user_id: int, status: str) -> None:
    """
    به‌روزرسانی وضعیت کاربر (pending, approved, rejected, awaiting).
    """
    sql = f"UPDATE users SET status={_PLACEHOLDER} WHERE user_id={_PLACEHOLDER}"
    _exec(sql, (status, user_id))


def save_subscription(user_id: int, days: int = 30) -> None:
    """
    فعال‌سازی اشتراک کاربر به مدت مشخص (پیش‌فرض: ۳۰ روز).
    """
    expire_at = datetime.utcnow() + timedelta(days=days)
    sql = f"UPDATE users SET expire_at={_PLACEHOLDER}, status='approved' WHERE user_id={_PLACEHOLDER}"
    _exec(sql, (expire_at, user_id))


def has_active_subscription(user_id: int) -> bool:
    """
    بررسی دارد یا ندارد اشتراک فعال.
    """
    row = _fetchone(
        f"SELECT expire_at FROM users WHERE user_id={_PLACEHOLDER} AND status='approved'",
        (user_id,),
    )
    if not row or not row[0]:
        return False

    expire_at = row[0]
    if isinstance(expire_at, str):
        expire_at = datetime.fromisoformat(expire_at)

    return expire_at >= datetime.utcnow()


def save_question(user_id: int, question: str, answer: str) -> None:
    """
    ذخیره سؤال و پاسخ برای اهداف آرشیو و تحلیل.
    """
    sql = f"INSERT INTO questions (user_id, question, answer) VALUES ({_PLACEHOLDER},{_PLACEHOLDER},{_PLACEHOLDER})"
    _exec(sql, (user_id, question, answer))

# ---------------------------------------------------------------------------#
# 3. OpenAI interface & long-message helper                                  #
# ---------------------------------------------------------------------------#

async def ask_openai(question: str, *, user_lang: str = "fa") -> str:
    """
    ارسال سؤال به مدل OpenAI و دریافت پاسخ.
    در صورت بروز خطا، پیام مناسب و کاربرپسند برمی‌گرداند.
    """
    system_msg = (
        "You are an experienced Iranian lawyer. Answer in formal Persian with citations to relevant statutes."
        if user_lang == "fa"
        else "You are an experienced international lawyer. Respond clearly and professionally in English."
    )

    try:
        response = await client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": question},
            ],
            temperature=0.6,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    except RateLimitError:
        return "❗️ظرفیت سرویس موقتاً پر است. لطفاً چند لحظه دیگر تلاش کنید."
    except AuthenticationError:
        return "❌ کلید API نامعتبر است. لطفاً مدیر ربات را مطلع کنید."
    except APIError as err:
        logger.error("OpenAI API error: %s", err)
        return f"⚠️ خطای ارتباط با OpenAI: {err}"


def _split_message(text: str, limit: int = 4096) -> List[str]:
    """
    شکستن پیام‌های بیش‌ازحد طولانی به قطعات قابل‌ارسال در تلگرام.
    """
    if len(text) <= limit:
        return [text]

    parts: List[str] = []
    while len(text) > limit:
        split_at = max(text.rfind(sep, 0, limit) for sep in ("\n\n", "\n", " "))
        split_at = split_at if split_at != -1 else limit
        parts.append(text[:split_at].rstrip())
        text = text[split_at:].lstrip()
    if text:
        parts.append(text)
    return parts


async def send_long(update: Update, text: str, *, parse_mode: Optional[str] = ParseMode.HTML) -> None:
    """
    ارسال پیام طولانی به صورت قطعه‌قطعه در چند پیام پشت‌سرهم.
    """
    for chunk in _split_message(text):
        await update.message.reply_text(chunk, parse_mode=parse_mode)


async def answer_question(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    question: str,
    lang: str = "fa",
) -> None:
    """
    ارسال سؤال به OpenAI، ذخیره در پایگاه‌داده و ارسال پاسخ به کاربر.
    """
    uid = update.effective_user.id
    await update.message.chat.send_action(ChatAction.TYPING)

    answer = await ask_openai(question, user_lang=lang)
    save_question(uid, question, answer)
    await send_long(update, answer)

# ---------------------------------------------------------------------------#
# 4. Receipt flow – user → admin review → subscription grant                 #
# ---------------------------------------------------------------------------#

ADMIN_ID = int(getenv_or_die("ADMIN_ID"))
SUBS_DAYS = int(os.getenv("SUBSCRIPTION_DAYS", "30"))


async def handle_receipt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    گرفتن رسید (عکس یا متن) از کاربر، ثبت وضعیت، ارسال به مدیر با دکمه تأیید/رد.
    """
    msg: Message = update.message
    uid = update.effective_user.id

    # فقط وقتی منتظر رسید هستیم یا پیام عکس دارد ادامه بده
    if not context.user_data.get("awaiting_receipt") and not msg.photo:
        return

    # پاک‌کردن فلگ حالت دریافت رسید
    context.user_data["awaiting_receipt"] = False

    # ثبت یا به‌روزرسانی اطلاعات کاربر
    upsert_user(uid, msg.from_user.username, msg.from_user.first_name, msg.from_user.last_name)

    # ذخیرهٔ رسید در DB
    photo_id = msg.photo[-1].file_id if msg.photo else None
    save_receipt_request(uid, photo_id or msg.text or "")

    # ساخت دکمه‌های اینلاین برای مدیر
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ تأیید", callback_data=f"approve:{uid}"),
            InlineKeyboardButton("❌ رد", callback_data=f"reject:{uid}")
        ]
    ])

    caption_head = (
        f"📄 رسید جدید از <a href='tg://user?id={uid}'>{uid}</a>\n"
        f"👤 نام: {msg.from_user.full_name}\n"
        f"🕒 زمان: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"📥 برای بررسی:"
    )

    if photo_id:
        await context.bot.send_photo(
            chat_id=ADMIN_ID,
            photo=photo_id,
            caption=caption_head,
            reply_markup=keyboard,
            parse_mode=ParseMode.HTML,
        )
    else:
        await context.bot.send_message(
            chat_id=ADMIN_ID,
            text=f"{caption_head}\n\n{msg.text or 'رسید متنی'}",
            reply_markup=keyboard,
            parse_mode=ParseMode.HTML,
        )

    await msg.reply_text("✅ رسید شما برای بررسی ارسال شد. لطفاً منتظر تأیید مدیر بمانید.")


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    پردازش کلیک روی دکمه‌های تأیید یا رد رسید.
    فقط مدیر مجاز است عملیات انجام دهد.
    """
    query = update.callback_query
    await query.answer()

    # استخراج نوع عملیات و ID کاربر هدف
    try:
        action, uid_str = query.data.split(":")
        target_uid = int(uid_str)
    except Exception as err:
        logger.error("Invalid callback_data format: %s", err)
        return

    # محدودیت دسترسی
    if update.effective_user.id != ADMIN_ID:
        await query.answer("⛔️ فقط مدیر اجازه دارد این کار را انجام دهد.", show_alert=True)
        return

    # انجام عملیات بر اساس تأیید یا رد
    if action == "approve":
        save_subscription(target_uid, days=SUBS_DAYS)
        await context.bot.send_message(
            chat_id=target_uid,
            text=f"🎉 اشتراک شما تأیید شد و تا {SUBS_DAYS} روز فعال است. اکنون می‌توانید سؤال حقوقی بپرسید.",
        )
        status_note = "✔️ تأیید شد"
    elif action == "reject":
        set_user_status(target_uid, "rejected")
        await context.bot.send_message(
            chat_id=target_uid,
            text="❌ رسید شما رد شد. لطفاً دوباره با رسید معتبر اقدام کنید.",
        )
        status_note = "❌ رد شد"
    else:
        status_note = "⛔️ عملیات ناشناخته"

    # بروزرسانی پیام مدیر
    message = query.message
    new_text = (message.caption or message.text or "") + f"\n\n<b>وضعیت: {status_note}</b>"

    try:
        if message.photo:
            await message.edit_caption(new_text, parse_mode=ParseMode.HTML, reply_markup=None)
        else:
            await message.edit_text(new_text, parse_mode=ParseMode.HTML, reply_markup=None)
    except Exception as e:
        logger.error("Failed to edit admin message: %s", e)

# ---------------------------------------------------------------------------#
# 5. Command handlers & menu router                                          #
# ---------------------------------------------------------------------------#

TON_WALLET_ADDR = getenv_or_die("TON_WALLET_ADDRESS")
BANK_CARD = getenv_or_die("BANK_CARD_NUMBER")

BUY_TEXT_FA = (
    "🛒 <b>راهنمای خرید اشتراک</b>\n\n"
    "۱️⃣ پرداخت 1 TON به آدرس کیف‌پول زیر:\n"
    f"<code>{TON_WALLET_ADDR}</code>\n\n"
    "۲️⃣ یا واریز ۵۰۰٬۰۰۰ تومان به شماره کارت زیر:\n"
    f"<code>{BANK_CARD}</code>\n\n"
    "پس از پرداخت، از دکمه «📤 ارسال رسید» استفاده کنید."
)
BUY_TEXT_EN = (
    "🛒 <b>Subscription Purchase Guide</b>\n\n"
    "1️⃣ Pay 1 TON to the wallet address below:\n"
    f"<code>{TON_WALLET_ADDR}</code>\n\n"
    "2️⃣ Or deposit 500,000 IRR to the following bank card:\n"
    f"<code>{BANK_CARD}</code>\n\n"
    "After payment, use the '📤 Send Receipt' button."
)

WELCOME_FA = (
    "سلام! 👋\n"
    "من <b>ربات حقوقی RebLawBot</b> هستم.\n\n"
    "با تهیه اشتراک می‌توانید سؤالات حقوقی خود را بپرسید.\n"
    "برای شروع یکی از گزینه‌های زیر را انتخاب کنید:"
)
WELCOME_EN = (
    "Hello! 👋\n"
    "I am <b>RebLawBot</b>, your legal assistant.\n\n"
    "Purchase a subscription to ask legal questions.\n"
    "Please choose an option from the menu:"
)

MSG_NO_SUB_FA = "❌ اشتراک فعالی ندارید. لطفاً ابتدا اشتراک خریداری کنید."
MSG_NO_SUB_EN = "❌ You do not have an active subscription. Please purchase a subscription first."

ASK_PROMPT_FA = "سؤال خود را بعد از /ask بفرستید.\nمثال:\n<code>/ask قانون کار چیست؟</code>"
ASK_PROMPT_EN = "Send your question after /ask.\nExample:\n<code>/ask What is labor law?</code>"

MENU_KB = ReplyKeyboardMarkup(
    [
        [KeyboardButton("🛒 خرید اشتراک"), KeyboardButton("📤 ارسال رسید")],
        [KeyboardButton("⚖️ سؤال حقوقی"), KeyboardButton("ℹ️ درباره توکن")],
    ],
    resize_keyboard=True,
)

# ─── فرمان /start ──────────────────────────────────────────────────────────
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lang = "fa" if update.effective_user.language_code and update.effective_user.language_code.startswith("fa") else "en"
    text = WELCOME_FA if lang == "fa" else WELCOME_EN
    await update.message.reply_text(text, reply_markup=MENU_KB, parse_mode=ParseMode.HTML)

# ─── فرمان /buy ────────────────────────────────────────────────────────────
async def buy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lang = "fa" if update.effective_user.language_code and update.effective_user.language_code.startswith("fa") else "en"
    text = BUY_TEXT_FA if lang == "fa" else BUY_TEXT_EN
    await update.message.reply_text(text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

# ─── فرمان /send_receipt ───────────────────────────────────────────────────
async def send_receipt_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["awaiting_receipt"] = True
    await update.message.reply_text("لطفاً تصویر یا متن رسید پرداخت را ارسال کنید.")

# ─── فرمان /status ─────────────────────────────────────────────────────────
async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    lang = "fa" if update.effective_user.language_code and update.effective_user.language_code.startswith("fa") else "en"

    if has_active_subscription(uid):
        row = _fetchone("SELECT expire_at FROM users WHERE user_id=" + _PLACEHOLDER, (uid,))
        expire_at = row[0]
        if isinstance(expire_at, str):
            expire_at = datetime.fromisoformat(expire_at)
        msg = f"✅ اشتراک شما تا <b>{expire_at:%Y-%m-%d}</b> فعال است." if lang == "fa" else f"✅ Your subscription is active until <b>{expire_at:%Y-%m-%d}</b>."
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text(MSG_NO_SUB_FA if lang == "fa" else MSG_NO_SUB_EN)

# ─── فرمان /ask ────────────────────────────────────────────────────────────
async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    lang = "fa" if update.effective_user.language_code and update.effective_user.language_code.startswith("fa") else "en"

    if not has_active_subscription(uid):
        await update.message.reply_text(MSG_NO_SUB_FA if lang == "fa" else MSG_NO_SUB_EN)
        return

    question = " ".join(context.args)
    if not question:
        await update.message.reply_text(ASK_PROMPT_FA if lang == "fa" else ASK_PROMPT_EN)
        return

    await answer_question(update, context, question, lang=lang)

# ─── مسیریابی پیام‌های منو ────────────────────────────────────────────────
async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()
    lang = "fa" if update.effective_user.language_code and update.effective_user.language_code.startswith("fa") else "en"

    if text.startswith("/"):
        return  # فرمان‌ها جداگانه هندل می‌شوند

    if (lang == "fa" and text == "🛒 خرید اشتراک") or (lang == "en" and text.lower() == "buy subscription"):
        await buy_cmd(update, context)
    elif (lang == "fa" and text == "📤 ارسال رسید") or (lang == "en" and text.lower() == "send receipt"):
        await send_receipt_cmd(update, context)
    elif (lang == "fa" and text == "⚖️ سؤال حقوقی") or (lang == "en" and text.lower() == "legal question"):
        await update.message.reply_text(ASK_PROMPT_FA if lang == "fa" else ASK_PROMPT_EN, parse_mode=ParseMode.HTML)
    elif (lang == "fa" and text == "ℹ️ درباره توکن") or (lang == "en" and text.lower() == "about token"):
        await about_token(update, context)
    else:
        await update.message.reply_text("دستور نامعتبر است. از منو استفاده کنید." if lang == "fa" else "Invalid command. Please use the menu.")

TOKEN_IMG = Path(__file__).with_name("reblawcoin.png")  # تصویر لوگوی RLC

async def about_token(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    ارسال معرفی توکن RebLawCoin (RLC) به همراه تصویر و لینک خرید.
    به زبان فارسی یا انگلیسی بر اساس زبان کاربر.
    """
    msg = update.effective_message
    lang = "fa" if update.effective_user.language_code and update.effective_user.language_code.startswith("fa") else "en"

    if TOKEN_IMG.exists():
        await msg.reply_photo(TOKEN_IMG.open("rb"))

    text_fa = (
        "🎉 <b>توکن RebLawCoin (RLC)</b> – اولین ارز دیجیتال با محوریت خدمات حقوقی.\n\n"
        "<b>اهداف پروژه:</b>\n"
        "• سرمایه‌گذاری در نوآوری‌های حقوقی\n"
        "• نهادینه‌سازی عدالت روی بلاک‌چین\n"
        "• سودآوری پایدار برای سرمایه‌گذاران\n\n"
        "🔗 برای خرید سریع روی لینک زیر بزنید:\n"
        "<a href='https://t.me/blum/app?startapp=memepadjetton_RLC_JpMH5-ref_1wgcKkl94N'>خرید از Blum</a>"
    )
    text_en = (
        "🎉 <b>RebLawCoin (RLC) Token</b> – The first cryptocurrency focused on legal services.\n\n"
        "<b>Project Goals:</b>\n"
        "• Investing in legal innovations\n"
        "• Institutionalizing justice on blockchain\n"
        "• Sustainable profit for investors\n\n"
        "🔗 Click the link below for quick purchase:\n"
        "<a href='https://t.me/blum/app?startapp=memepadjetton_RLC_JpMH5-ref_1wgcKkl94N'>Buy on Blum</a>"
    )

    await msg.reply_text(
        text_fa if lang == "fa" else text_en,
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )

# ─── ثبت تمام هندلرها ───────────────────────────────────────────────────────
def register_handlers(app: Application) -> None:
    # دستورات اصلی
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("buy", buy_cmd))
    app.add_handler(CommandHandler("send_receipt", send_receipt_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(CommandHandler("about_token", about_token))

    # دکمه‌های تأیید/رد رسید (گروه 0 = اولویت بالا)
    app.add_handler(
        CallbackQueryHandler(callback_handler, pattern=r"^(approve|reject):\d+$"),
        group=0,
    )

    # رسید (عکس یا متن) – گروه 1
    app.add_handler(
        MessageHandler(filters.PHOTO | (filters.TEXT & ~filters.COMMAND), handle_receipt),
        group=1,
    )

    # سایر پیام‌های متنی منو – گروه 2
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, text_router),
        group=2,
    )

# ─── نقطهٔ ورود اصلی ────────────────────────────────────────────────────────
def main() -> None:
    bot_token = getenv_or_die("BOT_TOKEN")
    init_db()
    application = Application.builder().token(bot_token).build()
    register_handlers(application)

    if os.getenv("USE_WEBHOOK", "false").lower() == "true":
        domain = getenv_or_die("WEBHOOK_DOMAIN")
        application.run_webhook(
            listen="0.0.0.0",
            port=int(os.getenv("PORT", "8443")),
            url_path=bot_token,
            webhook_url=f"{domain}/{bot_token}",
            allowed_updates=Update.ALL_TYPES,
        )
    else:
        application.run_polling(allowed_updates=Update.ALL_TYPES)

