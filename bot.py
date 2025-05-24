#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RebLawBot – Telegram bot that sells subscriptions and answers legal questions using OpenAI.
Version 2025-05-13 (compat OpenAI 1.x)
"""

from __future__ import annotations

# ─── استاندارد کتابخانه ───────────────────────────────────────────────────────
import asyncio
import logging
import os
import sqlite3
import whisper
import tempfile
import ffmpeg
from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Generator, List, Optional, Tuple
from telegram import ReplyKeyboardMarkup, KeyboardButton
from texts import TEXTS
# ─── کتابخانه‌های خارجی ───────────────────────────────────────────────────────
from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError, RateLimitError, AuthenticationError
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
from telegram.ext import MessageHandler, filters
# ─── محیط و تنظیمات جهانی ─────────────────────────────────────────────────────
load_dotenv()  # متغیرهای محیطی را از .env می‌خواند

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger("RebLawBot")

# کلاینت غیرهمزمان OpenAI؛ تمام فراخوانی‌ها از همین نمونه استفاده می‌کنند
client = AsyncOpenAI()
# ---------------------------------------------------------------------------#
# 0. Utilities                                                               #
# ---------------------------------------------------------------------------#
# بارگذاری مدل فقط یک‌بار در ابتدای اجرا
whisper_model = whisper.load_model("base")

def voice_to_text(file_path: str) -> str:
    """تبدیل فایل صوتی به متن با استفاده از Whisper"""
    result = whisper_model.transcribe(file_path)
    return result["text"]

def get_main_menu(lang: str):
    menus = {
        "fa": [
            [KeyboardButton("🛒 خرید اشتراک"), KeyboardButton("📤 ارسال رسید")],
            [KeyboardButton("⚖️ سؤال حقوقی"), KeyboardButton("🎤 سؤال صوتی")],
            [KeyboardButton("ℹ️ درباره توکن"), KeyboardButton("/lang")]
        ],
        "en": [
            [KeyboardButton("🛒 Buy Subscription"), KeyboardButton("📤 Send Receipt")],
            [KeyboardButton("⚖️ Legal Question"), KeyboardButton("🎤 Voice Question")],
            [KeyboardButton("ℹ️ About Token"), KeyboardButton("/lang")]
        ],
        "ku": [
            [KeyboardButton("🛒 کڕینی بەشداریکردن"), KeyboardButton("📤 ناردنی پسوڵە")],
            [KeyboardButton("⚖️ پرسیاری یاسایی"), KeyboardButton("🎤 پرسیاری دەنگی")],
            [KeyboardButton("ℹ️ دەربارەی تۆکێن"), KeyboardButton("/lang")]
        ]
    }
    return ReplyKeyboardMarkup(menus.get(lang, menus["fa"]), resize_keyboard=True)



def tr(key: str, lang: str = "fa", **kwargs) -> str:
    """دریافت متن ترجمه‌شده بر اساس کلید و زبان کاربر"""
    base = TEXTS.get(key, {}).get(lang) or TEXTS.get(key, {}).get("fa") or ""
    return base.format(**kwargs)

def getenv_or_die(key: str) -> str:
    """
    برمی‌گرداند مقدار متغیر محیطی *key*؛
    اگر وجود نداشته باشد، خطای RuntimeError می‌دهد.

    برای متغیرهای ضروری مانند BOT_TOKEN، POSTGRES_URL یا OPENAI_API_KEY
    از این تابع استفاده کنید تا در صورت پیکربندی ناقص،
    ربات به‌صراحت اخطار دهد و متوقف شود.
    """
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Environment variable {key!r} is missing")
    return value

def get_lang(context):
    lang = context.user_data.get("lang")
    if lang not in ("fa", "en", "ku"):
        lang = "fa"
        context.user_data["lang"] = lang
    return lang


# ---------------------------------------------------------------------------#
# 1. Database layer – PostgreSQL → SQLite fallback                           #
# ---------------------------------------------------------------------------#
import threading

# فایل لوکال SQLite (اگر PostgreSQL در دسترس نبود)
SQLITE_FILE = Path("users.db")
POOL: Optional[SimpleConnectionPool] = None
USE_PG = False                        # پس از init_db مشخص می‌شود
_sqlite_lock = threading.RLock()      # برای ایمنی رشته‌ای روی SQLite

def init_db() -> None:
    """
    تلاش می‌کند به PostgreSQL متصل شود؛ در صورت شکست،
    SQLite را به‌عنوان جایگزین برمی‌گزیند. باید یک بار در startup اجرا شود.
    """
    global POOL, USE_PG

    try:
        pg_url = os.getenv("POSTGRES_URL")  # شکل کامل: postgres://user:pass@host:port/db
        if not pg_url:
            raise ValueError("POSTGRES_URL not set")

        POOL = SimpleConnectionPool(
            minconn=1,
            maxconn=5,
            dsn=pg_url,
            connect_timeout=10,
            sslmode="require",
        )
        # تست سادهٔ اتصال
        with POOL.getconn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
        USE_PG = True
        logger.info("✅ Connected to PostgreSQL")
        _setup_schema_pg()

    except Exception as exc:
        logger.warning("PostgreSQL unavailable (%s), switching to SQLite.", exc)
        USE_PG = False
        _setup_schema_sqlite()

def _setup_schema_pg() -> None:
    """ایجاد جدول بر روی PostgreSQL (اگر وجود نداشته باشد)."""
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
    """ایجاد جدول بر روی SQLite (اگر وجود نداشته باشد)."""
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
    کانتکست‌منیجر واحد برای دریافت اتصال به پایگاه‌داده.
    روی PostgreSQL، اتصال را از POOL می‌گیرد؛ روی SQLite، یک اتصال
    جدید با قفل سراسری می‌سازد.
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
# مقدار جایگزین پارامتر در SQL (بعد از init_db دوباره ست می‌شود)
_PLACEHOLDER = "%s" if USE_PG else "?"

def _update_placeholder() -> None:
    """پس از init_db فراخوانی می‌شود تا مقدار صحیح برای PG/SQLite را ست کند."""
    global _PLACEHOLDER
    _PLACEHOLDER = "%s" if USE_PG else "?"

# ─────────────────────────────────────────────────────────────────────────────
def _exec(sql: str, params: Tuple = ()) -> None:
    "اجرای INSERT/UPDATE/DELETE در هر دو پایگاه‌داده."
    with get_db() as conn:
        if USE_PG:
            with conn.cursor() as cur:
                cur.execute(sql, params)
        else:                           # sqlite.cursor کلید contextmanager ندارد
            cur = conn.cursor()
            try:
                cur.execute(sql, params)
            finally:
                cur.close()
        conn.commit()

def _fetchone(sql: str, params: Tuple = ()):
    "اجرای SELECT و برگرداندن یک سطر."
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

# ─── تقسیم پیام بلند به قطعات کوچکتر ─────────────────────────────────────────
def _split_message(text: str, limit: int = 4096) -> List[str]:
    """
    متن بیش‌ازحد بلند را روی \n\n یا \n یا فاصله می‌شکند تا تلگرام خطا ندهد.
    """
    if len(text) <= limit:
        return [text]

    parts: List[str] = []
    while len(text) > limit:
        breakpoints = [text.rfind(sep, 0, limit) for sep in ("\n\n", "\n", " ")]
        idx = max(breakpoints)
        idx = idx if idx != -1 else limit
        parts.append(text[:idx].rstrip())
        text = text[idx:].lstrip()
    if text:
        parts.append(text)
    return parts

# ─────────────────────────────────────────────────────────────────────────────
def upsert_user(user_id: int, username: str | None,
                first: str | None, last: str | None) -> None:
    """
    درج یا به‌روزرسانی پروفایل کاربر. وضعیت اولیه 'pending' است.
    """
    sql = (
        f"INSERT INTO users (user_id, username, first_name, last_name) "
        f"VALUES ({_PLACEHOLDER},{_PLACEHOLDER},{_PLACEHOLDER},{_PLACEHOLDER}) "
        f"ON CONFLICT (user_id) DO UPDATE SET "
        f"username=EXCLUDED.username, first_name=EXCLUDED.first_name, last_name=EXCLUDED.last_name"
        if USE_PG else
        f"INSERT INTO users (user_id, username, first_name, last_name) "
        f"VALUES ({_PLACEHOLDER},{_PLACEHOLDER},{_PLACEHOLDER},{_PLACEHOLDER}) "
        f"ON CONFLICT(user_id) DO UPDATE SET "
        f"username=excluded.username, first_name=excluded.first_name, last_name=excluded.last_name"
    )
    _exec(sql, (user_id, username, first, last))

# ─────────────────────────────────────────────────────────────────────────────
def save_receipt_request(user_id: int, photo_id: str) -> None:
    "ذخیرهٔ شناسهٔ فایل رسید و تغییر وضعیت کاربر به 'awaiting'."
    sql = (
        f"UPDATE users SET receipt_photo_id={_PLACEHOLDER}, status='awaiting' "
        f"WHERE user_id={_PLACEHOLDER}"
    )
    _exec(sql, (photo_id, user_id))

def set_user_status(user_id: int, status: str) -> None:
    "به‌روزرسانی ستون status (pending / approved / rejected / awaiting)."
    _exec(
        f"UPDATE users SET status={_PLACEHOLDER} WHERE user_id={_PLACEHOLDER}",
        (status, user_id),
    )

# ─────────────────────────────────────────────────────────────────────────────
def save_subscription(user_id: int, days: int = 30) -> None:
    """
    هنگام تأیید رسید، تاریخ انقضا را به 'امروز + days' ست می‌کند
    و وضعیت کاربر را 'approved' می‌گذارد.
    """
    expire_at = datetime.utcnow() + timedelta(days=days)
    sql = (
        f"UPDATE users SET expire_at={_PLACEHOLDER}, status='approved' "
        f"WHERE user_id={_PLACEHOLDER}"
    )
    _exec(sql, (expire_at, user_id))

def has_active_subscription(user_id: int) -> bool:
    """
    بازمی‌گرداند کاربر اشتراک معتبر دارد یا خیر.
    """
    row = _fetchone(
        f"SELECT expire_at FROM users WHERE user_id={_PLACEHOLDER} AND status='approved'",
        (user_id,),
    )
    if not row or row[0] is None:
        return False
    expire_at = row[0]  # datetime در PG، str در SQLite (تبدیل ↓)
    if isinstance(expire_at, str):
        expire_at = datetime.fromisoformat(expire_at)
    return expire_at >= datetime.utcnow()

# ─────────────────────────────────────────────────────────────────────────────
def save_question(user_id: int, question: str, answer: str) -> None:
    "ذخیرهٔ سؤال و جواب برای لاگ و تحلیل‌های بعدی."
    sql = (
        f"INSERT INTO questions (user_id, question, answer) "
        f"VALUES ({_PLACEHOLDER},{_PLACEHOLDER},{_PLACEHOLDER})"
    )
    _exec(sql, (user_id, question, answer))
# ---------------------------------------------------------------------------#
# 3. OpenAI interface & long-message helper                                  #
# ---------------------------------------------------------------------------#
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lang = get_lang(context)
    welcome_text = {
        "fa": "سلام! 👋\nمن <b>ربات حقوقی RebLawBot</b> هستم.\nبا تهیه اشتراک می‌توانید سؤالات حقوقی خود را بپرسید.\nبرای شروع یکی از گزینه‌های زیر را انتخاب کنید:",
        "en": "Hello! 👋\nI am <b>RebLawBot</b>, your legal assistant.\nPurchase a subscription to ask legal questions.\nPlease choose an option from the menu:",
        "ku": "سڵاو! 👋\nمن <b>ڕۆبۆتی یاسایی RebLawBot</b>م.\nبە بەشداربوون دەتوانیت پرسیاری یاساییت بکەیت.\nتکایە هەڵبژاردنێک بکە لە خوارەوە:"
    }
    await update.message.reply_text(
        welcome_text.get(lang, welcome_text["fa"]),
        reply_markup=get_main_menu(lang),
        parse_mode=ParseMode.HTML
    )

async def ask_openai(question: str, *, user_lang: str = "fa") -> str:
    """
    ارسال سؤال به GPT و برگرداندن پاسخ متنی.
    در صورت بروز خطا، پیام کاربرپسند برمی‌گرداند.
    """
    system_msg = (
        "You are an experienced Iranian lawyer. Answer in formal Persian with citations to relevant statutes where possible."
        if user_lang == "fa"
        else "You are an experienced lawyer. Answer in clear English."
    )

    try:
        rsp = await client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": question},
            ],
            temperature=0.6,
            max_tokens=1024,
        )
        return rsp.choices[0].message.content.strip()

    except RateLimitError:
        return "❗️ظرفیت سرویس موقتاً پر است؛ چند ثانیهٔ دیگر تلاش کنید."
    except AuthenticationError:
        return "❌ کلید OpenAI نامعتبر است؛ لطفاً مدیر را مطلع کنید."
    except APIError as exc:
        logger.error("OpenAI API error: %s", exc)
        return f"⚠️ خطای سرویس OpenAI: {exc}"

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    lang = get_lang(context)

    if not has_active_subscription(uid):
        await update.message.reply_text(tr("no_sub", lang))
        return

    voice_file = await update.message.voice.get_file()

    # دانلود فایل در یک فایل موقت
    with tempfile.NamedTemporaryFile(suffix=".ogg") as temp_audio:
        await voice_file.download_to_drive(temp_audio.name)

        await update.message.reply_text({
            "fa": "🎤 در حال پردازش صدای شما...",
            "en": "🎤 Processing your voice message...",
            "ku": "🎤 پەیامی دەنگیت هەڵسەنگاندنە..."
        }.get(lang, "در حال پردازش صوت..."))

        # تبدیل صوت به متن
        try:
            question_text = voice_to_text(temp_audio.name)
        except Exception as e:
            logger.error("Voice processing error: %s", e)
            await update.message.reply_text("❌ خطا در تبدیل صوت به متن.")
            return

    # ارسال به OpenAI و دریافت پاسخ
    await answer_question(update, context, question_text, lang)


async def send_long(update: Update, text: str, *, parse_mode: str | None = ParseMode.HTML) -> None:
    """ارسال امن پیام‌های طولانی در چند بخش پیاپی."""
    for chunk in _split_message(text):
        await update.message.reply_text(chunk, parse_mode=parse_mode)


async def answer_question(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    question: str,
    lang: str = "fa",
) -> None:
    """
    پاسخ به سؤال حقوقی، ذخیره در DB و ارسال در چند قطعه (در صورت نیاز).
    """
    uid = update.effective_user.id
    await update.message.chat.send_action(ChatAction.TYPING)

    answer = await ask_openai(question, user_lang=lang)
    save_question(uid, question, answer)
    await send_long(update, answer)
# ---------------------------------------------------------------------------#
# 4. Receipt flow – user → admin review → subscription grant                 #
# ---------------------------------------------------------------------------#
ADMIN_ID = int(getenv_or_die("ADMIN_ID"))          # آی‌دی تِلگرامی مدیر
SUBS_DAYS = int(os.getenv("SUBSCRIPTION_DAYS", "30"))   # طول پیش‌فرض اشتراک

async def handle_receipt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # فقط وقتی منتظر رسید هستیم یا عکس دریافت شده ادامه دهیم
    if not context.user_data.get("awaiting_receipt") and not update.message.photo:
        return  # این پیام رسید نیست؛ بگذار هندلر بعدی پردازش کند

    # پس از پذیرش، فلگ را پاک می‌کنیم
    context.user_data["awaiting_receipt"] = False

    """
    رسید عکس یا متنی را از کاربر می‌گیرد، با دکمهٔ تأیید/رد برای مدیر می‌فرستد
    و وضعیت کاربر را 'awaiting' می‌گذارد.
    """
    msg: Message = update.message
    uid = update.effective_user.id

    # پروفایل کاربر را درج/به‌روزرسانی کنیم
    upsert_user(
        uid,
        msg.from_user.username,
        msg.from_user.first_name,
        msg.from_user.last_name,
    )

    # ذخیرهٔ درخواست رسید
    photo_id: Optional[str] = None
    if msg.photo:
        photo_id = msg.photo[-1].file_id
    save_receipt_request(uid, photo_id or "")

    # ساخت دکمه‌های اینلاین
    kb = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✅ تأیید", callback_data=f"approve:{uid}"),
                InlineKeyboardButton("❌ رد", callback_data=f"reject:{uid}"),
            ]
        ]
    )

    # ارسال به مدیر
    caption_head = (
        f"📄 رسید جدید از <a href='tg://user?id={uid}'>{uid}</a>\n"
        f"نام: {msg.from_user.full_name}\n"
        "برای بررسی:"
    )
    if photo_id:
        await context.bot.send_photo(
            ADMIN_ID,
            photo_id,
            caption=caption_head,
            reply_markup=kb,
            parse_mode=ParseMode.HTML,
        )
    else:
        text = msg.text or "رسید متنی"
        await context.bot.send_message(
            ADMIN_ID,
            f"{caption_head}\n\n{text}",
            reply_markup=kb,
            parse_mode=ParseMode.HTML,
        )

    await msg.reply_text("✅ رسید شما برای بررسی ارسال شد؛ لطفاً منتظر تأیید مدیر بمانید.")


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    هندلر دکمه‌های اینلاین «تأیید/رد» در پیام مدیر.
    پس از کلیک، وضعیت پایگاه‌داده را به‌روز کرده و به کاربر اطلاع می‌دهد.
    """
    query = update.callback_query
    await query.answer()

    try:
        action, uid_str = query.data.split(":")
        target_uid = int(uid_str)
    except (ValueError, AttributeError):
        return  # دادهٔ نامعتبر

    # محدودیت: فقط مدیر مجاز است
    if update.effective_user.id != ADMIN_ID:
        await query.answer("دسترسی شما محدود است.", show_alert=True)
        return

    if action == "approve":
        save_subscription(target_uid, days=SUBS_DAYS)
        await context.bot.send_message(
            target_uid,
            f"🎉 اشتراک شما تأیید شد و تا {SUBS_DAYS} روز فعال است. اکنون می‌توانید سؤال حقوقی بپرسید.",
        )
        status_note = "✔️ تأیید شد"
    else:  # reject
        set_user_status(target_uid, "rejected")
        await context.bot.send_message(
            target_uid,
            "❌ رسید شما رد شد. لطفاً دوباره با رسید صحیح اقدام کنید.",
        )
        status_note = "❌ رد شد"

    # ویرایش پیام مدیر برای ثبت نتیجه
    new_text = (query.message.caption or query.message.text) + f"\n\nوضعیت: {status_note}"
    if query.message.photo:
        await query.message.edit_caption(new_text, parse_mode=ParseMode.HTML)
    else:
        await query.message.edit_text(new_text, parse_mode=ParseMode.HTML)
# ---------------------------------------------------------------------------#
# 5. Command handlers & menu router                                          #
# ---------------------------------------------------------------------------#

# ─── متن‌های ثابت (FA/EN) ───────────────────────────────────────────────────
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

# جایگزینی تابع

MENU_KB = "کیبورد منو"

def register_handlers(app):
        app.add_handler(CommandHandler("buy", buy_cmd))
        app.add_handler(CommandHandler("start", start_cmd))

        # --- انتخاب زبان (Language Keyboard) ---
LANG_KB = ReplyKeyboardMarkup(
    [
        [KeyboardButton("فارسی"), KeyboardButton("English"), KeyboardButton("کوردی")],
    ],
    resize_keyboard=True,
    one_time_keyboard=True,
)

# ─── فرمان‌ها ────────────────────────────────────────────────────────────────
MENU_KB = ReplyKeyboardMarkup(
    [
        [KeyboardButton("🛒 خرید اشتراک"), KeyboardButton("📤 ارسال رسید")],
        [KeyboardButton("⚖️ سؤال حقوقی"), KeyboardButton("ℹ️ درباره توکن")],
        [KeyboardButton("/lang")],  # این خط را اضافه کنید
    ],
    resize_keyboard=True,
)

TON_WALLET_ADDR = getenv_or_die("TON_WALLET_ADDRESS")
BANK_CARD = getenv_or_die("BANK_CARD_NUMBER")

async def buy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lang = get_lang(context)
    ton_wallet = getenv_or_die("TON_WALLET_ADDRESS")
    bank_card = getenv_or_die("BANK_CARD_NUMBER")
    rlc_wallet = os.getenv("RLC_WALLET_ADDRESS", "آدرس تنظیم نشده")

    price_text = {
        "fa": (
            f"🔸 قیمت اشتراک یک‌ماهه:\n\n"
            f"💳 کارت بانکی: 700،000 تومان\n"
            f"🏦 شماره کارت: <code>{bank_card}</code>\n\n"
            f"💎 تون کوین (TON): 1 \n"
            f"👛 آدرس کیف پول: <code>{ton_wallet}</code>\n\n"
            f"🚀 توکن RLC: 1,800,000\n"
            f"🔗 آدرس والت RLC: <code>{rlc_wallet}</code>\n"
        ),
        "en": (
            f"🔸 One-month subscription price:\n\n"
            f"💳 Bank Card: 700،000 IRR\n"
            f"🏦 Card Number: <code>{bank_card}</code>\n\n"
            f"💎 TON Coin (TON): 1 \n"
            f"👛 Wallet Address: <code>{ton_wallet}</code>\n\n"
            f"🚀 RLC Token: 1,800,000\n"
            f"🔗 RLC Wallet Address: <code>{rlc_wallet}</code>\n"
        ),
        "ku": (
            f"🔸 نرخی بەشداریکردنی مانگانە:\n\n"
            f"💳 کارتی بانک: 700،000 تومان\n"
            f"🏦 ژمارەی کارت: <code>{bank_card}</code>\n\n"
            f"💎 تۆن کۆین (TON): 1 \n"
            f"👛 ناونیشانی جزدان: <code>{ton_wallet}</code>\n\n"
            f"🚀 تۆکێنی RLC: ١٬٨٠٠٬٠٠٠\n"
            f"🔗 ناونیشانی والت RLC: <code>{rlc_wallet}</code>\n"
        ),
    }

    await update.message.reply_text(
        price_text.get(lang, price_text["fa"]),
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )


async def lang_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """نمایش کیبورد انتخاب زبان هنگام اجرای /lang"""
    await update.message.reply_text(
        "لطفاً زبان مورد نظر را انتخاب کنید:\nPlease select your preferred language:\nتکایە زمانت هەلبژێرە:",
        reply_markup=LANG_KB,
    )


# دکمه یا فرمان «📤 ارسال رسید»؛ کاربر باید بلافاصله عکس یا متن ارسال کند
async def send_receipt_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lang = get_lang(context)
    context.user_data["awaiting_receipt"] = True
    await update.message.reply_text(tr("send_receipt_prompt", lang))

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    lang = get_lang(context)

    if has_active_subscription(uid):
        row = _fetchone("SELECT expire_at FROM users WHERE user_id=" + _PLACEHOLDER, (uid,))
        expire_at = row[0]
        if isinstance(expire_at, str):
            expire_at = datetime.fromisoformat(expire_at)
        await update.message.reply_text(
            tr("status_active", lang).format(date=expire_at.strftime("%Y-%m-%d")),
            parse_mode=ParseMode.HTML,
        )
    else:
        await update.message.reply_text(tr("no_sub", lang))



async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    lang = get_lang(context)

    if not has_active_subscription(uid):
        await update.message.reply_text(tr("no_sub", lang))
        return

    question = " ".join(context.args)
    if not question:
        await update.message.reply_text({
            "fa": "❓ لطفاً سؤال را بعد از دستور بنویسید.",
            "en": "❓ Please write your legal question after the command.",
            "ku": "❓ تکایە پرسیارت لە دوای فەرمانەکە بنوسە.",
        }.get(lang, "❓ لطفاً سؤال را بعد از دستور بنویسید."))
        return

    await answer_question(update, context, question, lang)


# ─── روتر پیام‌های متنی منو ─────────────────────────────────────────────────
async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()
    lang = get_lang(context)

    # دستورات منو بر اساس زبان
    if lang == "fa":
        if text == "🛒 خرید اشتراک":
            await buy_cmd(update, context)
        elif text == "📤 ارسال رسید":
            await send_receipt_cmd(update, context)
        elif text == "⚖️ سؤال حقوقی":
            await update.message.reply_text("سؤال خود را بعد از /ask بفرستید.\nمثال:\n<code>/ask قانون کار چیست؟</code>", parse_mode=ParseMode.HTML)
        elif text == "🎤 سؤال صوتی":
            await update.message.reply_text("🎙️ لطفاً سؤال خود را به صورت پیام صوتی (voice) ارسال نمایید.\n\n📌 فقط پیام صوتی تلگرام پشتیبانی می‌شود.")
        elif text == "ℹ️ درباره توکن":
            await about_token(update, context)

    elif lang == "en":
        if text == "🛒 Buy Subscription":
            await buy_cmd(update, context)
        elif text == "📤 Send Receipt":
            await send_receipt_cmd(update, context)
        elif text == "⚖️ Legal Question":
            await update.message.reply_text("Send your question after /ask.\nExample:\n<code>/ask What is labor law?</code>", parse_mode=ParseMode.HTML)
        elif text == "🎤 Voice Question":
            await update.message.reply_text("🎙️ Please send your legal question as a Telegram voice message.\n\n📌 Only Telegram voice messages are supported.")
        elif text == "ℹ️ About Token":
            await about_token(update, context)

    elif lang == "ku":
        if text == "🛒 کڕینی بەشداریکردن":
            await buy_cmd(update, context)
        elif text == "📤 ناردنی پسوڵە":
            await send_receipt_cmd(update, context)
        elif text == "⚖️ پرسیاری یاسایی":
            await update.message.reply_text("پرسیارەکەت بنێرە لە دوای /ask.\nنموونە:\n<code>/ask یاسای کار چییە؟</code>", parse_mode=ParseMode.HTML)
        elif text == "🎤 پرسیاری دەنگی":
            await update.message.reply_text("🎙️ تکایە پرسیارەکەت بە شێوەی پەیامی دەنگی بنێرە.\n\n📌 تەنها پەیامەکانی دەنگی تێلەگرام پشتیوانی دەکرێن.")
        elif text == "ℹ️ دەربارەی تۆکێن":
            await about_token(update, context)

    else:
        await update.message.reply_text("❌ دستور نامعتبر است. لطفاً از منو استفاده کنید.")


async def lang_text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """بررسی و تنظیم زبان پس از انتخاب توسط کاربر"""
    text = (update.message.text or "").strip()
    lang_options = {
        "فارسی": "fa",
        "English": "en",
        "کوردی": "ku"
    }

    if text in lang_options:
        lang = lang_options[text]
        context.user_data["lang"] = lang

        await update.message.reply_text({
            "fa": "✅ زبان به فارسی تغییر کرد.",
            "en": "✅ Language changed to English.",
            "ku": "✅ زمان بۆ کوردی گۆڕدرا."
        }[lang], reply_markup=get_main_menu(lang))
        return

    await text_router(update, context)

# ---------------------------------------------------------------------------#
# 6. Token info, handler wiring & main                                       #
# ---------------------------------------------------------------------------#
TOKEN_IMG = Path(__file__).with_name("reblawcoin.png")  # تصویر لوگوی RLC

async def about_token(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """اطلاعات توکن RLC + لینک خرید (چندزبانه)."""
    msg = update.effective_message
    lang = get_lang(context)

    token_info = {
        "fa": (
            "🎉 <b>توکن RebLawCoin (RLC)</b> – اولین ارز دیجیتال با محوریت خدمات حقوقی.\n\n"
            "<b>اهداف پروژه:</b>\n"
            "• سرمایه‌گذاری در نوآوری‌های حقوقی\n"
            "• نهادینه‌سازی عدالت روی بلاک‌چین\n"
            "• سودآوری پایدار برای سرمایه‌گذاران\n\n"
            "برای خرید سریع روی لینک زیر بزنید:\n"
            "<a href='https://t.me/blum/app?startapp=memepadjetton_RLC_JpMH5-ref_1wgcKkl94N'>خرید از Blum</a>"
        ),
        "en": (
            "🎉 <b>RebLawCoin (RLC)</b> – The first cryptocurrency focused on legal services.\n\n"
            "<b>Project Objectives:</b>\n"
            "• Investing in legal innovations\n"
            "• Institutionalizing justice on blockchain\n"
            "• Sustainable profitability for investors\n\n"
            "Click the link below for quick purchase:\n"
            "<a href='https://t.me/blum/app?startapp=memepadjetton_RLC_JpMH5-ref_1wgcKkl94N'>Buy from Blum</a>"
        ),
        "ku": (
            "🎉 <b>تۆکێنی RebLawCoin (RLC)</b> – یەکەم دراوە دیجیتاڵیی تایبەت بە خزمەتگوزارییە یاساییەکان.\n\n"
            "<b>ئامانجەکانی پڕۆژەکە:</b>\n"
            "• وەبەرهێنان لە داهێنانی یاسایی\n"
            "• دامەزراندنی دادپەروەری بە شێوەی بلۆکچەین\n"
            "• قازانجی بەردەوام بۆ وەبەرهێنەران\n\n"
            "بۆ کڕینی خێرا لەسەر ئەم لینکە کلیک بکە:\n"
            "<a href='https://t.me/blum/app?startapp=memepadjetton_RLC_JpMH5-ref_1wgcKkl94N'>کڕین لە Blum</a>"
        ),
    }

    if TOKEN_IMG.exists():
        await msg.reply_photo(TOKEN_IMG.open("rb"))

    await msg.reply_text(
        token_info.get(lang, token_info["fa"]),
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )

import whisper
import tempfile

model = whisper.load_model("base")  # می‌توانید از مدل‌های دقیق‌تر مانند "small" یا "medium" هم استفاده کنید

async def voice_to_text(file_path):
    result = model.transcribe(file_path)
    return result["text"]

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    lang = get_lang(context)

    if not has_active_subscription(user_id):
        await update.message.reply_text(tr("no_sub", lang))
        return

    voice_file = await update.message.voice.get_file()

    # دانلود فایل صوتی
    with tempfile.NamedTemporaryFile(suffix=".ogg") as voice_temp:
        await voice_file.download_to_drive(voice_temp.name)

        await update.message.reply_text({
            "fa": "🎤 در حال پردازش سؤال صوتی شما...",
            "en": "🎤 Processing your voice message...",
            "ku": "🎤 هەڵسەنگاندنی دەنگی نێردراوت..."
        }[lang])

        # تبدیل صوت به متن
        question_text = await voice_to_text(voice_temp.name)

    # ارسال پاسخ تولیدشده توسط OpenAI
    await answer_question(update, context, question_text, lang)
    await update.message.reply_text({
    "fa": "✅ پاسخ ارسال شد. در صورت نیاز می‌توانید سؤال صوتی دیگری بفرستید.",
    "en": "✅ Answer sent. You may send another voice question if needed.",
    "ku": "✅ وەڵام نێردرا. دەتوانیت پرسیاری دەنگییەکی تر بنێریت."
}[lang])
    

# ─── ثبت تمام هندلرها ───────────────────────────────────────────────────────
def register_handlers(app: Application) -> None:
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("buy", buy_cmd))
    app.add_handler(CommandHandler("send_receipt", send_receipt_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(CommandHandler("about_token", about_token))
    app.add_handler(CommandHandler("lang", lang_cmd))

    app.add_handler(CallbackQueryHandler(callback_handler, pattern=r"^(approve|reject):\d+$"), group=0)

    # ابتدا متن زبان را بررسی و سپس متن رسید را بررسی کنید
    app.add_handler(MessageHandler(filters.Regex("^(فارسی|English|کوردی)$"), lang_text_router), group=1)
    app.add_handler(MessageHandler(filters.PHOTO | (filters.TEXT & ~filters.COMMAND), handle_receipt), group=2)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router), group=3)
    app.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice_message), group=1)

# ─── نقطهٔ ورود اصلی ────────────────────────────────────────────────────────
def main() -> None:
    # ۱) متغیرهای حیاتی
    bot_token = getenv_or_die("BOT_TOKEN")

    # ۲) پایگاه‌داده
    init_db()

    # ۳) ساخت اپلیکیشن
    application = Application.builder().token(bot_token).build()

    # ۴) ثبت هندلرها
    register_handlers(application)

    # ۵) اجرا: polling یا webhook بر اساس USE_WEBHOOK

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
