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
from telegram import ReplyKeyboardMarkup, KeyboardButton
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

def search_law(country: str, keyword: str) -> List[Tuple[str, str, str]]:
    """
    جستجو در پایگاه داده بر اساس کشور و کلیدواژه.
    برمی‌گرداند: (law_title, article_number, article_text)
    """
    conn = sqlite3.connect("laws.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    query = """
    SELECT law_title, article_number, article_text
    FROM laws
    WHERE country LIKE ? AND article_text LIKE ?
    LIMIT 10;
    """
    cur.execute(query, (f"%{country}%", f"%{keyword}%"))
    results = cur.fetchall()
    conn.close()

    return [(row["law_title"], row["article_number"], row["article_text"]) for row in results]

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
    هندلر دکمه‌های اینلاین «تأیید با نوع پرداخت» یا «رد» رسید کاربر.
    فقط مدیر مجاز به استفاده است.
    """
    query = update.callback_query
    await query.answer()

    try:
        action, uid_str = query.data.split(":")
        target_uid = int(uid_str)
    except (ValueError, AttributeError):
        return  # دادهٔ callback نامعتبر است

    # فقط مدیر مجاز به انجام این عملیات است
    if update.effective_user.id != ADMIN_ID:
        await query.answer("⛔️ فقط مدیر مجاز به این عملیات است.", show_alert=True)
        return

    if action.startswith("approve"):
        if action == "approve_rlc":
            days = 45
            status_note = "✔️ تأیید شد (RLC)"
        elif action == "approve_ton":
            days = 30
            status_note = "✔️ تأیید شد (TON)"
        else:  # approve_card
            days = 30
            status_note = "✔️ تأیید شد (کارت بانکی)"

        # ذخیره در پایگاه‌داده
        set_user_expiration(target_uid, days)

        # پیام به کاربر
        await context.bot.send_message(
            target_uid,
            f"🎉 اشتراک شما تأیید شد و تا <b>{days} روز</b> فعال است.",
            parse_mode=ParseMode.HTML
        )

    elif action == "reject":
        set_user_status(target_uid, "rejected")
        await context.bot.send_message(
            target_uid,
            "❌ رسید شما رد شد. لطفاً با رسید معتبر دوباره تلاش کنید."
        )
        status_note = "❌ رد شد"
    else:
        return  # callback غیرمنتظره

    # ویرایش پیام مدیر برای ثبت نتیجه
    new_text = (query.message.caption or query.message.text or "") + f"\n\n<b>وضعیت:</b> {status_note}"
    if query.message.photo:
        await query.message.edit_caption(new_text, parse_mode=ParseMode.HTML)
    else:
        await query.message.edit_text(new_text, parse_mode=ParseMode.HTML)


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

TON_WALLET_ADDR = os.getenv("TON_WALLET_ADDRESS", "TON_NOT_SET")
BANK_CARD = os.getenv("BANK_CARD_NUMBER", "CARD_NOT_SET")


# جایگزینی تابع

async def buy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        BUY_TEXT_FA,
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )



BUY_TEXT_FA = (
    "🛒 <b>راهنمای خرید اشتراک</b>\n\n"
    "۱️⃣ پرداخت 1 TON به آدرس کیف‌پول:\n"
    f"<code>{TON_WALLET_ADDR}</code>\n\n"
    "۲️⃣ واریز ۵۰۰٬۰۰۰ تومان به شماره کارت:\n"
    f"<code>{BANK_CARD}</code>\n\n"
    "۳️⃣ یا پرداخت با 1,800,000 <b>RLC</b> به آدرس:\n"
    "<code>UQBkRlKAi6Rk4EuZqJ8QrxDgugKK1kLUS6Yp4lOE6MPiRkGW</code>\n"
    "🔗 <a href='https://t.me/blum/app?startapp=memepadjetton_RLC_JpMH5-ref_1wgcKkl94N'>خرید مستقیم از Blum</a>\n\n"
    "<b>🎁 مزایای پرداخت با RLC:</b>\n"
    "• اشتراک ۴۵ روزه (به‌جای ۳۰ روز)\n"
    "• دسترسی رایگان به قوانین بین‌المللی\n"
    "• اولویت در پاسخ‌دهی به سؤالات حقوقی\n\n"
    "پس از پرداخت، از دکمه «📤 ارسال رسید» استفاده کنید."
)



def register_handlers(app: Application) -> None:
    # فرمان‌های اصلی
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("buy", buy_cmd))
    app.add_handler(CommandHandler("send_receipt", send_receipt_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(CommandHandler("about_token", about_token))

    # دکمه‌های تأیید/رد رسید با نوع پرداخت
    app.add_handler(
    CallbackQueryHandler(callback_handler, pattern=r"^approve_(rlc|ton|card):\d+$"),
    group=0,
)  
    app.add_handler(
    CallbackQueryHandler(callback_handler, pattern=r"^reject:\d+$"),
    group=0,
)


    # هندل رسید عکس یا متن – گروه 1
    app.add_handler(
        MessageHandler(filters.PHOTO | (filters.TEXT & ~filters.COMMAND), handle_receipt),
        group=1,
    )

    # پیام‌های متنی منو – گروه 2
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, text_router),
        group=2,
    )


# ─── فرمان‌ها ────────────────────────────────────────────────────────────────
LANG_KB = InlineKeyboardMarkup([
    [
        InlineKeyboardButton("🇮🇷 فارسی", callback_data="setlang:fa"),
        InlineKeyboardButton("🇬🇧 English", callback_data="setlang:en"),
        InlineKeyboardButton("🇮🇶 کوردی", callback_data="setlang:ku")
    ]
])

MENU_KB_FA = ReplyKeyboardMarkup(
    [
        [KeyboardButton("🛒 خرید اشتراک"), KeyboardButton("📤 ارسال رسید")],
        [KeyboardButton("⚖️ سؤال حقوقی"), KeyboardButton("📚 جستجوی قانون")],
        [KeyboardButton("ℹ️ درباره توکن")],
    ],
    resize_keyboard=True,
)

MENU_KB_EN = ReplyKeyboardMarkup(
    [
        [KeyboardButton("🛒 Buy Subscription"), KeyboardButton("📤 Send Receipt")],
        [KeyboardButton("⚖️ Legal Question"), KeyboardButton("📚 Search Law")],
        [KeyboardButton("ℹ️ About Token")],
    ],
    resize_keyboard=True,
)

MENU_KB_KU = ReplyKeyboardMarkup(
    [
        [KeyboardButton("🛒 کڕینی بەشداریکردن"), KeyboardButton("📤 ناردنی وەرگرتن")],
        [KeyboardButton("⚖️ پرسیارى یاسایی"), KeyboardButton("📚 گەڕان لە یاسا")],
        [KeyboardButton("ℹ️ دەربارەی توکەن")],
    ],
    resize_keyboard=True,
)
TON_WALLET_ADDR = getenv_or_die("TON_WALLET_ADDRESS")
BANK_CARD = getenv_or_die("BANK_CARD_NUMBER")

async def lang_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🌐 زبان مورد نظر خود را انتخاب کنید:",
        reply_markup=LANG_KB
    )

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    پیام خوش‌آمد و منوی چندزبانه برای کاربر – فارسی، کردی، انگلیسی
    """
    # گرفتن کد زبان، پیش‌فرض خالی (برای جلوگیری از خطا)
    lang_code = (update.effective_user.language_code or "").lower()

    # تشخیص زبان از روی language_code
    if "ku" in lang_code:
        lang = "ku"
    elif "fa" in lang_code:
        lang = "fa"
    else:
        lang = "en"

    # پیام خوش‌آمد و کیبورد بر اساس زبان
    if lang == "ku":
        text = (
            "سڵاو! 👋\n"
            "ئەمە <b>RebLawBot</b> ـە، یارمەتیدەری یاساییی تۆ.\n\n"
            "بۆ دەستپێکردن، یەکێک لە هەلبژاردەکانی خوارەوە دیاری بکە 👇"
        )
        kb = MENU_KB_KU

    elif lang == "fa":
        text = (
            "سلام! 👋\n"
            "من <b>ربات حقوقی RebLawBot</b> هستم.\n\n"
            "با تهیه اشتراک می‌توانید سؤالات حقوقی خود را بپرسید.\n"
            "یکی از گزینه‌های زیر را انتخاب کنید 👇"
        )
        kb = MENU_KB_FA

    else:
        text = (
            "Hi! 👋\n"
            "I am <b>RebLawBot</b>, your legal assistant.\n\n"
            "To get started, choose an option below 👇"
        )
        kb = MENU_KB_EN

    await update.message.reply_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)

async def lang_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    lang = query.data.split(":")[1]

    # ذخیره زبان انتخاب‌شده در user_data
    context.user_data["lang"] = lang

    # پیام تأیید و اجرای start مجدد برای بازسازی منو
    await query.edit_message_text("✅ زبان با موفقیت تنظیم شد.")
    update.message = query.message  # برای سازگاری با start_cmd
    await start_cmd(update, context)

# دکمه یا فرمان «📤 ارسال رسید»؛ کاربر باید بلافاصله عکس یا متن ارسال کند
async def send_receipt_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["awaiting_receipt"] = True
    await update.message.reply_text("لطفاً تصویر یا متن رسید پرداخت را ارسال کنید.")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    if has_active_subscription(uid):
        row = _fetchone("SELECT expire_at FROM users WHERE user_id=" + _PLACEHOLDER, (uid,))
        expire_at = row[0]
        if isinstance(expire_at, str):
            expire_at = datetime.fromisoformat(expire_at)
        await update.message.reply_text(
            f"✅ اشتراک شما تا <b>{expire_at:%Y-%m-%d}</b> فعال است.",
            parse_mode=ParseMode.HTML,
        )
    else:
        await update.message.reply_text(MSG_NO_SUB_FA)

async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    if not has_active_subscription(uid):
        await update.message.reply_text(MSG_NO_SUB_FA)
        return

    question = " ".join(context.args)
    if not question:
        await update.message.reply_text("❓ سؤال را بعد از دستور بنویسید.")
        return

    await answer_question(update, context, question)

async def law_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if len(context.args) < 2:
        await update.message.reply_text("❗️مثال استفاده:\n<code>/law iran کار</code>", parse_mode=ParseMode.HTML)
        return

    country = context.args[0].capitalize()
    keyword = " ".join(context.args[1:])
    results = search_law(country, keyword)

    if not results:
        await update.message.reply_text("❌ موردی یافت نشد.")
        return

    for title, number, text in results:
        await update.message.reply_text(
            f"<b>{title}</b>\n📘 <b>{number}</b>\n{text}",
            parse_mode=ParseMode.HTML
        )

# ─── روتر پیام‌های متنی منو ─────────────────────────────────────────────────
async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip().lower()

    # فارسی
    if text == "🛒 خرید اشتراک":
        await buy_cmd(update, context)
    elif text == "📤 ارسال رسید":
        await send_receipt_cmd(update, context)
    elif text == "⚖️ سؤال حقوقی":
        await update.message.reply_text(
            "برای پرسیدن سؤال حقوقی، از دستور زیر استفاده کنید:\n"
            "<code>/ask قانون کار چیست؟</code>",
            parse_mode=ParseMode.HTML
        )
    elif text == "📚 جستجوی قانون":
        await update.message.reply_text(
            "برای جستجوی مادهٔ قانونی:\n"
            "<code>/law ایران کار</code>\n"
            "یا\n"
            "<code>/law france constitution</code>",
            parse_mode=ParseMode.HTML
        )
    elif text == "ℹ️ درباره توکن":
        await about_token(update, context)

    # English
    elif text == "🛒 buy subscription":
        await buy_cmd(update, context)
    elif text == "📤 send receipt":
        await send_receipt_cmd(update, context)
    elif text == "⚖️ legal question":
        await update.message.reply_text(
            "To ask a legal question, use:\n"
            "<code>/ask What is labor law?</code>",
            parse_mode=ParseMode.HTML
        )
    elif text == "📚 search law":
        await update.message.reply_text(
            "To search laws by keyword:\n"
            "<code>/law france constitution</code>\n"
            "<code>/law iran contract</code>",
            parse_mode=ParseMode.HTML
        )
    elif text == "ℹ️ about token":
        await about_token(update, context)

    elif text == "⚖️ پرسیارى یاسایی":
        await update.message.reply_text(
        "تکایە پرسیارت بە دوای /ask بنووسە.\nوەکوو نموونە:\n<code>/ask یاسای کار چییە؟</code>",
        parse_mode=ParseMode.HTML
    )    

    else:
        await update.message.reply_text("❓ دستور نامعتبر است. لطفاً از منو استفاده کنید.") 
        

# ---------------------------------------------------------------------------#
# 6. Token info, handler wiring & main                                       #
# ---------------------------------------------------------------------------#
TOKEN_IMG = Path(__file__).with_name("reblawcoin.png")  # تصویر لوگوی RLC

async def about_token(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """اطلاعات توکن RLC + لینک خرید."""
    msg = update.effective_message
    if TOKEN_IMG.exists():
        await msg.reply_photo(TOKEN_IMG.open("rb"))
    await msg.reply_text(
        (
            "🎉 <b>توکن RebLawCoin (RLC)</b> – اولین ارز دیجیتال با محوریت خدمات حقوقی.\n\n"
            "<b>اهداف پروژه:</b>\n"
            "• سرمایه‌گذاری در نوآوری‌های حقوقی\n"
            "• نهادینه‌سازی عدالت روی بلاک‌چین\n"
            "• سودآوری پایدار برای سرمایه‌گذاران\n\n"
            "برای خرید سریع روی لینک زیر بزنید:\n"
            "<a href='https://t.me/blum/app?startapp=memepadjetton_RLC_JpMH5-ref_1wgcKkl94N'>خرید از Blum</a>"
        ),
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lang_code = (update.effective_user.language_code or "").lower()

    if "ku" in lang_code:
        text = (
            "📘 <b>یارمەتیدانی بۆ بەکارهێنانی RebLawBot</b>\n\n"
            "• کڕینی بەشداریکردن: /buy\n"
            "• ناردنی وەرگرتن: /send_receipt\n"
            "• پرسیاری یاسایی: /ask <پرسیار>\n"
            "• گەڕان لە یاسا: /law iran کار\n"
            "• گۆڕینی زمان: /lang"
        )
    elif "fa" in lang_code:
        text = (
            "📘 <b>راهنمای استفاده از RebLawBot</b>\n\n"
            "• خرید اشتراک: /buy\n"
            "• ارسال رسید: /send_receipt\n"
            "• پرسش حقوقی: /ask <سؤال>\n"
            "• جستجوی قانون: /law iran کار\n"
            "• تغییر زبان: /lang"
        )
    else:
        text = (
            "📘 <b>How to use RebLawBot</b>\n\n"
            "• Buy a subscription: /buy\n"
            "• Send receipt: /send_receipt\n"
            "• Ask a legal question: /ask <your question>\n"
            "• Search laws: /law france constitution\n"
            "• Change language: /lang"
        )

    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

# ─── ثبت تمام هندلرها ───────────────────────────────────────────────────────
def register_handlers(app: Application) -> None:
    # دستورات اصلی
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("buy", buy_cmd))
    app.add_handler(CommandHandler("send_receipt", send_receipt_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(CommandHandler("about_token", about_token))
    app.add_handler(CommandHandler("law", law_cmd))
    app.add_handler(CommandHandler("lang", lang_cmd))
    app.add_handler(CallbackQueryHandler(lang_callback, pattern=r"^setlang:(fa|en|ku)$"))

 
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
    app.add_handler(CommandHandler("help", help_cmd))


    # سایر پیام‌های متنی منو – گروه 2
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, text_router),
        group=2,
    )

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
