#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RebLawBot – Telegram bot for legal consultation with RLC subscription
Version 2025-05-17
"""

from __future__ import annotations
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple

from dotenv import load_dotenv
from telegram import Update, Message, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, KeyboardButton
from telegram.constants import ChatAction, ParseMode
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, ContextTypes, filters

from openai import AsyncOpenAI
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# بارگذاری متغیرهای محیطی
load_dotenv()

def getenv_or_die(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing environment variable: {key}")
    return value

BOT_TOKEN = getenv_or_die("BOT_TOKEN")
ADMIN_ID = int(getenv_or_die("ADMIN_ID"))
TON_WALLET_ADDR = getenv_or_die("TON_WALLET_ADDRESS")
BANK_CARD = getenv_or_die("BANK_CARD_NUMBER")
SUBS_DAYS = int(os.getenv("SUBSCRIPTION_DAYS", "30"))
RLC_BONUS_DAYS = int(os.getenv("RLC_BONUS_DAYS", "45"))

# ─── پایگاه داده (SQLite) ─────────────────────────────────────────────
DB_FILE = Path("users.db")
DB_FILE.touch(exist_ok=True)

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()

def init_db() -> None:
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            last_name TEXT,
            expire_at TEXT,
            receipt_photo_id TEXT
        );
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            question TEXT,
            answer TEXT,
            asked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

def upsert_user(user_id: int, username: str | None, first_name: str | None, last_name: str | None) -> None:
    with get_db() as conn:
        conn.execute("""
        INSERT INTO users (user_id, username, first_name, last_name)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            username = excluded.username,
            first_name = excluded.first_name,
            last_name = excluded.last_name
        """, (user_id, username, first_name, last_name))

def has_active_subscription(user_id: int) -> bool:
    with get_db() as conn:
        row = conn.execute("SELECT expire_at FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if not row or not row["expire_at"]:
            return False
        return datetime.fromisoformat(row["expire_at"]) > datetime.utcnow()

def save_subscription(user_id: int, days: int) -> None:
    expire_at = datetime.utcnow() + timedelta(days=days)
    with get_db() as conn:
        conn.execute("UPDATE users SET expire_at = ? WHERE user_id = ?", (expire_at.isoformat(), user_id))

def save_receipt_request(user_id: int, photo_id: str) -> None:
    with get_db() as conn:
        conn.execute("UPDATE users SET receipt_photo_id = ?, expire_at = NULL WHERE user_id = ?", (photo_id, user_id))

def save_question(user_id: int, question: str, answer: str) -> None:
    with get_db() as conn:
        conn.execute("INSERT INTO questions (user_id, question, answer) VALUES (?, ?, ?)", (user_id, question, answer))

# ─── کیبوردهای چندزبانه و پیام‌های خوش‌آمد ────────────────────────────────

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

WELCOME_TEXTS = {
    "fa": (
        "سلام! 👋\n"
        "من <b>ربات حقوقی RebLawBot</b> هستم.\n\n"
        "با تهیه اشتراک می‌توانید سؤالات حقوقی خود را بپرسید.\n"
        "یکی از گزینه‌های زیر را انتخاب کنید 👇"
    ),
    "en": (
        "Hi! 👋\n"
        "I am <b>RebLawBot</b>, your legal assistant.\n\n"
        "To get started, choose an option below 👇"
    ),
    "ku": (
        "سڵاو! 👋\n"
        "ئەمە <b>RebLawBot</b> ـە، یارمەتیدەری یاساییی تۆ.\n\n"
        "بۆ دەستپێکردن، یەکێک لە هەلبژاردەکانی خوارەوە دیاری بکە 👇"
    ),
}

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    پیام خوش‌آمد و منوی چندزبانه بر اساس language_code یا زبان انتخاب‌شدهٔ قبلی
    """
    lang = context.user_data.get("lang")

    if not lang:
        # تشخیص خودکار از language_code کاربر
        lang_code = (update.effective_user.language_code or "").lower()
        if "ku" in lang_code:
            lang = "ku"
        elif "fa" in lang_code:
            lang = "fa"
        else:
            lang = "en"
        context.user_data["lang"] = lang

    # متن و کیبورد مناسب
    text = WELCOME_TEXTS.get(lang, WELCOME_TEXTS["en"])
    kb = {
        "fa": MENU_KB_FA,
        "en": MENU_KB_EN,
        "ku": MENU_KB_KU,
    }.get(lang, MENU_KB_EN)

    await update.message.reply_text(
        text,
        reply_markup=kb,
        parse_mode=ParseMode.HTML,
    )

LANG_KB = InlineKeyboardMarkup([
    [
        InlineKeyboardButton("🇮🇷 فارسی", callback_data="setlang:fa"),
        InlineKeyboardButton("🇬🇧 English", callback_data="setlang:en"),
        InlineKeyboardButton("🇮🇶 کوردی", callback_data="setlang:ku"),
    ]
])

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📌 راهنمای استفاده از RebLawBot:\n\n"
        "• برای پرسیدن سؤال حقوقی:\n"
        "<code>/ask قانون مدنی چیست؟</code>\n\n"
        "• برای تغییر زبان:\n"
        "<code>/lang</code>\n\n"
        "• برای خرید اشتراک:\n"
        "از منو روی «🛒 خرید اشتراک» بزنید.\n\n"
        "هر سوالی داشتید، در خدمت‌تان هستیم 🙏",
        parse_mode=ParseMode.HTML
    )

async def lang_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    پیام انتخاب زبان توسط دکمه‌های اینلاین
    """
    await update.message.reply_text(
        "🌐 لطفاً زبان مورد نظر را انتخاب کنید:",
        reply_markup=LANG_KB
    )
async def lang_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    هندلر پاسخ به کلیک روی دکمه‌های زبان. زبان انتخاب‌شده در context ذخیره شده و منو مجدداً ارسال می‌شود.
    """
    query = update.callback_query
    await query.answer()

    try:
        lang = query.data.split(":")[1]
    except (IndexError, AttributeError):
        return

    # ذخیره انتخاب زبان در context.user_data
    context.user_data["lang"] = lang

    # نمایش پیام تأیید
    await query.edit_message_text("✅ زبان با موفقیت تنظیم شد.")

    # فراخوانی مجدد منوی start بر اساس زبان انتخاب‌شده
    update.message = query.message  # سازگار با start_cmd
    await start_cmd(update, context)

# متن‌های خوش‌آمد بر اساس زبان
WELCOME_FA = (
    "سلام! 👋\n"
    "من <b>ربات حقوقی RebLawBot</b> هستم.\n\n"
    "با تهیه اشتراک می‌توانید سؤالات حقوقی خود را بپرسید.\n"
    "یکی از گزینه‌های زیر را انتخاب کنید 👇"
)

WELCOME_EN = (
    "Hello! 👋\n"
    "I am <b>RebLawBot</b>, your legal assistant.\n\n"
    "To get started, choose an option below 👇"
)

WELCOME_KU = (
    "سڵاو! 👋\n"
    "من <b>RebLawBot</b> ـم، یارمەتیدەری یاساییی تۆ.\n\n"
    "بۆ دەستپێکردن، یەکێک لە هەلبژاردەکانی خوارەوە دیاری بکە 👇"
)

# کیبورد منو بر اساس زبان
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

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    پیام خوش‌آمد و منوی چندزبانه برای کاربر – فارسی، کردی، انگلیسی
    """
    lang_code = (update.effective_user.language_code or "").lower()

    # تشخیص زبان از language_code
    if "ku" in lang_code:
        lang = "ku"
        text = WELCOME_KU
        kb = MENU_KB_KU
    elif "fa" in lang_code:
        lang = "fa"
        text = WELCOME_FA
        kb = MENU_KB_FA
    else:
        lang = "en"
        text = WELCOME_EN
        kb = MENU_KB_EN

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    بررسی وضعیت اشتراک کاربر و نمایش تاریخ انقضا (در صورت وجود)
    """
    uid = update.effective_user.id

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT expire_at FROM users WHERE user_id = ?", (uid,))
        row = cur.fetchone()

    if not row or not row["expire_at"]:
        await update.message.reply_text("❌ شما اشتراک فعالی ندارید.")
        return

    expire_at = row["expire_at"]
    if isinstance(expire_at, str):
        expire_at = datetime.fromisoformat(expire_at)

    if expire_at < datetime.utcnow():
        await update.message.reply_text("⚠️ اشتراک شما منقضی شده است.")
    else:
        await update.message.reply_text(
            f"✅ اشتراک شما تا <b>{expire_at:%Y-%m-%d}</b> فعال است.",
            parse_mode=ParseMode.HTML
        )

    # ذخیره زبان برای استفاده‌های بعدی
    context.user_data["lang"] = lang

    await update.message.reply_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)

async def buy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    ارسال متن راهنمای خرید اشتراک شامل روش‌های پرداخت و مزایای پرداخت با RLC.
    """
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

    await update.message.reply_text(
        BUY_TEXT_FA,
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )

async def send_receipt_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    فعال‌کردن حالت انتظار برای ارسال رسید پرداخت توسط کاربر.
    """
    context.user_data["awaiting_receipt"] = True
    await update.message.reply_text("لطفاً تصویر یا متن رسید پرداخت را ارسال کنید.")

async def handle_receipt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    رسید دریافتی (عکس یا متن) را به مدیر ارسال می‌کند با دکمه‌های تأیید یا رد.
    """
    # اگر منتظر رسید نیستیم و پیام عکس نیست، رد کن
    if not context.user_data.get("awaiting_receipt") and not update.message.photo:
        return

    context.user_data["awaiting_receipt"] = False
    msg: Message = update.message
    user = update.effective_user

    # ذخیرهٔ عکس رسید در دیتابیس
    if msg.photo:
        photo_id = msg.photo[-1].file_id
        save_receipt_request(user.id, photo_id)
    else:
        photo_id = None

    # دکمه‌های اینلاین برای انتخاب نوع پرداخت
    buttons = [
        [
            InlineKeyboardButton("✅ تأیید پرداخت RLC", callback_data=f"approve_rlc:{user.id}"),
            InlineKeyboardButton("✅ تأیید پرداخت TON", callback_data=f"approve_ton:{user.id}"),
            InlineKeyboardButton("✅ تأیید کارت بانکی", callback_data=f"approve_card:{user.id}"),
        ],
        [InlineKeyboardButton("❌ رد", callback_data=f"reject:{user.id}")]
    ]
    kb = InlineKeyboardMarkup(buttons)

    caption = (
        f"📥 رسید جدید از <a href='tg://user?id={user.id}'>{user.full_name}</a>\n"
        f"نام کاربری: @{user.username or '—'}\n\n"
        f"برای تأیید یا رد یکی از گزینه‌ها را انتخاب کنید 👇"
    )

    if photo_id:
        await context.bot.send_photo(
            ADMIN_ID,
            photo=photo_id,
            caption=caption,
            reply_markup=kb,
            parse_mode=ParseMode.HTML
        )
    else:
        text = msg.text or "—"
        await context.bot.send_message(
            ADMIN_ID,
            f"{caption}\n\n📝 {text}",
            reply_markup=kb,
            parse_mode=ParseMode.HTML
        )

    await msg.reply_text("✅ رسید شما برای بررسی مدیر ارسال شد. لطفاً منتظر بمانید.")

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    مدیریت دکمه‌های تأیید یا رد رسید توسط مدیر.
    تأیید = فعال‌سازی اشتراک کاربر
    """
    query = update.callback_query
    await query.answer()

    try:
        action, uid_str = query.data.split(":")
        user_id = int(uid_str)
    except (ValueError, AttributeError):
        return  # دادهٔ نامعتبر

    if update.effective_user.id != ADMIN_ID:
        await query.answer("⛔️ فقط مدیر مجاز به انجام این عملیات است.", show_alert=True)
        return

    # تعیین نوع پرداخت و مدت اشتراک
    if action.startswith("approve_"):
        method = action.split("_")[1]
        if method == "rlc":
            days = 45
            method_text = "پرداخت با RLC"
        elif method == "ton":
            days = 30
            method_text = "پرداخت با TON"
        elif method == "card":
            days = 30
            method_text = "پرداخت کارت بانکی"
        else:
            return

        # فعال‌سازی اشتراک
        save_subscription(user_id, days)
        expire_date = (datetime.utcnow() + timedelta(days=days)).strftime("%Y-%m-%d")
        await context.bot.send_message(
            user_id,
            f"🎉 اشتراک شما تأیید شد و تا <b>{expire_date}</b> فعال است.\nروش پرداخت: <b>{method_text}</b>",
            parse_mode=ParseMode.HTML
        )
        status_note = f"✔️ تأیید شد ({method_text})"

    elif action == "reject":
        await context.bot.send_message(
            user_id,
            "❌ رسید شما رد شد. لطفاً دوباره با رسید معتبر اقدام کنید."
        )
        status_note = "❌ رد شد"

    else:
        return

    # ویرایش پیام مدیر با وضعیت جدید
    new_text = (query.message.caption or query.message.text or "") + f"\n\n<b>{status_note}</b>"
    if query.message.photo:
        await query.message.edit_caption(new_text, parse_mode=ParseMode.HTML)
    else:
        await query.message.edit_text(new_text, parse_mode=ParseMode.HTML)

async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    دریافت سؤال حقوقی از کاربر، بررسی اشتراک، و پاسخ‌دهی با OpenAI.
    """
    uid = update.effective_user.id

    if not has_active_subscription(uid):
        await update.message.reply_text("❌ برای پرسیدن سؤال، ابتدا باید اشتراک تهیه کنید.")
        return

    question = " ".join(context.args)
    if not question:
        await update.message.reply_text("❓ لطفاً سؤال خود را بعد از /ask بنویسید.")
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "شما یک وکیل با تجربه ایرانی هستید. پاسخ را رسمی و دقیق به زبان فارسی با ارجاع به قانون بده."},
                {"role": "user", "content": question},
            ],
            temperature=0.6,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()
        await update.message.reply_text(answer)

    except Exception as e:
        logger.error("خطا در پاسخ OpenAI: %s", e)
        await update.message.reply_text("⚠️ خطایی در پاسخ‌دهی رخ داد. لطفاً بعداً تلاش کنید.")

async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    مسیریابی پیام‌های متنی غیر‌دستور برای منوهای فارسی، انگلیسی، کردی.
    """
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

    # کردی
    elif text == "⚖️ پرسیارى یاسایی":
        await update.message.reply_text(
            "تکایە پرسیارت بە دوای /ask بنووسە.\nوەکوو نموونە:\n<code>/ask یاسای کار چییە؟</code>",
            parse_mode=ParseMode.HTML
        )

    else:
        await update.message.reply_text("❓ دستور نامعتبر است. لطفاً از منو استفاده کنید.")


async def answer_question(update: Update, context: ContextTypes.DEFAULT_TYPE, question: str) -> None:
    """
    ارسال سؤال به OpenAI و پاسخ‌دهی با ذخیره در پایگاه‌داده.
    """
    uid = update.effective_user.id
    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "شما یک وکیل باتجربه هستید. پاسخ‌ها را به‌صورت رسمی، دقیق و با ارجاع به قوانین ارائه دهید."},
                {"role": "user", "content": question}
            ],
            temperature=0.4,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()

        await update.message.reply_text(answer)
        save_question(uid, question, answer)

    except AuthenticationError:
        await update.message.reply_text("❌ کلید OpenAI نامعتبر است.")
    except RateLimitError:
        await update.message.reply_text("⚠️ محدودیت تعداد درخواست به OpenAI فعال شده. لطفاً بعداً تلاش کنید.")
    except APIError as e:
        await update.message.reply_text(f"⚠️ خطای OpenAI: {e}")
    except Exception as e:
        await update.message.reply_text("⚠️ بروز خطا در هنگام دریافت پاسخ. لطفاً بعداً تلاش کنید.")
        logger.error("❌ خطا در ارتباط با OpenAI: %s", e)

def search_law(country: str, keyword: str, limit: int = 3) -> List[Tuple[str, str, str]]:
    """
    جستجوی ساده در پایگاه‌داده قوانین بر اساس کشور و کلمه‌کلیدی.

    پارامترها:
    - country: نام کشور به انگلیسی (مانند iran یا france)
    - keyword: واژهٔ کلیدی برای جستجو در عنوان یا متن ماده
    - limit: تعداد نتایج برگشتی

    خروجی:
    - فهرستی از (عنوان، شماره ماده، متن ماده)
    """
    db_path = Path("laws.db")
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT title, article_number, text
        FROM laws
        WHERE country = ?
        AND (title LIKE ? OR text LIKE ?)
        LIMIT ?
    """, (country.lower(), f"%{keyword}%", f"%{keyword}%", limit))

    results = [(row["title"], row["article_number"], row["text"]) for row in cur.fetchall()]
    conn.close()
    return results

def getenv_or_die(key: str) -> str:
    """
    خواندن متغیر محیطی و توقف اگر مقداردهی نشده باشد.

    مثال:
        TOKEN = getenv_or_die("BOT_TOKEN")

    اگر مقدار یافت نشود، برنامه با خطای RuntimeError متوقف خواهد شد.
    """
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Environment variable {key!r} is missing.")
    return value

def init_db() -> None:
    """
    ایجاد پایگاه‌داده SQLite و جدول‌های users و questions در صورت عدم وجود.
    """
    with sqlite3.connect("users.db") as conn:
        cur = conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                last_name TEXT,
                receipt_photo_id TEXT,
                expire_at TEXT,
                status TEXT DEFAULT 'pending'
            );
            
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                question TEXT,
                answer TEXT,
                asked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.commit()

def has_active_subscription(user_id: int) -> bool:
    """
    بررسی می‌کند که آیا کاربر دارای اشتراک فعال است یا خیر.
    """
    try:
        with sqlite3.connect("users.db") as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT expire_at FROM users WHERE user_id = ?", (user_id,))
            row = cur.fetchone()

        if not row or not row["expire_at"]:
            return False

        expire_at = row["expire_at"]
        if isinstance(expire_at, str):
            expire_at = datetime.fromisoformat(expire_at)

        return expire_at >= datetime.utcnow()

    except Exception as e:
        print("DB Error in has_active_subscription:", e)
        return False
def getenv_or_die(key: str) -> str:
    """
    دریافت مقدار یک متغیر محیطی یا توقف اجرای برنامه در صورت نبود آن.
    """
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Environment variable '{key}' is missing.")
    return value

def has_active_subscription(user_id: int) -> bool:
    """
    بررسی می‌کند که آیا کاربر دارای اشتراک فعال است یا خیر.
    """
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT expire_at FROM users WHERE user_id = ?", (user_id,))
        row = cur.fetchone()

    if not row or not row["expire_at"]:
        return False

    try:
        expire_at = datetime.fromisoformat(row["expire_at"])
    except ValueError:
        return False

    return expire_at >= datetime.utcnow()

def save_subscription(user_id: int, days: int = 30) -> None:
    """
    ذخیره تاریخ انقضا برای اشتراک کاربر.
    """
    expire_at = (datetime.utcnow() + timedelta(days=days)).isoformat(timespec="seconds")
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE users
            SET expire_at = ?
            WHERE user_id = ?
        """, (expire_at, user_id))
        conn.commit()

def set_user_status(user_id: int, status: str) -> None:
    """
    به‌روزرسانی وضعیت کاربر (مثلاً 'rejected' برای رسید ردشده).
    """
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE users
            SET status = ?
            WHERE user_id = ?
        """, (status, user_id))
        conn.commit()

def save_subscription(user_id: int, days: int = 30) -> None:
    """
    ذخیره تاریخ انقضای اشتراک در پایگاه‌داده برای مدت مشخص (پیش‌فرض: ۳۰ روز).
    """
    expire_at = (datetime.utcnow() + timedelta(days=days)).isoformat(timespec="seconds")
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE users
            SET expire_at = ?, status = 'active'
            WHERE user_id = ?
        """, (expire_at, user_id))
        conn.commit()

def set_user_status(user_id: int, status: str) -> None:
    """
    به‌روزرسانی وضعیت کاربر در جدول users (مثلاً: 'pending'، 'active'، 'rejected').
    """
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE users
            SET status = ?
            WHERE user_id = ?
        """, (status, user_id))
        conn.commit()

def save_subscription(user_id: int, days: int = 30) -> None:
    """
    تنظیم تاریخ انقضای اشتراک کاربر به تعداد روز مشخص‌شده (پیش‌فرض: 30).
    """
    expire_at = (datetime.utcnow() + timedelta(days=days)).strftime("%Y-%m-%d")
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE users
            SET expire_at = ?, status = 'active'
            WHERE user_id = ?
        """, (expire_at, user_id))
        conn.commit()

def set_user_status(user_id: int, status: str) -> None:
    """
    به‌روزرسانی وضعیت کاربر (مثلاً: 'active', 'pending', 'rejected')
    """
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE users
            SET status = ?
            WHERE user_id = ?
        """, (status, user_id))
        conn.commit()

def save_subscription(user_id: int, days: int) -> None:
    """
    فعال‌سازی اشتراک کاربر برای تعداد روز مشخص
    """
    expire_at = (datetime.utcnow() + timedelta(days=days)).isoformat(timespec="seconds")
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE users
            SET expire_at = ?, status = 'active'
            WHERE user_id = ?
        """, (expire_at, user_id))
        conn.commit()

def set_user_status(user_id: int, status: str) -> None:
    """
    به‌روزرسانی وضعیت کاربر (مثلاً 'pending'، 'rejected'، 'active')
    """
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE users
            SET status = ?
            WHERE user_id = ?
        """, (status, user_id))
        conn.commit()

def save_subscription(user_id: int, days: int = 30) -> None:
    """
    ثبت یا تمدید اشتراک کاربر به‌مدت مشخص (پیش‌فرض ۳۰ روز).
    """
    expire_at = (datetime.utcnow() + timedelta(days=days)).isoformat(timespec="seconds")
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE users
            SET expire_at = ?, status = 'active'
            WHERE user_id = ?
        """, (expire_at, user_id))
        conn.commit()

def get_user_lang(context: ContextTypes.DEFAULT_TYPE) -> str:
    """
    بازیابی زبان ذخیره‌شده در user_data. اگر نباشد، 'fa' پیش‌فرض است.
    """
    return context.user_data.get("lang", "fa")

async def law_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    جستجوی قوانین در پایگاه داده با استفاده از نام کشور و کلیدواژه.
    مثال: /law iran کار
    """
    if len(context.args) < 2:
        await update.message.reply_text(
            "❗️مثال استفاده:\n<code>/law iran کار</code>",
            parse_mode=ParseMode.HTML,
        )
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

TOKEN_IMG = Path(__file__).with_name("reblawcoin.png")  # فایل تصویر توکن RLC

async def about_token(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    ارسال توضیح درباره RebLawCoin (RLC) و لینک خرید.
    """
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

LANG_KB = InlineKeyboardMarkup([
    [
        InlineKeyboardButton("🇮🇷 فارسی", callback_data="setlang:fa"),
        InlineKeyboardButton("🇬🇧 English", callback_data="setlang:en"),
        InlineKeyboardButton("🇮🇶 کوردی", callback_data="setlang:ku")
    ]
])

async def lang_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    نمایش کیبورد انتخاب زبان.
    """
    await update.message.reply_text(
        "🌐 لطفاً زبان مورد نظر خود را انتخاب کنید:",
        reply_markup=LANG_KB
    )

async def lang_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    ذخیره زبان انتخاب‌شده و اجرای مجدد /start برای نمایش منو.
    """
    query = update.callback_query
    await query.answer()

    lang = query.data.split(":")[1]
    context.user_data["lang"] = lang

    # اجرای مجدد /start با زبان جدید
    update.message = query.message  # برای سازگاری
    await start_cmd(update, context)

def register_handlers(app: Application) -> None:
    """ثبت فرمان‌ها و هندلرهای ربات."""
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("buy", buy_cmd))
    app.add_handler(CommandHandler("send_receipt", send_receipt_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(CommandHandler("about_token", about_token))
    app.add_handler(CommandHandler("law", law_cmd))
    app.add_handler(CommandHandler("lang", lang_cmd))
    app.add_handler(CommandHandler("help", help_cmd))

    # انتخاب زبان از طریق دکمه‌های اینلاین
    app.add_handler(CallbackQueryHandler(lang_callback, pattern=r"^setlang:(fa|en|ku)$"))

    # تأیید یا رد رسید توسط مدیر
    app.add_handler(CallbackQueryHandler(callback_handler, pattern=r"^(approve_(rlc|ton|card)|reject):\d+$"), group=0)

    # هندل عکس یا متن به‌عنوان رسید
    app.add_handler(MessageHandler(filters.PHOTO | (filters.TEXT & ~filters.COMMAND), handle_receipt), group=1)

    # پیام‌های متنی منوی اصلی
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router), group=2)

def main() -> None:
    """اجرای اصلی ربات با بارگذاری توکن، دیتابیس و راه‌اندازی اپلیکیشن."""
    # گرفتن توکن از .env
    bot_token = getenv_or_die("BOT_TOKEN")

    # مقداردهی اولیه پایگاه‌داده
    init_db()

    # ساخت اپلیکیشن با توکن
    application = Application.builder().token(bot_token).build()

    # ثبت هندلرها
    register_handlers(application)

    # اجرای ربات به‌صورت polling (در صورت نیاز می‌توان به webhook تغییر داد)
    application.run_polling(allowed_updates=Update.ALL_TYPES)
if __name__ == "__main__":
    main()

TOKEN_IMG = Path(__file__).with_name("reblawcoin.png")

async def about_token(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    ارسال اطلاعات درباره توکن RebLawCoin (RLC) به همراه لینک خرید و تصویر.
    """
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
