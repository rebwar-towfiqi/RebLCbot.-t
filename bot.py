#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RebLawBot – Telegram bot that sells subscriptions and answers legal questions using OpenAI.
Version 2025-05-13 – Fixed handler order, buy_cmd bug, and HTML formatting
"""

from __future__ import annotations


from http import client
import logging
from mimetypes import init_db
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta

from pathlib import Path
from typing import Generator, Optional, Tuple, List
from venv import logger

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError, RateLimitError, AuthenticationError
from psycopg2.pool import SimpleConnectionPool
from telegram import (
    Update, Message, InlineKeyboardMarkup, InlineKeyboardButton,
    ReplyKeyboardMarkup, KeyboardButton
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, ContextTypes, filters
)
TON_WALLET_ADDR = os.getenv("TON_WALLET_ADDRESS", "TON_NOT_SET")
BANK_CARD = os.getenv("BANK_CARD_NUMBER", "CARD_NOT_SET")
SUBS_DAYS = int(os.getenv("SUBSCRIPTION_DAYS", "30"))

# ─── تنظیم پایگاه داده SQLite ──────────────────────────────────────
SQLITE_FILE = Path("users.db")
_sqlite_lock = sqlite3.RLock()

@contextmanager
def get_db():
    with _sqlite_lock, sqlite3.connect(SQLITE_FILE) as conn:
        conn.row_factory = sqlite3.Row
        yield conn

def init_db():
    SQLITE_FILE.touch(exist_ok=True)
    ddl = """
    CREATE TABLE IF NOT EXISTS users (
        user_id          INTEGER PRIMARY KEY,
        username         TEXT,
        first_name       TEXT,
        last_name        TEXT,

        receipt_photo_id TEXT,
        expire_at        TEXT
    );
    CREATE TABLE IF NOT EXISTS questions (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    INTEGER,
        question   TEXT,
        answer     TEXT,
        asked_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    with get_db() as conn:
        conn.executescript(ddl)
        conn.commit()

def upsert_user(user_id: int, username: str | None, first_name: str | None, last_name: str | None) -> None:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO users (user_id, username, first_name, last_name)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                username = excluded.username,
                first_name = excluded.first_name,
                last_name = excluded.last_name
        """, (user_id, username, first_name, last_name))
        conn.commit()
def save_receipt_request(user_id: int, photo_id: str) -> None:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE users
            SET receipt_photo_id = ?, expire_at = NULL
            WHERE user_id = ?
        """, (photo_id, user_id))
        conn.commit()
def save_question(user_id: int, question: str, answer: str) -> None:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO questions (user_id, question, answer)
            VALUES (?, ?, ?)
        """, (user_id, question, answer))
        conn.commit()

def set_user_expiration(user_id: int, days: int = 30) -> None:
    expire_at = (datetime.utcnow() + timedelta(days=days)).isoformat(timespec="seconds")
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE users
            SET expire_at = ?
            WHERE user_id = ?
        """, (expire_at, user_id))
        conn.commit()

def has_active_subscription(user_id: int) -> bool:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT expire_at FROM users WHERE user_id = ?", (user_id,))
        row = cur.fetchone()

    if not row or not row["expire_at"]:
        return False

    expire_at = row["expire_at"]
    if isinstance(expire_at, str):
        expire_at = datetime.fromisoformat(expire_at)

    return expire_at >= datetime.utcnow()

# ─── متن راهنمای خرید اشتراک ────────────────────────────────
BUY_TEXT_FA = (
    "🛒 <b>راهنمای خرید اشتراک</b>\n\n"
    "۱️⃣ پرداخت 1 TON به آدرس کیف‌پول زیر:\n"
    f"<code>{TON_WALLET_ADDR}</code>\n\n"
    "۲️⃣ واریز ۵۰۰٬۰۰۰ تومان به شماره‌کارت زیر:\n"
    f"<code>{BANK_CARD}</code>\n\n"
    "پس از پرداخت، از دکمه «📤 ارسال رسید» استفاده کنید."
)

# ─── تابع پاسخ به دستور /buy یا دکمه «خرید اشتراک» ──────────
async def buy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        BUY_TEXT_FA,
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )   
# ─── منوی اصلی (کلیدهای فارسی) ────────────────────────────────
MENU_KB = ReplyKeyboardMarkup(
    [
        [KeyboardButton("🛒 خرید اشتراک"), KeyboardButton("📤 ارسال رسید")],
        [KeyboardButton("⚖️ سؤال حقوقی"), KeyboardButton("ℹ️ درباره توکن")],
    ],
    resize_keyboard=True,
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
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))

# ─── تابع /start ─────────────────────────────────────────────
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lang = "fa" if update.effective_user.language_code.startswith("fa") else "en"
    text = WELCOME_FA if lang == "fa" else WELCOME_EN
    await update.message.reply_text(
        text,
        reply_markup=MENU_KB,
        parse_mode=ParseMode.HTML
    )

# ─── فرمان ارسال رسید (فعالسازی حالت انتظار برای عکس یا متن رسید) ────────────
async def send_receipt_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["awaiting_receipt"] = True
    await update.message.reply_text(
        "لطفاً تصویر یا متن رسید پرداخت را ارسال کنید."
    )

# ─── تابع دریافت رسید و ارسال برای بررسی مدیر ─────────────────────────────
async def handle_receipt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # فقط وقتی منتظر رسید هستیم یا عکس دریافت شده ادامه دهیم
    if not context.user_data.get("awaiting_receipt") and not update.message.photo:
        return  # این پیام رسید نیست؛ بگذار هندلر بعدی بررسی کند

    # پس از پذیرش، فلگ را پاک می‌کنیم
    context.user_data["awaiting_receipt"] = False


    msg: Message = update.message
    uid = update.effective_user.id

    # ساخت دکمه‌های تأیید و رد
    kb = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✅ تأیید", callback_data=f"approve:{uid}"),
                InlineKeyboardButton("❌ رد", callback_data=f"reject:{uid}"),
            ]
        ]
    )

    caption = (
        f"📄 رسید جدید از <a href='tg://user?id={uid}'>{uid}</a>\n"
        f"نام: {msg.from_user.full_name}\n"
        "برای بررسی:"
    )

    if msg.photo:
        photo_id = msg.photo[-1].file_id
        await context.bot.send_photo(
            ADMIN_ID,
            photo_id,
            caption=caption,
            reply_markup=kb,
            parse_mode=ParseMode.HTML,
        )
    else:
        text = msg.text or "رسید متنی"
        await context.bot.send_message(
            ADMIN_ID,
            f"{caption}\n\n{text}",
            reply_markup=kb,
            parse_mode=ParseMode.HTML,
        )

    await msg.reply_text("✅ رسید شما برای بررسی ارسال شد؛ لطفاً منتظر تأیید مدیر بمانید.")

# ─── تابع پاسخ به دکمه‌های تأیید یا رد رسید ─────────────────────────────
# ─── تابع پاسخ به دکمه‌های تأیید یا رد رسید ─────────────────────────────
# ─── تابع پاسخ به دکمه‌های تأیید یا رد رسید ─────────────────────────────
async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  
    query = update.callback_query
    await query.answer()

    try:
        action, uid_str = query.data.split(":")
        target_uid = int(uid_str)
    except (ValueError, AttributeError):
        return  # دادهٔ callback نامعتبر است

    # فقط مدیر مجاز به کلیک دکمه است
    if update.effective_user.id != ADMIN_ID:
        await query.answer("⛔️ شما مجاز به انجام این عملیات نیستید.", show_alert=True)
        return

    if action == "approve":
        # فعال‌سازی اشتراک کاربر
        expire_at = (datetime.utcnow() + timedelta(days=SUBS_DAYS)).strftime("%Y-%m-%d")
        await context.bot.send_message(
            target_uid,
            f"🎉 اشتراک شما تأیید شد و تا <b>{expire_at}</b> فعال است.",
            parse_mode=ParseMode.HTML
        )
        status_note = "✔️ تأیید شد"
    else:
        await context.bot.send_message(
            target_uid,
            "❌ رسید شما رد شد. لطفاً با رسید معتبر دوباره تلاش کنید."
        )
        status_note = "❌ رد شد"

    # ویرایش پیام رسید ارسالی به مدیر و افزودن وضعیت
    new_text = (query.message.caption or query.message.text) + f"\n\nوضعیت: {status_note}"
    if query.message.photo:
        await query.message.edit_caption(new_text, parse_mode=ParseMode.HTML)
    else:
        await query.message.edit_text(new_text, parse_mode=ParseMode.HTML)
    # ویرایش پیام رسید ارسالی به مدیر و افزودن وضعیت
    new_text = (query.message.caption or query.message.text) + f"\n\nوضعیت: {status_note}"
    if query.message.photo:
        await query.message.edit_caption(new_text, parse_mode=ParseMode.HTML)
    else:
        await query.message.edit_text(new_text, parse_mode=ParseMode.HTML)
    # ویرایش پیام رسید ارسالی به مدیر و افزودن وضعیت
    new_text = (query.message.caption or query.message.text) + f"\n\nوضعیت: {status_note}"
    if query.message.photo:
        await query.message.edit_caption(new_text, parse_mode=ParseMode.HTML)
    else:
        await query.message.edit_text(new_text, parse_mode=ParseMode.HTML)
# ─── بررسی وضعیت اشتراک کاربر ─────────────────────────────
async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id

    # بررسی پایگاه داده برای اشتراک فعال
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT expire_at FROM users WHERE user_id = ?", (uid,))
        row = cur.fetchone()

    if not row or not row[0]:
        await update.message.reply_text("❌ اشتراک فعالی برای شما ثبت نشده است.")
        return

    expire_at = row[0]
    if isinstance(expire_at, str):
        expire_at = datetime.fromisoformat(expire_at)

    if expire_at < datetime.utcnow():
        await update.message.reply_text("⚠️ اشتراک شما منقضی شده است.")
    else:
        await update.message.reply_text(
            f"✅ اشتراک شما تا <b>{expire_at:%Y-%m-%d}</b> فعال است.",
            parse_mode=ParseMode.HTML
        )
# ─── دریافت پاسخ حقوقی با OpenAI فقط برای کاربران دارای اشتراک ───────────────
async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id

    # بررسی اشتراک
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT expire_at FROM users WHERE user_id = ?", (uid,))
        row = cur.fetchone()

    if not row or not row[0] or datetime.fromisoformat(row[0]) < datetime.utcnow():
        await update.message.reply_text("❌ برای پرسش، ابتدا باید اشتراک معتبر تهیه کنید.")
        return

    # ترکیب آرگومان‌های دستور به‌عنوان سؤال
    question = " ".join(context.args)
    if not question:
        await update.message.reply_text("❓ لطفاً سؤال خود را پس از دستور /ask بنویسید.")
        return

    # ارسال وضعیت در حال نوشتن
    await update.message.chat.send_action(ChatAction.TYPING)

    # ارسال به OpenAI
    try:
        rsp = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "شما یک وکیل با تجربه ایرانی هستید. پاسخ را رسمی و دقیق به زبان فارسی با ارجاع به قوانین بده."},
                {"role": "user", "content": question}
            ],
            temperature=0.6,
            max_tokens=1024
        )
        answer = rsp.choices[0].message.content.strip()
        await update.message.reply_text(answer)
    except Exception as e:
        logger.error("خطا در ارتباط با OpenAI: %s", e)
        await update.message.reply_text("⚠️ خطا در ارتباط با سرویس پاسخ‌دهی. لطفاً بعداً دوباره تلاش کنید.")

# ─── ارسال اطلاعات درباره توکن RebLawCoin (RLC) ──────────────────────────────
TOKEN_IMG = Path(__file__).with_name("reblawcoin.png")  # ← فایل لوگو در کنار bot.py

async def about_token(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
   
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
            "برای خرید توکن روی لینک زیر بزنید:\n"
            "<a href='https://t.me/blum/app?startapp=memepadjetton_RLC_JpMH5-ref_1wgcKkl94N'>خرید از Blum</a>"
        ),
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True
    )

# ─── پاسخ‌دهی به پیام‌های منوی فارسی ─────────────────────────────
async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()

    if text.startswith("/"):
        return  # فرمان‌ها جداگانه هندل می‌شوند

    if text == "🛒 خرید اشتراک":
        await buy_cmd(update, context)

    elif text == "📤 ارسال رسید":
        await send_receipt_cmd(update, context)

    elif text == "⚖️ سؤال حقوقی":
        await update.message.reply_text(
            "سؤال خود را پس از دستور /ask بنویسید.\nمثال:\n<code>/ask قانون کار چیست؟</code>",
            parse_mode=ParseMode.HTML
        )

    elif text == "ℹ️ درباره توکن":
        await about_token(update, context)

    else:
        await update.message.reply_text("❓ دستور نامعتبر است. لطفاً از منو استفاده کنید.")

# ─── ثبت فرمان‌ها و هندلرها ─────────────────────────────────────────────
def register_handlers(app: Application) -> None:
    # فرمان‌های اصلی
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("buy", buy_cmd))
    # سایر هندلرها در ادامه اضافه می‌شوند (مثل send_receipt_cmd و ask_cmd)

    app.add_handler(CommandHandler("send_receipt", send_receipt_cmd))
    
    app.add_handler(
        MessageHandler(filters.PHOTO | (filters.TEXT & ~filters.COMMAND), handle_receipt),
        group=1,
    )
    app.add_handler(
        CallbackQueryHandler(callback_handler, pattern=r"^(approve|reject):\d+$"),
        group=0,
    )
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(CommandHandler("about_token", about_token))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, text_router),
        group=2,
    )

    init_db()  # ← پیش از ساخت application
# ─── تابع اصلی اجرای ربات ─────────────────────────────────────────────
BOT_TOKEN = os.getenv("BOT_TOKEN")

def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is missing in environment variables.")

    # ساخت اپلیکیشن
    application = Application.builder().token(BOT_TOKEN).build()

    # ثبت فرمان‌ها
    register_handlers(application)

    # اجرای ربات به‌صورت polling
    application.run_polling(allowed_updates=Update.ALL_TYPES)


# ─── اجرای مستقیم در حالت script ─────────────────────────────
if __name__ == "__main__":
    main()
