###############################################
# RebLCbot – Secure, Whisper-Ready Dockerfile #
###############################################

# 1. پایه: Python سبک برای حجم کمتر
FROM python:3.11-slim

# 2. نصب ابزارهای سیستمی موردنیاز: git + ffmpeg + build tools
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. تنظیم پوشه کاری داخل کانتینر
WORKDIR /app

# 4. غیرفعال کردن .pyc و بافر stdout/stderr برای دیباگ راحت‌تر
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 5. کپی requirements.txt و نصب پکیج‌های پایتون (استفاده از لایه cache)
COPY requirements.txt .

# 6. ارتقای pip و نصب پکیج‌ها از requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 7. کپی کل پروژه
COPY . .

# 8. اجرای ربات
CMD ["python", "bot.py"]
