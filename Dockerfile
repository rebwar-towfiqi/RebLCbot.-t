###############################################
# RebLawBot – Dockerfile                      #
###############################################
# Base image: official slim Python (small footprint)
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code
COPY . .

# Expose no ports – the bot connects outward via HTTPS long‑polling

# Default command: run the Telegram bot
CMD ["python", "bot.py"]
