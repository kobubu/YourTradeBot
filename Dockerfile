# Lightweight CPU image
FROM python:3.10-slim

# System deps (some libs used by numpy/pandas/statsmodels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libatlas-base-dev \
    gfortran \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run the bot
CMD ["python", "bot.py"]
