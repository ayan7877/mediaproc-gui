FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install FFmpeg + image libs
RUN apt-get update -qq && \
    apt-get install -y -q --no-install-recommends \
        ffmpeg \
        libmagic1 \
        libjpeg62-turbo \
        libpng16-16 \
        libwebp7 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY static/ static/

RUN mkdir -p /tmp/mediaproc_uploads /tmp/mediaproc_output

EXPOSE 8000

CMD gunicorn --bind "0.0.0.0:${PORT:-8000}" --workers 2 --timeout 300 --keep-alive 5 server:app
