FROM python:3.12-slim

# ─── System deps ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
  ffmpeg curl wget \
  && rm -rf /var/lib/apt/lists/*

# ─── KS-Downloader deps ───────────────────────────────────────────
RUN pip install --no-cache-dir \
  httpx[socks] aiofiles aiosqlite lxml pyyaml rich uvicorn fastapi emoji

# ─── Flask deps ───────────────────────────────────────────────────
RUN pip install --no-cache-dir flask httpx

# ─── App ──────────────────────────────────────────────────────────
WORKDIR /app
COPY ks-downloader ./ks-downloader
COPY app.py ./
COPY templates ./templates

RUN mkdir -p /tmp/asr /app/ks-downloader/Volume

ENV PORT=3000 \
  TEMP_DIR=/tmp/asr \
  PYTHONUNBUFFERED=1

EXPOSE 3000

RUN printf '#!/bin/sh\n\
echo "[startup] Starting KS-Downloader API on :5557..."\n\
cd /app/ks-downloader && PYTHONPATH=/app/ks-downloader python main.py api --host 0.0.0.0 --port 5557 >> /tmp/ks-api.log 2>&1 &\n\
KS_PID=$!\n\
echo "[startup] KS-Downloader PID=$KS_PID"\n\
sleep 8\n\
echo "[startup] KS-API log:"\n\
cat /tmp/ks-api.log\n\
echo "[startup] Starting Flask on :${PORT}"\n\
exec python /app/app.py\n' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
