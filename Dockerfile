FROM python:3.12-slim

# ─── System deps ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
  ffmpeg curl wget unzip ca-certificates git \
  && rm -rf /var/lib/apt/lists/*

# ─── Python deps ──────────────────────────────────────────────────
RUN pip install --no-cache-dir \
  flask httpx[socks] edge-tts \
  aiofiles aiosqlite lxml pyyaml rich uvicorn fastapi emoji \
  "yt-dlp[default]" instaloader \
  && yt-dlp --version \
  && instaloader --version

# ─── Deno (YouTube JS challenge bypass) ───────────────────────────
RUN curl -fsSL https://deno.land/install.sh | sh \
  && mv /root/.deno/bin/deno /usr/local/bin/deno \
  && deno --version

# ─── Xray-core ────────────────────────────────────────────────────
RUN ARCH=$(dpkg --print-architecture) \
  && if [ "$ARCH" = "amd64" ]; then XARCH="64"; else XARCH="arm64-v8a"; fi \
  && wget -q "https://github.com/XTLS/Xray-core/releases/latest/download/Xray-linux-${XARCH}.zip" -O /tmp/xray.zip \
  && unzip -q /tmp/xray.zip -d /tmp/xray \
  && mv /tmp/xray/xray /usr/local/bin/xray \
  && chmod +x /usr/local/bin/xray \
  && rm -rf /tmp/xray.zip /tmp/xray \
  && xray version || true

# ─── App ──────────────────────────────────────────────────────────
WORKDIR /app
COPY ks-downloader ./ks-downloader
COPY app.py ./
COPY scenevideo.py ./
COPY templates ./templates

RUN mkdir -p /tmp/asr /app/data/cookies /app/data /app/ks-downloader/Volume

ENV PORT=3000 \
  TEMP_DIR=/tmp/asr \
  COOKIES_FILE=/app/data/cookies/cookies.txt \
  CONFIG_FILE=/app/data/config.json \
  XRAY_BIN=/usr/local/bin/xray \
  XRAY_CONFIG=/app/data/xray-config.json \
  PYTHONUNBUFFERED=1

EXPOSE 3000

RUN printf '#!/bin/sh\n\
echo "[startup] Starting KS-Downloader API on :5557..."\n\
cd /app/ks-downloader && PYTHONPATH=/app/ks-downloader python main.py api --host 0.0.0.0 --port 5557 >> /tmp/ks-api.log 2>&1 &\n\
KS_PID=$!\n\
echo "[startup] KS-Downloader PID=$KS_PID"\n\
i=0\n\
while [ $i -lt 30 ]; do\n\
  if curl -sf http://localhost:5557/docs > /dev/null 2>&1; then\n\
    echo "[startup] KS-API ready after ${i}s"\n\
    break\n\
  fi\n\
  sleep 1\n\
  i=$((i+1))\n\
done\n\
echo "[startup] KS-API log:"\n\
cat /tmp/ks-api.log\n\
echo "[startup] Starting Flask on :${PORT}"\n\
exec python /app/app.py\n' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
