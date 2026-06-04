import os, json, subprocess, tempfile, asyncio, httpx, time
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from pathlib import Path

app = Flask(__name__)

TEMP_DIR  = os.environ.get("TEMP_DIR", "/tmp/asr")
KS_API    = "http://localhost:5557"
KS_PROXY  = os.environ.get("KS_PROXY", "")

Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)

# ── SSE helper ────────────────────────────────────────────────────
def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

# ── KS-Downloader: video URL বের করো ─────────────────────────────
async def get_ks_video_url(page_url: str) -> str:
    payload = {
        "text":  page_url,
        "proxy": KS_PROXY or "socks5://127.0.0.1:10808",
    }
    for attempt in range(15):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(f"{KS_API}/detail/", json=payload)
                resp.raise_for_status()
                data = resp.json()
            break
        except Exception as e:
            if attempt < 14:
                await asyncio.sleep(2)
                continue
            raise ValueError(f"KS-API not available: {e}")

    if not data.get("data"):
        raise ValueError(f"KS-API: {data.get('message', 'no data')}")

    downloads = data["data"].get("download", [])
    if isinstance(downloads, str):
        downloads = downloads.split()
    if not downloads:
        raise ValueError("No download URL in KS response")
    return downloads[0]

    if not data.get("data"):
        raise ValueError(f"KS-API: {data.get('message', 'no data')}")

    downloads = data["data"].get("download", [])
    if isinstance(downloads, str):
        downloads = downloads.split()
    if not downloads:
        raise ValueError("No download URL in KS response")
    return downloads[0]

# ── Video download ────────────────────────────────────────────────
async def download_video(video_url: str, out_path: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.kuaishou.com/",
    }
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        async with client.stream("GET", video_url, headers=headers) as resp:
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                async for chunk in resp.aiter_bytes(8192):
                    f.write(chunk)

# ── FFmpeg: audio extract ─────────────────────────────────────────
def extract_audio(video_path: str, mp3_path: str):
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "16000", "-ac", "1", "-b:a", "128k",
        "-vn", mp3_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ── Groq Whisper: transcribe ──────────────────────────────────────
async def groq_transcribe(mp3_path: str, groq_key: str) -> dict:
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {groq_key}"}
    async with httpx.AsyncClient(timeout=120) as client:
        with open(mp3_path, "rb") as f:
            resp = await client.post(url, headers=headers, files={
                "file": ("audio.mp3", f, "audio/mpeg"),
            }, data={
                "model": "whisper-large-v3",
                "language": "zh",
                "response_format": "verbose_json",
                "timestamp_granularities[]": "segment",
            })
        resp.raise_for_status()
        return resp.json()

# ── Main SSE pipeline ─────────────────────────────────────────────
def transcribe_stream(url: str, groq_key: str):
    job_id    = f"asr_{int(time.time())}"
    video_path = os.path.join(TEMP_DIR, f"{job_id}.mp4")
    mp3_path   = os.path.join(TEMP_DIR, f"{job_id}.mp3")

    try:
        yield sse("log", {"msg": "⏳ Getting video URL from KS-Downloader..."})
        video_url = asyncio.run(get_ks_video_url(url))
        yield sse("log", {"msg": "✅ Video URL found"})

        yield sse("log", {"msg": "⬇ Downloading video..."})
        asyncio.run(download_video(video_url, video_path))
        size_mb = os.path.getsize(video_path) / 1024 / 1024
        yield sse("log", {"msg": f"✅ Downloaded ({size_mb:.1f} MB)"})

        yield sse("log", {"msg": "🔊 Extracting audio (ffmpeg)..."})
        extract_audio(video_path, mp3_path)
        yield sse("log", {"msg": "✅ Audio extracted"})

        yield sse("log", {"msg": "🤖 Sending to Groq Whisper..."})
        result = asyncio.run(groq_transcribe(mp3_path, groq_key))
        yield sse("log", {"msg": f"✅ Transcription done! ({result.get('duration', 0):.1f}s audio)"})

        # Format segments
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": seg["start"],
                "end":   seg["end"],
                "text":  seg["text"].strip(),
            })

        yield sse("done", {
            "text":     result.get("text", ""),
            "segments": segments,
            "duration": result.get("duration", 0),
        })

    except Exception as e:
        yield sse("error", {"msg": str(e)})

    finally:
        for p in [video_path, mp3_path]:
            try: os.unlink(p)
            except: pass

# ── Routes ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe")
def transcribe():
    url      = request.args.get("url", "").strip()
    groq_key = request.args.get("groq_key", "").strip()

    if not url:
        return jsonify({"error": "url required"}), 400
    if not groq_key:
        return jsonify({"error": "groq_key required"}), 400

    return Response(
        stream_with_context(transcribe_stream(url, groq_key)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )

@app.route("/health")
def health():
    return jsonify({"ok": True})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
