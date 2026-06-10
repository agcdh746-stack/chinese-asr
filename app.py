import os
import re
import json
import time
import uuid
import struct
import asyncio
import subprocess
import tempfile
import base64
from pathlib import Path

import httpx
from flask import Flask, request, jsonify, render_template, Response, stream_with_context, send_from_directory

app = Flask(__name__)

TEMP_DIR = os.environ.get("TEMP_DIR", "/tmp/asr")
KS_API = os.environ.get("KS_API", "http://localhost:5557")
KS_PROXY = os.environ.get("KS_PROXY", "").strip()
KS_COOKIE = os.environ.get("KS_COOKIE", "").strip()

Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)

PHOTO_ID_RE = re.compile(r"/(?:short-video|video|photo)/([A-Za-z0-9_-]+)")


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def parse_groq_keys(raw: str) -> list[str]:
    keys = []
    for part in re.split(r"[\n,]+", raw or ""):
        part = part.strip()
        if part and part not in keys:
            keys.append(part)
    return keys


def extract_photo_id(url: str) -> str | None:
    m = PHOTO_ID_RE.search(url or "")
    return m.group(1) if m else None


async def wait_for_ks_api(max_wait_ms: int = 10000) -> bool:
    start = time.time()
    while (time.time() - start) * 1000 < max_wait_ms:
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                resp = await client.get(f"{KS_API}/docs")
                if resp.status_code == 200:
                    return True
        except Exception:
            pass
        await asyncio.sleep(1.5)
    return False


async def resolve_ks_url(url: str) -> str:
    direct = extract_photo_id(url)
    if direct:
        return url
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        async with httpx.AsyncClient(timeout=12, follow_redirects=True, headers=headers) as client:
            resp = await client.get(url)
            return str(resp.url)
    except Exception:
        return url


async def get_ks_video_url_via_api(raw_url: str) -> tuple[str, dict]:
    payload = {"text": raw_url}
    if KS_COOKIE:
        payload["cookie"] = KS_COOKIE
    if KS_PROXY:
        payload["proxy"] = KS_PROXY

    last_error = None
    for attempt in range(4):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(f"{KS_API}/detail/", json=payload)
                resp.raise_for_status()
                body = resp.json()

            data = body.get("data")
            if not data:
                raise ValueError(f"KS-API: {body.get('message', 'no data')}")

            downloads = data.get("download", [])
            if isinstance(downloads, str):
                downloads = downloads.split()
            if not downloads:
                raise ValueError("KS-API: no download URL in response")
            return downloads[0], data
        except Exception as e:
            last_error = e
            if attempt < 3:
                await asyncio.sleep(1.5)
    raise ValueError(str(last_error or "KS-API failed"))


async def get_ks_video_url_via_graphql(raw_url: str) -> tuple[str, dict]:
    resolved_url = await resolve_ks_url(raw_url)
    photo_id = extract_photo_id(resolved_url)
    if not photo_id:
        raise ValueError(f"GraphQL: cannot extract photoId from {resolved_url}")

    payload = {
        "operationName": "visionVideoDetail",
        "variables": {"photoId": photo_id, "page": "detail"},
        "query": (
            "query visionVideoDetail($photoId: String, $page: String) { "
            "visionVideoDetail(photoId: $photoId, page: $page) { "
            "photo { id caption photoUrl duration } } }"
        ),
    }

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": f"https://www.kuaishou.com/short-video/{photo_id}",
        "Origin": "https://www.kuaishou.com",
        "Accept": "*/*",
    }
    if KS_COOKIE:
        headers["Cookie"] = KS_COOKIE

    async def _try_graphql(proxy: str | None):
        kwargs = {"timeout": 45}
        if proxy:
            kwargs["proxy"] = proxy
        async with httpx.AsyncClient(**kwargs) as client:
            resp = await client.post("https://www.kuaishou.com/graphql", headers=headers, json=payload)
            resp.raise_for_status()
            body = resp.json()
            video_url = body.get("data", {}).get("visionVideoDetail", {}).get("photo", {}).get("photoUrl")
            if not video_url:
                err = body.get("errors", [{}])[0].get("message", "no photoUrl")
                raise ValueError(f"GraphQL: {err}")
            photo = body.get("data", {}).get("visionVideoDetail", {}).get("photo", {})
            return video_url, {
                "photoId": photo.get("id") or photo_id,
                "caption": photo.get("caption") or photo_id,
                "duration": photo.get("duration") or 0,
                "resolved_url": resolved_url,
            }

    try:
        return await _try_graphql(None)
    except Exception:
        if KS_PROXY:
            return await _try_graphql(KS_PROXY)
        raise


async def get_ks_video_url(raw_url: str) -> tuple[str, dict, str]:
    api_ready = await wait_for_ks_api(12000)
    if api_ready:
        try:
            video_url, meta = await get_ks_video_url_via_api(raw_url)
            return video_url, meta, "ks-downloader-api"
        except Exception:
            pass
    video_url, meta = await get_ks_video_url_via_graphql(raw_url)
    return video_url, meta, "graphql-fallback"


async def download_video(video_url: str, out_path: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.kuaishou.com/",
    }
    async with httpx.AsyncClient(timeout=180, follow_redirects=True) as client:
        async with client.stream("GET", video_url, headers=headers) as resp:
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                async for chunk in resp.aiter_bytes(1024 * 64):
                    f.write(chunk)


def extract_audio(video_path: str, mp3_path: str):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-b:a",
            "128k",
            "-vn",
            mp3_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


async def groq_transcribe(mp3_path: str, groq_keys: list[str], language: str = "zh") -> dict:
    if not groq_keys:
        raise ValueError("No Groq API key provided")

    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    retryable_statuses = {402, 420, 429}
    last_error = None

    for idx, groq_key in enumerate(groq_keys, start=1):
        headers = {"Authorization": f"Bearer {groq_key}"}
        # ✅ একসাথে segment ও word দুটোই চাও
        data = {
            "model": "whisper-large-v3",
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["segment", "word"],
        }
        if language and language != "auto":
            data["language"] = language

        try:
            async with httpx.AsyncClient(timeout=180) as client:
                with open(mp3_path, "rb") as f:
                    resp = await client.post(
                        url,
                        headers=headers,
                        files={"file": ("audio.mp3", f, "audio/mpeg")},
                        data=data,
                    )

            if resp.status_code in retryable_statuses:
                last_error = ValueError(f"Groq key {idx} quota/rate-limited ({resp.status_code})")
                continue
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            if status in retryable_statuses or (status is not None and status >= 500):
                last_error = ValueError(f"Groq key {idx} failed with status {status}")
                continue
            raise
        except Exception as e:
            last_error = e
            if idx < len(groq_keys):
                continue
            raise

    raise ValueError(str(last_error or "All Groq keys failed"))


def has_chinese(text: str) -> bool:
    return any('\u4e00' <= c <= '\u9fff' for c in text)


def split_long_segments(segments: list, max_dur: float = 8.0) -> list:
    """Split segments longer than max_dur into equal time chunks."""
    result = []
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        text = seg["text"]
        dur = end - start
        if dur <= max_dur:
            result.append(seg)
            continue
        n = max(2, int(dur / max_dur) + 1)
        words = text.split()
        time_step = dur / n
        chunk_size = max(1, len(words) // n)
        for i in range(n):
            t_start = start + i * time_step
            t_end = start + (i + 1) * time_step if i < n - 1 else end
            if i < n - 1:
                chunk_words = words[i * chunk_size:(i + 1) * chunk_size]
            else:
                chunk_words = words[i * chunk_size:]
            chunk_text = " ".join(chunk_words).strip()
            if chunk_text:
                result.append({"start": round(t_start, 3), "end": round(t_end, 3), "text": chunk_text})
    return result


def transcribe_stream(url: str, groq_keys_raw: str, language: str = "zh"):
    job_id = f"asr_{int(time.time())}"
    video_path = os.path.join(TEMP_DIR, f"{job_id}.mp4")
    mp3_path = os.path.join(TEMP_DIR, f"{job_id}.mp3")
    groq_keys = parse_groq_keys(groq_keys_raw)

    try:
        yield sse("log", {"msg": "⏳ Getting video URL..."})
        video_url, meta, strategy = asyncio.run(get_ks_video_url(url))
        caption = (meta or {}).get("caption") or (meta or {}).get("photoId") or "Kuaishou"
        yield sse("log", {"msg": f"✅ Video URL found via {strategy}"})
        yield sse("log", {"msg": f"🎬 Source: {caption}"})

        yield sse("log", {"msg": "⬇ Downloading video..."})
        asyncio.run(download_video(video_url, video_path))
        size_mb = os.path.getsize(video_path) / 1024 / 1024
        yield sse("log", {"msg": f"✅ Downloaded ({size_mb:.1f} MB)"})

        yield sse("log", {"msg": "🔊 Extracting audio (ffmpeg)..."})
        extract_audio(video_path, mp3_path)
        yield sse("log", {"msg": "✅ Audio extracted"})

        yield sse("log", {"msg": f"🤖 Sending to Groq Whisper ({language})..."})
        result = asyncio.run(groq_transcribe(mp3_path, groq_keys, language))
        yield sse("log", {"msg": f"✅ Transcription done! ({result.get('duration', 0):.1f}s audio)"})

        # ✅ ডিফল্ট সেগমেন্ট ব্যবহার করো (pause‑ভিত্তিক, সব ভাষায় নির্ভুল)
        segments = []
        raw_segments = result.get("segments")
        if raw_segments:
            for seg in raw_segments:
                segments.append({
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": (seg.get("text") or "").strip(),
                })
        else:
            # fallback (সাধারণত দরকার হয় না)
            words = result.get("words", [])
            current_words = []
            current_start = None
            for w in words:
                word_text = w.get("word", "").strip()
                if not word_text:
                    continue
                if current_start is None:
                    current_start = w["start"]
                current_words.append(word_text)
                if word_text[-1] in '।.?!\n':
                    joined = "".join(current_words) if has_chinese(word_text) else " ".join(current_words)
                    segments.append({
                        "start": round(current_start, 3),
                        "end": round(w["end"], 3),
                        "text": joined
                    })
                    current_words = []
                    current_start = None
            if current_words:
                joined = "".join(current_words) if has_chinese(words[-1]["word"]) else " ".join(current_words)
                segments.append({
                    "start": round(current_start, 3),
                    "end": round(words[-1]["end"], 3),
                    "text": joined
                })

        segments = split_long_segments(segments, max_dur=8.0)

        yield sse(
            "done",
            {
                "text": result.get("text", ""),
                "segments": segments,
                "duration": result.get("duration", 0),
                "strategy": strategy,
                "language": language,
            },
        )

    except Exception as e:
        yield sse("error", {"msg": str(e)})
    finally:
        for p in [video_path, mp3_path]:
            try:
                os.unlink(p)
            except Exception:
                pass


# ──────────────────────── edge‑tts CLI (নির্ভরযোগ্য) ────────────────────────
@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    voice = data.get("voice", "bn-IN-TanishaaNeural")
    pitch = data.get("pitch", "-5%")
    rate  = data.get("rate", "+12%")

    if not text:
        return jsonify({"error": "text required"}), 400

    mp3_path = None
    wav_path = None
    try:
        fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)

        subprocess.run([
            "edge-tts",
            "--voice", voice,
            "--text", text,
            "--pitch", pitch,
            "--rate", rate,
            "--write-media", mp3_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        subprocess.run([
            "ffmpeg", "-y",
            "-i", mp3_path,
            "-ar", "24000",
            "-ac", "1",
            "-sample_fmt", "s16",
            wav_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        with open(wav_path, "rb") as f:
            wav_bytes = f.read()
        pcm_bytes = wav_bytes[44:]
        pcm_b64 = base64.b64encode(pcm_bytes).decode()

        return jsonify({"pcm_b64": pcm_b64, "sample_rate": 24000})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        for p in (mp3_path, wav_path):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/transcribe")
def transcribe():
    url = request.args.get("url", "").strip()
    groq_keys = request.args.get("groq_keys", "").strip() or request.args.get("groq_key", "").strip()
    language = request.args.get("language", "zh").strip() or "zh"

    if not url:
        return jsonify({"error": "url required"}), 400
    if not groq_keys:
        return jsonify({"error": "groq_keys required"}), 400

    return Response(
        stream_with_context(transcribe_stream(url, groq_keys, language)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/dub", methods=["POST"])
def dub():
    data = request.get_json(force=True)
    video_url = (data.get("video_url") or "").strip()
    segments  = data.get("segments", [])

    if not video_url:
        return jsonify({"error": "video_url required"}), 400
    if not segments:
        return jsonify({"error": "segments required"}), 400

    job_id   = f"dub_{uuid.uuid4().hex[:8]}"
    job_dir  = os.path.join(TEMP_DIR, job_id)
    Path(job_dir).mkdir(parents=True, exist_ok=True)

    video_path = os.path.join(job_dir, "original.mp4")
    out_path   = os.path.join(job_dir, "dubbed.mp4")

    try:
        video_dl_url, _, _ = asyncio.run(get_ks_video_url(video_url))
        asyncio.run(download_video(video_dl_url, video_path))

        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
            capture_output=True, text=True, check=True
        )
        duration = float(json.loads(probe.stdout)["format"]["duration"])

        base_audio = os.path.join(job_dir, "base.wav")
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"anullsrc=r=24000:cl=mono",
            "-t", str(duration), base_audio
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        seg_files = []
        for i, seg in enumerate(segments):
            start    = float(seg["start"])
            end      = float(seg["end"])
            pcm_b64  = seg.get("pcm_b64", "")
            sr       = int(seg.get("sample_rate", 24000))
            target_dur = max(0.2, end - start)

            if not pcm_b64:
                continue

            pcm_bytes = base64.b64decode(pcm_b64)

            num_channels   = 1
            bits_per_sample = 16
            data_size      = len(pcm_bytes)
            byte_rate      = sr * num_channels * (bits_per_sample // 8)
            block_align    = num_channels * (bits_per_sample // 8)
            chunk_size     = 36 + data_size
            header = struct.pack(
                "<4sI4s4sIHHIIHH4sI",
                b"RIFF", chunk_size, b"WAVE", b"fmt ", 16, 1,
                num_channels, sr, byte_rate, block_align, bits_per_sample,
                b"data", data_size
            )
            raw_wav = os.path.join(job_dir, f"seg_{i}_raw.wav")
            with open(raw_wav, "wb") as f:
                f.write(header + pcm_bytes)

            probe2 = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_streams", raw_wav],
                capture_output=True, text=True
            )
            try:
                tts_dur = float(json.loads(probe2.stdout)["streams"][0]["duration"])
            except Exception:
                tts_dur = target_dur

            ratio = tts_dur / target_dur
            ratio = max(0.5, min(2.0, ratio))

            atempo_filters = []
            r = ratio
            while r > 2.0:
                atempo_filters.append("atempo=2.0")
                r /= 2.0
            while r < 0.5:
                atempo_filters.append("atempo=0.5")
                r /= 0.5
            atempo_filters.append(f"atempo={r:.4f}")
            af = ",".join(atempo_filters)

            fit_wav = os.path.join(job_dir, f"seg_{i}_fit.wav")
            subprocess.run([
                "ffmpeg", "-y", "-i", raw_wav,
                "-af", af, fit_wav
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            seg_files.append({"path": fit_wav, "start": start})

        if seg_files:
            filter_parts = []
            inputs = ["-i", base_audio]
            for idx, sf in enumerate(seg_files):
                inputs += ["-i", sf["path"]]
                delay_ms = int(sf["start"] * 1000)
                filter_parts.append(f"[{idx+1}]adelay={delay_ms}|{delay_ms}[d{idx}]")

            mixed_labels = "[0]" + "".join(f"[d{i}]" for i in range(len(seg_files)))
            filter_parts.append(f"{mixed_labels}amix=inputs={len(seg_files)+1}:normalize=0[aout]")
            filter_str = ";".join(filter_parts)

            mixed_audio = os.path.join(job_dir, "mixed.wav")
            subprocess.run(
                ["ffmpeg", "-y"] + inputs +
                ["-filter_complex", filter_str, "-map", "[aout]", mixed_audio],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        else:
            mixed_audio = base_audio

        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", mixed_audio,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "128k",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest", out_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        video_serve_url = f"/dub_video/{job_id}/dubbed.mp4"
        return jsonify({"video_url": video_serve_url, "job_id": job_id})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/dub_video/<job_id>/<filename>")
def serve_dub_video(job_id, filename):
    job_dir = os.path.join(TEMP_DIR, job_id)
    return send_from_directory(job_dir, filename)


@app.route("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
