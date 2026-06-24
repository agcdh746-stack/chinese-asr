# -*- coding: utf-8 -*-
"""
scenevideo.py — Hybrid Scene Video (v2)
"""

import os
import re
import json
import math
import time
import uuid
import base64
import shutil
import subprocess
import threading
import wave
import urllib.parse

import httpx
from flask import request, jsonify, Response, stream_with_context, send_file

# ───────────────────────── config ───────────────────────────────────────

SCENEVIDEO_JOBS_DIR = os.path.join(os.getcwd(), "scenevideo_jobs")
os.makedirs(SCENEVIDEO_JOBS_DIR, exist_ok=True)

TTS_MODEL  = "gemini-2.5-flash-preview-tts"
TEXT_MODEL = "gemini-2.5-flash"

TTS_ENDPOINT  = f"https://generativelanguage.googleapis.com/v1beta/models/{TTS_MODEL}:generateContent"
TEXT_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{TEXT_MODEL}:generateContent"

STREAMELEMENTS_TTS_URL = "https://api.streamelements.com/kappa/v2/speech"
STREAMELEMENTS_VOICE   = "Brian"

POLLINATIONS_MODEL = "flux"

PEXELS_VIDEO_URL   = "https://api.pexels.com/videos/search"
PEXELS_IMAGE_URL   = "https://api.pexels.com/v1/search"
PIXABAY_VIDEO_URL  = "https://pixabay.com/api/videos/"
PIXABAY_IMAGE_URL  = "https://pixabay.com/api/"
UNSPLASH_IMAGE_URL = "https://api.unsplash.com/search/photos"

MUSIC_TRACKS = {
    "epic":     "https://storage.googleapis.com/aistudio-hosting/music/cinematic-documentary.mp3",
    "upbeat":   "https://storage.googleapis.com/aistudio-hosting/music/corporate-upbeat.mp3",
    "relaxing": "https://storage.googleapis.com/aistudio-hosting/music/relaxing-ambient.mp3",
}

MIN_CLIP_DURATION = 2.0
TRANSITION_DUR    = 0.5
CLIP_INTERVAL     = 4.0   # প্রতি ৪ সেকেন্ডে নতুন visual

RESOLUTIONS = {
    "1920x1080": (1920, 1080),
    "1280x720":  (1280, 720),
    "720x1280":  (720, 1280),
    "1080x1920": (1080, 1920),
}

def _res(job):
    return RESOLUTIONS.get(job.get("resolution", "720x1280"), (720, 1280))

# ───────────────────────── job helpers ──────────────────────────────────

def _job_path(job_id):
    return os.path.join(SCENEVIDEO_JOBS_DIR, f"{job_id}.json")

def load_job(job_id):
    p = _job_path(job_id)
    if not os.path.exists(p): return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_job(job):
    with open(_job_path(job["id"]), "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)

def job_log(job, msg, lvl="info"):
    job.setdefault("logs", []).append({"t": time.time(), "lvl": lvl, "msg": msg})
    print(f"[scenevideo:{job['id']}] {msg}", flush=True)
    save_job(job)


# ───────────────────────── 1. Script + Scene Breakdown ──────────────────

SCENE_SYSTEM_PROMPT = """তুমি একজন বাংলা নিউজ ভিডিও স্ক্রিপ্ট রাইটার ও visual director।
ইউজার একটা বাস্তব ঘটনার বিবরণ দেবে। তোমার কাজ:

1. ঘটনাটাকে ৪-৮টা scene-এ ভাগ করো (কালানুক্রমিকভাবে)।
2. প্রতিটা scene-এ:
   - narration: বাংলা ১-২ বাক্য (সাংবাদিকতার ভাষায়, নিরপেক্ষ)
   - search_queries: Pexels stock footage search-এর জন্য ৩টা English query — hierarchy অনুযায়ী:
       q1 (specific):  ২-৩ word, NOUN first, scene-এর exact visual subject
       q2 (broader):   ২ word, broader concept
       q3 (generic):   ১-২ word, সবসময় Pexels-এ পাওয়া যাবে এমন generic visual
     VISUAL_STYLE_NOTE
     CRITICAL RULES:
       - সম্পূর্ণ English — কোনো বাংলা শব্দ নিষিদ্ধ
       - Noun দিয়ে শুরু করো (adjective দিয়ে না)
       - কোনো proper noun, country name, brand name নয়
       - q1 example: "fish floating river", "man bathing water", "pigeon street"
       - q2 example: "river water", "bird ground", "kitchen cooking"  
       - q3 example: "nature water", "city street", "food cooking"
   - ai_prompt: Pollinations-এর জন্য detailed English visual description।
     Style: "illustrated editorial news-style digital painting"
     No violence/gore/blood/realistic face।

JSON ফরম্যাট (শুধু JSON, অন্য কিছু না):
{
  "title": "ভিডিওর শিরোনাম",
  "scenes": [
    {
      "narration": "বাংলা বাক্য",
      "search_queries": {"q1": "specific query", "q2": "broader query", "q3": "generic query"},
      "ai_prompt": "English AI image description"
    }
  ]
}
"""

def generate_script_and_scenes(event_description, api_key, visual_style="realistic"):
    if visual_style == "cartoon":
        style_note = (
            "CARTOON mode — search_queries rules:\n"
            "  q1/q2/q3 সব English only, noun first।\n"
            "  q1 example: 'fish river cartoon', 'cooking pot animated'\n"
            "  q2 example: 'fish water', 'cooking food'\n"
            "  q3 example: 'nature', 'kitchen'\n"
            "  Note: cartoon/animated word যোগ করা optional, noun আগে।"
        )
    else:
        style_note = (
            "REALISTIC mode — search_queries rules:\n"
            "  q1/q2/q3 সব English only, noun first, literal descriptive।\n"
            "  q1 example: 'fish floating river', 'man bathing water', 'pigeon street ground'\n"
            "  q2 example: 'river water', 'bird ground', 'street crowd'\n"
            "  q3 example: 'nature water', 'city', 'outdoor'\n"
            "  No adjective-first queries like 'dead fish' or 'empty packet'।"
        )
    prompt = (
        SCENE_SYSTEM_PROMPT.replace("VISUAL_STYLE_NOTE", style_note)
        + f"\n\nঘটনার বিবরণ:\n{event_description}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.7},
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    resp = None
    for attempt in range(5):
        with httpx.Client(timeout=60) as client:
            resp = client.post(TEXT_ENDPOINT, headers=headers, json=payload)
        if resp.status_code == 200:
            break
        if resp.status_code in (503, 429, 500):
            wait = 10 * (attempt + 1)
            print(f"[script] {resp.status_code} — {wait}s retry ({attempt+1}/5)...", flush=True)
            time.sleep(wait)
        else:
            break
    if resp.status_code != 200:
        raise RuntimeError(f"script generation failed ({resp.status_code}): {resp.text[:500]}")
    data = resp.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"unexpected script response: {data}") from e
    parsed = json.loads(text)
    if "scenes" not in parsed or not parsed["scenes"]:
        raise RuntimeError("কোনো scene পাওয়া যায়নি")
    return parsed


# ───────────────────────── 2. TTS ───────────────────────────────────────

BREAK_TAG = '<break time="700ms"/>'

def synthesize_full_audio_gemini(scenes, api_key, voice, out_wav_path):
    full_text = f" {BREAK_TAG} ".join(s["narration"].strip() for s in scenes)
    payload = {
        "contents": [{"parts": [{"text": full_text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    resp = None
    for attempt in range(4):
        with httpx.Client(timeout=120) as client:
            resp = client.post(TTS_ENDPOINT, headers=headers, json=payload)
        if resp.status_code == 200:
            break
        if resp.status_code in (503, 429, 500):
            wait = 15 * (attempt + 1)
            print(f"[tts] {resp.status_code} — {wait}s retry ({attempt+1}/4)...", flush=True)
            time.sleep(wait)
        else:
            break
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini TTS failed ({resp.status_code}): {resp.text[:500]}")
    data = resp.json()
    pcm_b64 = None
    try:
        for part in data["candidates"][0]["content"]["parts"]:
            if "inlineData" in part and part["inlineData"].get("data"):
                pcm_b64 = part["inlineData"]["data"]; break
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"unexpected TTS response: {data}") from e
    if not pcm_b64:
        raise RuntimeError("TTS response-এ কোনো audio data নেই")
    pcm_data = base64.b64decode(pcm_b64)
    with wave.open(out_wav_path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000)
        wf.writeframes(pcm_data)
    return out_wav_path


def _streamelements_tts_chunk(text, voice=STREAMELEMENTS_VOICE):
    url = f"{STREAMELEMENTS_TTS_URL}?voice={voice}&text={urllib.parse.quote(text)}"
    with httpx.Client(timeout=30) as client:
        resp = client.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"StreamElements TTS failed ({resp.status_code})")
    return resp.content


def synthesize_full_audio_streamelements(scenes, out_mp3_path, voice=STREAMELEMENTS_VOICE):
    tmp_dir = out_mp3_path + "_parts"
    os.makedirs(tmp_dir, exist_ok=True)
    part_files = []

    for idx, scene in enumerate(scenes):
        text = scene["narration"].strip()
        chunks = [text[i:i+200] for i in range(0, len(text), 200)]
        chunk_files = []
        for ci, chunk in enumerate(chunks):
            part_path = os.path.join(tmp_dir, f"scene_{idx:02d}_chunk_{ci:02d}.mp3")
            for attempt in range(3):
                try:
                    data = _streamelements_tts_chunk(chunk, voice)
                    with open(part_path, "wb") as f:
                        f.write(data)
                    chunk_files.append(part_path)
                    break
                except Exception as e:
                    if attempt == 2: raise
                    time.sleep(1 * (attempt + 1))
        scene_mp3 = os.path.join(tmp_dir, f"scene_{idx:02d}.mp3")
        if len(chunk_files) == 1:
            shutil.copy(chunk_files[0], scene_mp3)
        else:
            lst = scene_mp3 + ".txt"
            with open(lst, "w") as f:
                for p in chunk_files: f.write(f"file '{p}'\n")
            subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                            "-i", lst, "-c", "copy", scene_mp3], capture_output=True)
        part_files.append(scene_mp3)
        if idx < len(scenes) - 1:
            silence_path = os.path.join(tmp_dir, f"silence_{idx:02d}.mp3")
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                "-t", "0.7", "-q:a", "9", "-acodec", "libmp3lame", silence_path
            ], capture_output=True)
            part_files.append(silence_path)

    lst_path = out_mp3_path + "_list.txt"
    with open(lst_path, "w") as f:
        for p in part_files: f.write(f"file '{p}'\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", lst_path, "-c", "copy", out_mp3_path], capture_output=True, check=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_mp3_path


def synthesize_audio(job, scenes, job_dir):
    tts_key = job.get("tts_api_key", "").strip()
    voice   = job.get("voice", "Charon")
    raw_wav = os.path.join(job_dir, "full_audio.wav")

    if tts_key:
        try:
            job_log(job, "🎙️  Gemini TTS দিয়ে audio বানানো হচ্ছে...")
            synthesize_full_audio_gemini(scenes, tts_key, voice, raw_wav)
            job_log(job, "✅ Gemini TTS সম্পন্ন")
            return raw_wav
        except Exception as e:
            job_log(job, f"⚠️  Gemini TTS ব্যর্থ ({e}) — StreamElements fallback...", "warn")

    job_log(job, "🎙️  StreamElements TTS (free fallback) দিয়ে audio বানানো হচ্ছে...")
    raw_mp3  = os.path.join(job_dir, "full_audio.mp3")
    se_voice = job.get("se_voice", STREAMELEMENTS_VOICE)
    synthesize_full_audio_streamelements(scenes, raw_mp3, se_voice)
    subprocess.run(["ffmpeg", "-y", "-i", raw_mp3, "-ar", "24000", "-ac", "1", raw_wav],
                   capture_output=True, check=True)
    job_log(job, "✅ StreamElements TTS সম্পন্ন")
    return raw_wav


# ───────────────────────── 3. Silence split ─────────────────────────────

def _ffprobe_duration(path):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", path]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    return float(out) if out else 0.0


def _run_silencedetect(wav_path, noise_db="-30dB", min_dur=0.3):
    cmd = ["ffmpeg", "-i", wav_path, "-af",
           f"silencedetect=noise={noise_db}:d={min_dur}", "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    starts = [float(m) for m in re.findall(r"silence_start:\s*([\d.]+)", result.stderr)]
    ends   = [float(m) for m in re.findall(r"silence_end:\s*([\d.]+)", result.stderr)]
    return list(zip(starts, ends))


def split_audio_by_silence(wav_path, expected_scene_count, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    total_dur = _ffprobe_duration(wav_path)
    attempts = [("-30dB", 0.3), ("-25dB", 0.2), ("-22dB", 0.15), ("-35dB", 0.4)]
    pairs = None
    for noise_db, min_dur in attempts:
        candidate = _run_silencedetect(wav_path, noise_db, min_dur)
        if len(candidate) == expected_scene_count - 1:
            pairs = candidate; break
        if pairs is None or abs(len(candidate)-(expected_scene_count-1)) < abs(len(pairs)-(expected_scene_count-1)):
            pairs = candidate
    if pairs is None: pairs = []

    cut_points = [0.0]
    for s_start, s_end in pairs:
        cut_points.append(s_start); cut_points.append(s_end)
    cut_points.append(total_dur)

    scenes_out = []
    for i in range(0, len(cut_points)-1, 2):
        s, e = cut_points[i], cut_points[i+1]
        if e - s >= 0.05: scenes_out.append((s, e))

    if len(scenes_out) != expected_scene_count:
        if not scenes_out:
            chunk = total_dur / expected_scene_count
            scenes_out = [(i*chunk, (i+1)*chunk) for i in range(expected_scene_count)]
        elif len(scenes_out) > expected_scene_count:
            while len(scenes_out) > expected_scene_count:
                mi = min(range(len(scenes_out)-1),
                    key=lambda i: (scenes_out[i][1]-scenes_out[i][0])+(scenes_out[i+1][1]-scenes_out[i+1][0]))
                scenes_out = scenes_out[:mi] + [(scenes_out[mi][0], scenes_out[mi+1][1])] + scenes_out[mi+2:]
        else:
            while len(scenes_out) < expected_scene_count:
                mx = max(range(len(scenes_out)), key=lambda i: scenes_out[i][1]-scenes_out[i][0])
                s, e = scenes_out[mx]; mid = (s+e)/2
                scenes_out = scenes_out[:mx] + [(s, mid), (mid, e)] + scenes_out[mx+1:]

    results = []
    for idx, (start, end) in enumerate(scenes_out):
        out_file = os.path.join(out_dir, f"scene_{idx:02d}.wav")
        subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-ss", str(start), "-to", str(end),
                        "-c", "copy", out_file], capture_output=True)
        results.append((idx, out_file, end-start))
    return results


# ───────────────────────── 4. Stock Media ───────────────────────────────

def _pixabay_video_type(visual_style):
    return "animation" if visual_style == "cartoon" else "film"

def _pixabay_image_type(visual_style):
    return "illustration" if visual_style == "cartoon" else "photo"


def _build_fallback_queries(query):
    queries = [query]
    words = query.split()
    if len(words) >= 2:
        queries.append(words[0])
        queries.append(" ".join(words[:2]))
        queries.append(words[-1])
    queries += ["news broadcast", "city street", "people walking", "nature landscape"]
    seen = set()
    return [q for q in queries if q and not (q in seen or seen.add(q))]


def _pexels_video_search_one(query, api_key, orientation, page=1):
    try:
        params = {"query": query, "per_page": 15, "page": page,
                  "orientation": orientation, "size": "medium"}
        r = httpx.get(PEXELS_VIDEO_URL, headers={"Authorization": api_key},
                      params=params, timeout=20)
        if r.status_code != 200: return []
        videos = r.json().get("videos", [])
        return [v for v in videos if v.get("duration", 0) >= MIN_CLIP_DURATION]
    except Exception as e:
        print(f"[pexels] video search error ({query}): {e}", flush=True)
        return []


def _pick_best_pexels_video(videos, orientation, w, h):
    if not videos: return None
    chosen = videos[0]
    files = chosen.get("video_files", [])
    if not files: return None
    if orientation == "portrait":
        candidates = [f for f in files if f.get("height", 0) >= f.get("width", 0)] or files
    else:
        candidates = [f for f in files if f.get("width", 0) >= f.get("height", 0)] or files

    def res_diff(f):
        return abs(f.get("width", 0) - w) + abs(f.get("height", 0) - h)

    best = sorted(candidates, key=res_diff)[0]
    return {"url": best["link"], "duration": chosen["duration"],
            "kind": "video", "source": "pexels"}


def _pexels_image_search_one(query, api_key, orientation):
    try:
        params = {"query": query, "per_page": 10, "orientation": orientation}
        r = httpx.get(PEXELS_IMAGE_URL, headers={"Authorization": api_key},
                      params=params, timeout=15)
        if r.status_code != 200: return None
        photos = r.json().get("photos", [])
        if not photos: return None
        src = photos[0].get("src", {})
        url = (src.get("portrait") if orientation == "portrait" else src.get("landscape")) \
              or src.get("large") or src.get("original")
        return {"url": url, "kind": "image", "source": "pexels"} if url else None
    except Exception as e:
        print(f"[pexels] image search error ({query}): {e}", flush=True)
        return None


def _clean_pexels_query(query):
    """Pexels-এ cartoon/animated/illustration word কাজ করে না — strip করো।
    বাংলা character থাকলে সেটাও সরাও।"""
    stop_words = {"cartoon", "animated", "animation", "illustration", "illustrated"}
    words = query.lower().split()
    cleaned = " ".join(w for w in words if w not in stop_words)
    # বাংলা unicode range: \u0980-\u09FF
    import unicodedata
    cleaned = " ".join(
        w for w in cleaned.split()
        if not any("\u0980" <= c <= "\u09FF" for c in w)
    )
    return cleaned.strip() or query  # সব বাদ গেলে original রাখো


def fetch_pexels_media(search_queries, pexels_key, w=720, h=1280, slot=0):
    """
    search_queries: dict {"q1": specific, "q2": broader, "q3": generic}
                    অথবা plain string (backward compat)
    3-level hierarchy দিয়ে search — specific → broader → generic
    """
    orientation = "portrait" if h > w else "landscape"
    page = (slot % 2) + 1  # page 1 or 2 rotate

    # search_queries normalize
    if isinstance(search_queries, str):
        q = _clean_pexels_query(search_queries)
        words = q.split()
        ordered_queries = [
            q,
            " ".join(words[:2]) if len(words) >= 2 else q,
            words[0] if words else q,
        ]
    else:
        ordered_queries = [
            _clean_pexels_query(search_queries.get("q1", "")),
            _clean_pexels_query(search_queries.get("q2", "")),
            _clean_pexels_query(search_queries.get("q3", "")),
        ]
        ordered_queries = [q for q in ordered_queries if q]

    # Video search — specific → broader → generic
    for q in ordered_queries:
        if not q: continue
        videos = _pexels_video_search_one(q, pexels_key, orientation, page=page)
        if not videos and page > 1:
            videos = _pexels_video_search_one(q, pexels_key, orientation, page=1)
        result = _pick_best_pexels_video(videos, orientation, w, h)
        if result:
            print(f"[pexels] ✅ video: q='{q}' dur={result['duration']}s", flush=True)
            return result
        print(f"[pexels] ❌ miss: q='{q}'", flush=True)

    # Image fallback — same hierarchy
    for q in ordered_queries:
        if not q: continue
        result = _pexels_image_search_one(q, pexels_key, orientation)
        if result:
            print(f"[pexels] ✅ image: q='{q}'", flush=True)
            return result

    return None


def _search_pixabay_video(query, api_key, visual_style):
    try:
        params = {"key": api_key, "q": query, "per_page": 10,
                  "video_type": _pixabay_video_type(visual_style), "safesearch": "true"}
        r = httpx.get(PIXABAY_VIDEO_URL, params=params, timeout=15)
        if r.status_code != 200: return None
        hits = r.json().get("hits", [])
        if not hits: return None
        valid = [h for h in hits if h.get("duration", 0) >= MIN_CLIP_DURATION] or hits
        chosen = valid[0]
        vids = chosen.get("videos", {})
        for q in ("medium", "small", "large"):
            v = vids.get(q)
            if v and v.get("url"):
                return {"url": v["url"], "duration": chosen["duration"],
                        "kind": "video", "source": "pixabay"}
        return None
    except Exception as e:
        print(f"[pixabay] video error: {e}", flush=True); return None


def _search_pixabay_image(query, api_key, visual_style):
    try:
        params = {"key": api_key, "q": query, "per_page": 5,
                  "image_type": _pixabay_image_type(visual_style),
                  "orientation": "vertical", "safesearch": "true"}
        r = httpx.get(PIXABAY_IMAGE_URL, params=params, timeout=15)
        if r.status_code != 200: return None
        hits = r.json().get("hits", [])
        if not hits: return None
        url = hits[0].get("largeImageURL") or hits[0].get("webformatURL")
        return {"url": url, "kind": "image", "source": "pixabay"} if url else None
    except Exception as e:
        print(f"[pixabay] image error: {e}", flush=True); return None


def _search_unsplash_image(query, api_key, visual_style):
    try:
        params = {"query": query, "per_page": 5, "orientation": "portrait"}
        r = httpx.get(UNSPLASH_IMAGE_URL,
                      headers={"Authorization": f"Client-ID {api_key}"},
                      params=params, timeout=15)
        if r.status_code != 200: return None
        results = r.json().get("results", [])
        if not results: return None
        urls = results[0].get("urls", {})
        url = urls.get("regular") or urls.get("full")
        return {"url": url, "kind": "image", "source": "unsplash"} if url else None
    except Exception as e:
        print(f"[unsplash] error: {e}", flush=True); return None


def download_media(url, out_path):
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        r = client.get(url)
    if r.status_code != 200:
        raise RuntimeError(f"download failed ({r.status_code}): {url}")
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path


# ───────────────────────── 5. AI Image ──────────────────────────────────

def generate_scene_image(prompt, out_path, w=720, h=1280):
    suffix = "correct human anatomy, two arms, two hands, no extra limbs, photorealistic, high quality"
    full_prompt = f"{prompt.strip()}, {suffix}"
    encoded = urllib.parse.quote(full_prompt)
    url = (f"https://image.pollinations.ai/prompt/{encoded}"
           f"?model={POLLINATIONS_MODEL}&width={w}&height={h}&nologo=true&enhance=true")
    with httpx.Client(timeout=120) as client:
        resp = client.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"Pollinations failed ({resp.status_code})")
    with open(out_path, "wb") as f:
        f.write(resp.content)
    return out_path


# ───────────────────────── 6. Per-scene media ───────────────────────────

def get_scene_media(job, scene, scene_idx, media_dir,
                    mode, pexels_key, pixabay_key, unsplash_key, visual_style, w, h):
    # search_queries dict নাও (নতুন format) অথবা stock_query (পুরানো fallback)
    search_queries = scene.get("search_queries") or scene.get("stock_query", scene.get("ai_prompt", ""))

    def try_pexels():
        if not pexels_key: return None
        slot   = scene.get("_slot", 0)
        result = fetch_pexels_media(search_queries, pexels_key, w, h, slot=slot)
        if not result: return None
        ext      = ".mp4" if result["kind"] == "video" else ".jpg"
        out_path = os.path.join(media_dir, f"scene_{scene_idx:02d}_pexels{ext}")
        try:
            download_media(result["url"], out_path)
            return {"type": result["kind"], "path": out_path,
                    "source": "pexels", "duration": result.get("duration", 0)}
        except Exception as e:
            job_log(job, f"⚠️  Pexels download error scene {scene_idx}: {e}", "warn")
            return None

    def try_pixabay():
        if not pixabay_key: return None
        result = _search_pixabay_video(query, pixabay_key, visual_style)
        if not result:
            result = _search_pixabay_image(query, pixabay_key, visual_style)
        if not result: return None
        ext      = ".mp4" if result["kind"] == "video" else ".jpg"
        out_path = os.path.join(media_dir, f"scene_{scene_idx:02d}_pixabay{ext}")
        try:
            download_media(result["url"], out_path)
            return {"type": result["kind"], "path": out_path,
                    "source": "pixabay", "duration": result.get("duration", 0)}
        except Exception as e:
            job_log(job, f"⚠️  Pixabay download error: {e}", "warn")
            return None

    def try_unsplash():
        if not unsplash_key: return None
        result = _search_unsplash_image(query, unsplash_key, visual_style)
        if not result: return None
        out_path = os.path.join(media_dir, f"scene_{scene_idx:02d}_unsplash.jpg")
        try:
            download_media(result["url"], out_path)
            return {"type": "image", "path": out_path, "source": "unsplash", "duration": 0}
        except Exception as e:
            job_log(job, f"⚠️  Unsplash download error: {e}", "warn")
            return None

    def try_stock():
        r = try_pexels()
        if r: return r
        return try_unsplash()

    def try_ai():
        out_path = os.path.join(media_dir, f"scene_{scene_idx:02d}_ai.png")
        for attempt in range(3):
            try:
                generate_scene_image(scene["ai_prompt"], out_path, w, h)
                return {"type": "image", "path": out_path, "source": "pollinations", "duration": 0}
            except Exception:
                time.sleep(3 * (attempt + 1))
        return None

    if mode == "ai":
        return try_ai() or {"type": "image", "path": None, "source": "failed", "duration": 0}

    if mode == "stock":
        r = try_stock()
        if r: return r
        job_log(job, f"⚠️  Stock miss scene {scene_idx}, AI fallback...", "warn")
        return try_ai() or {"type": "image", "path": None, "source": "failed", "duration": 0}

    if mode == "hybrid":
        stock_result = [None]; ai_result = [None]
        def _s(): stock_result[0] = try_stock()
        def _a(): ai_result[0] = try_ai()
        t1 = threading.Thread(target=_s)
        t2 = threading.Thread(target=_a)
        t1.start(); t2.start(); t1.join(); t2.join()
        if stock_result[0] and stock_result[0]["type"] == "video": return stock_result[0]
        if stock_result[0]: return stock_result[0]
        if ai_result[0]: return ai_result[0]
        return {"type": "image", "path": None, "source": "failed", "duration": 0}

    return {"type": "image", "path": None, "source": "unknown_mode", "duration": 0}


# ───────────────────────── 7. Clip Builder ──────────────────────────────

def _scale_filter(w, h):
    return (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1")


def _build_image_clip(image_path, audio_path, duration_sec, out_path, zoom_in, w, h):
    """Image → Ken Burns. PATCH: -shortest -t dur শেষে, input limiter নেই।"""
    scale_w = int(w * 1.1); scale_h = int(h * 1.1)
    if zoom_in:
        vf = f"scale={scale_w}:{scale_h},crop={w}:{h}:0:0,setsar=1"
    else:
        ox = (scale_w-w)//2; oy = (scale_h-h)//2
        vf = f"scale={scale_w}:{scale_h},crop={w}:{h}:{ox}:{oy},setsar=1"
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", image_path,
        "-i", audio_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-pix_fmt", "yuv420p", "-threads", "1",
        "-c:a", "aac", "-b:a", "96k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", "-t", str(duration_sec), out_path,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"image clip failed: {r.stderr[-300:]}")
    return out_path


def _prepare_video_segment(video_path, duration, out_path, w, h):
    """
    MoneyPrinterTurbo approach: video track ONLY (no audio) — exact duration।
    audio পরে আলাদাভাবে overlay হবে।
    """
    video_dur = _ffprobe_duration(video_path)
    if video_dur <= 0:
        raise RuntimeError(f"video duration পড়তে পারেনি: {video_path}")

    sf = _scale_filter(w, h)

    if video_dur >= duration:
        # যথেষ্ট লম্বা → ১/৩ position থেকে শুরু করো
        start_t = max(0.0, (video_dur - duration) / 3)
        cmd = ["ffmpeg", "-y",
               "-ss", str(start_t), "-t", str(duration),
               "-i", video_path,
               "-vf", sf,
               "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
               "-pix_fmt", "yuv420p", "-r", "30", "-threads", "1",
               "-an", out_path]  # audio strip — MoneyPrinterTurbo approach
    else:
        # ছোট video → loop করো
        loop_count = int(duration / video_dur) + 2
        cmd = ["ffmpeg", "-y",
               "-stream_loop", str(loop_count),
               "-i", video_path,
               "-t", str(duration),
               "-vf", sf,
               "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
               "-pix_fmt", "yuv420p", "-r", "30", "-threads", "1",
               "-an", out_path]

    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"video segment failed: {r.stderr[-400:]}")
    return out_path


def _prepare_image_segment(image_path, duration, out_path, zoom_in, w, h):
    """Image → Ken Burns video segment (no audio)"""
    scale_w = int(w * 1.1); scale_h = int(h * 1.1)
    if zoom_in:
        vf = f"scale={scale_w}:{scale_h},crop={w}:{h}:0:0,setsar=1"
    else:
        ox = (scale_w-w)//2; oy = (scale_h-h)//2
        vf = f"scale={scale_w}:{scale_h},crop={w}:{h}:{ox}:{oy},setsar=1"
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", image_path,
        "-t", str(duration),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-pix_fmt", "yuv420p", "-r", "30", "-threads", "1",
        "-an", out_path,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"image segment failed: {r.stderr[-300:]}")
    return out_path


def _build_video_clip(video_path, audio_path, audio_dur, out_path, w, h):
    """Legacy wrapper — video+audio একসাথে (image clip-এর জন্য)"""
    tmp_v = out_path + "_v.mp4"
    _prepare_video_segment(video_path, audio_dur, tmp_v, w, h)
    # audio mux
    cmd = ["ffmpeg", "-y",
           "-i", tmp_v, "-i", audio_path,
           "-t", str(audio_dur),
           "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "96k",
           out_path]
    r = subprocess.run(cmd, capture_output=True)
    if os.path.exists(tmp_v): os.remove(tmp_v)
    if r.returncode != 0:
        raise RuntimeError(f"video clip mux failed: {r.stderr[-300:]}")
    return out_path


def _find_best_segment(video_path, video_dur, need_dur):
    """PATCH: dead code সরানো হয়েছে। scene change density দিয়ে best window।"""
    if video_dur <= need_dur:
        return 0.0
    try:
        scene_cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", "select='gt(scene,0.1)',showinfo",
            "-vsync", "vfr", "-f", "null", "-",
        ]
        result = subprocess.run(scene_cmd, capture_output=True, text=True, timeout=20)
        timestamps = []
        for line in result.stderr.splitlines():
            m = re.search(r"pts_time:([\d.]+)", line)
            if m: timestamps.append(float(m.group(1)))

        if not timestamps:
            return max(0.0, (video_dur - need_dur) / 2)

        best_start = 0.0; best_count = 0; step = 0.5; t = 0.0
        while t + need_dur <= video_dur:
            count = sum(1 for ts in timestamps if t <= ts <= t + need_dur)
            if count > best_count:
                best_count = count; best_start = t
            t += step
        return best_start
    except Exception as e:
        print(f"[best_segment] fallback to middle: {e}", flush=True)
        return max(0.0, (video_dur - need_dur) / 2)


def _make_black_frame(out_path, w, h):
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=black:size={w}x{h}:rate=1",
        "-frames:v", "1", out_path,
    ], capture_output=True)


# ───────────────────────── 8. Concat with transitions ───────────────────

def _normalize_clip(clip_path, out_path, w, h):
    """
    OpenMontage approach: concat আগে সব clip normalize করো।
    Same resolution + fps + codec → freeze/sync problem নেই।
    """
    cmd = [
        "ffmpeg", "-y", "-i", clip_path,
        "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1",
        "-r", "30",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "128k",
        out_path,
    ]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0


def concat_with_transitions(clip_paths, scene_durations, transition_style, out_path, w=720, h=1280):
    if len(clip_paths) == 1:
        shutil.copy(clip_paths[0], out_path); return out_path

    tmp_dir = out_path + "_norm_tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    # Step 1: সব clip normalize করো — OpenMontage এটাই করে
    norm_clips = []
    for i, cp in enumerate(clip_paths):
        norm_path = os.path.join(tmp_dir, f"norm_{i:03d}.mp4")
        ok = _normalize_clip(cp, norm_path, w, h)
        norm_clips.append(norm_path if ok else cp)

    try:
        if transition_style == "none":
            # Simple concat demuxer — clips already normalized
            list_path = os.path.join(tmp_dir, "concat.txt")
            with open(list_path, "w") as f:
                for p in norm_clips: f.write(f"file '{os.path.abspath(p)}'\n")
            cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                   "-c", "copy", out_path]
            subprocess.run(cmd, capture_output=True, check=True)
            return out_path

        # xfade + acrossfade — OpenMontage _chain_xfade approach
        # offset formula: cumulative_offset = prev_offset + clip_dur - transition_dur
        td = TRANSITION_DUR
        xfade_name = "fade" if transition_style == "fade" else "slideleft"
        n = len(norm_clips)

        inputs = []
        for p in norm_clips: inputs += ["-i", p]

        video_filters = []
        audio_filters = []
        cumulative_offset = 0.0

        for i in range(n - 1):
            clip_dur = _ffprobe_duration(norm_clips[i])
            offset = round(cumulative_offset + clip_dur - td, 3)
            offset = max(0.0, offset)

            v_in1 = "[0:v]" if i == 0 else f"[vfade{i-1}]"
            a_in1 = "[0:a]" if i == 0 else f"[afade{i-1}]"
            v_in2 = f"[{i+1}:v]"
            a_in2 = f"[{i+1}:a]"

            v_out = f"[vfade{i}]" if i < n-2 else "[vout]"
            a_out = f"[afade{i}]" if i < n-2 else "[aout]"

            video_filters.append(
                f"{v_in1}{v_in2}xfade=transition={xfade_name}:duration={td}:offset={offset}{v_out}"
            )
            # OpenMontage: acrossfade audio-র জন্য — xfade নয়
            audio_filters.append(
                f"{a_in1}{a_in2}acrossfade=d={td}{a_out}"
            )
            cumulative_offset = offset

        filter_complex = ";".join(video_filters + audio_filters)
        cmd = (["ffmpeg", "-y"] + inputs +
               ["-filter_complex", filter_complex,
                "-map", "[vout]", "-map", "[aout]",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-pix_fmt", "yuv420p", "-threads", "1",
                "-c:a", "aac", "-b:a", "128k",
                out_path])
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            print(f"[xfade] failed: {r.stderr[-300:]}", flush=True)
            # fallback: none transition
            return concat_with_transitions(clip_paths, scene_durations, "none", out_path, w, h)
        return out_path

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ───────────────────────── 9. Background Music ──────────────────────────

def mix_background_music(video_path, music_name, music_volume, out_path, job_dir):
    music_url = MUSIC_TRACKS.get(music_name)
    if not music_url:
        shutil.copy(video_path, out_path); return out_path

    music_path = os.path.join(job_dir, f"music_{music_name}.mp3")
    try:
        download_media(music_url, music_path)
    except Exception as e:
        print(f"[scenevideo] music download failed: {e}", flush=True)
        shutil.copy(video_path, out_path); return out_path

    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-i", music_path,
        "-filter_complex",
        f"[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=3:weights=1 {music_volume:.2f}[outa]",
        "-map", "0:v", "-map", "[outa]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", out_path,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print(f"[scenevideo] music mix failed: {r.stderr[-200:]}", flush=True)
        shutil.copy(video_path, out_path)
    return out_path


# ───────────────────────── Main job runner ──────────────────────────────

def _run_scenevideo_job(job_id):
    job = load_job(job_id)
    if not job: return

    job_dir = os.path.join(SCENEVIDEO_JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    mode         = job.get("media_mode", "hybrid")
    visual_style = job.get("visual_style", "realistic")
    pexels_key   = job.get("pexels_key", "")
    pixabay_key  = job.get("pixabay_key", "")
    unsplash_key = job.get("unsplash_key", "")
    transition   = job.get("transition_style", "fade")
    music_name   = job.get("background_music", "none")
    music_vol    = float(job.get("music_volume", 0.1))
    w, h         = _res(job)

    try:
        job["status"] = "running"; job["stage"] = "script"
        job_log(job, f"🚀 Job শুরু — mode:{mode}, style:{visual_style}, res:{w}x{h}, transition:{transition}")

        # 1) Script
        job_log(job, "📝 Script ও scene breakdown বানানো হচ্ছে...")
        result = generate_script_and_scenes(job["event_description"], job["text_api_key"], visual_style)
        scenes = result["scenes"]
        job["title"] = result.get("title", "")
        job["scenes"] = scenes
        job["scene_count"] = len(scenes)
        job_log(job, f"✅ {len(scenes)}টা scene: {job['title']}")

        # 2) TTS
        job["stage"] = "tts"
        raw_wav = synthesize_audio(job, scenes, job_dir)

        # 3) Silence-split
        job["stage"] = "split"
        job_log(job, "✂️  Scene-wise audio কাটা হচ্ছে...")
        audio_dir = os.path.join(job_dir, "scene_audio")
        scene_audio = split_audio_by_silence(raw_wav, len(scenes), audio_dir)
        for idx, path, dur in scene_audio:
            scenes[idx]["audio_path"] = path
            scenes[idx]["duration"]   = dur
        job_log(job, "✅ Audio split শেষ")
        save_job(job)

        # 4) Media fetch — প্রতি ৪ সেকেন্ডে নতুন visual slot
        job["stage"] = "images"
        job_log(job, f"🎨 Media fetch — {mode}, style:{visual_style}")
        media_dir = os.path.join(job_dir, "scene_media")
        os.makedirs(media_dir, exist_ok=True)

        for idx, scene in enumerate(scenes):
            if job.get("stop_requested"): raise RuntimeError("ইউজার স্টপ করেছে")
            audio_dur = scene.get("duration", 5.0)
            n_slots   = max(1, int(math.ceil(audio_dur / CLIP_INTERVAL)))
            job_log(job, f"   Scene {idx+1}/{len(scenes)} ({audio_dur:.1f}s) — {n_slots} slot")

            slot_medias = []
            for slot in range(n_slots):
                media = get_scene_media(
                    job, {**scene, "_slot": slot},
                    idx * 100 + slot, media_dir, mode,
                    pexels_key, pixabay_key, unsplash_key, visual_style, w, h
                )
                slot_medias.append(media)
                job_log(job, f"     slot {slot+1}: {media['type']} from {media.get('source','?')}")

            scene["slot_medias"] = slot_medias
            scene["n_slots"]     = n_slots
            job["image_done"]    = idx + 1
            save_job(job)

        # 5) Build clips — per slot
        job["stage"] = "video"
        job_log(job, "🎬 Scene clip বানানো হচ্ছে...")
        clip_dir = os.path.join(job_dir, "scene_clips")
        os.makedirs(clip_dir, exist_ok=True)
        clip_paths = []; clip_durations = []

        for i, scene in enumerate(scenes):
            if job.get("stop_requested"): raise RuntimeError("ইউজার স্টপ করেছে")
            audio_path  = scene["audio_path"]
            audio_dur   = scene["duration"]
            slot_medias = scene.get("slot_medias", [])
            n_slots     = scene.get("n_slots", 1)
            slot_dur    = audio_dur / n_slots

            slot_clips = []
            for s, media in enumerate(slot_medias):
                slot_start    = s * slot_dur
                actual_dur    = min(slot_dur, audio_dur - slot_start)
                slot_clip_path  = os.path.join(clip_dir, f"clip_{i:02d}_slot_{s:02d}.mp4")
                slot_audio_path = os.path.join(clip_dir, f"audio_{i:02d}_slot_{s:02d}.aac")

                subprocess.run([
                    "ffmpeg", "-y", "-i", audio_path,
                    "-ss", str(slot_start), "-t", str(actual_dur),
                    "-c:a", "aac", "-b:a", "96k",
                    "-avoid_negative_ts", "make_zero",
                    slot_audio_path,
                ], capture_output=True)

                media_type = media.get("type", "image")
                media_path = media.get("path")

                try:
                    # MoneyPrinterTurbo approach: video segment আলাদা বানাও (no audio)
                    # audio পরে concat-এর পরে overlay হবে
                    vid_only_path = slot_clip_path + "_v.mp4"
                    if media_type == "video" and media_path and os.path.exists(media_path):
                        job_log(job, f"   🎬 Clip {i+1} slot {s+1}: video ({actual_dur:.1f}s)")
                        _prepare_video_segment(media_path, actual_dur, vid_only_path, w, h)
                    else:
                        if not media_path or not os.path.exists(media_path or ""):
                            job_log(job, f"   ⚠️  Clip {i+1} slot {s+1}: media নেই, black frame", "warn")
                            media_path = os.path.join(media_dir, f"black_{i}_{s}.png")
                            _make_black_frame(media_path, w, h)
                        job_log(job, f"   🖼️  Clip {i+1} slot {s+1}: image ({actual_dur:.1f}s)")
                        _prepare_image_segment(media_path, actual_dur, vid_only_path, s % 2 == 0, w, h)
                    slot_clips.append((vid_only_path, actual_dur))
                except Exception as e:
                    job_log(job, f"   ❌ Clip {i+1} slot {s+1} ব্যর্থ: {e}", "error")

            if not slot_clips: continue

            scene_clip_path = os.path.join(clip_dir, f"clip_{i:02d}.mp4")
            # Step 1: video-only segments concat করো
            vid_only_scene = scene_clip_path + "_vonly.mp4"
            if len(slot_clips) == 1:
                shutil.copy(slot_clips[0][0], vid_only_scene)
            else:
                slot_list = scene_clip_path + "_slots.txt"
                with open(slot_list, "w") as f:
                    for sp, _ in slot_clips: f.write(f"file '{sp}'\n")
                r = subprocess.run([
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", slot_list,
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                    "-pix_fmt", "yuv420p", "-r", "30", "-threads", "1",
                    "-an", vid_only_scene,
                ], capture_output=True)
                if os.path.exists(slot_list): os.remove(slot_list)
                if r.returncode != 0:
                    raise RuntimeError(f"slot concat failed: {r.stderr[-200:]}")

            # Step 2: audio overlay — video duration = audio duration হওয়া guaranteed
            # কারণ প্রতিটা segment exact duration দিয়ে বানানো হয়েছে
            r2 = subprocess.run([
                "ffmpeg", "-y",
                "-i", vid_only_scene,
                "-i", audio_path,
                "-t", str(audio_dur),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy",  # video re-encode করা লাগবে না
                "-c:a", "aac", "-b:a", "96k",
                scene_clip_path,
            ], capture_output=True)
            if os.path.exists(vid_only_scene): os.remove(vid_only_scene)
            if r2.returncode != 0:
                raise RuntimeError(f"audio overlay failed: {r2.stderr[-200:]}")

            # actual duration measure করো (estimated audio_dur নয়) — xfade offset accurate হবে
            actual_clip_dur = _ffprobe_duration(scene_clip_path)
            if actual_clip_dur <= 0:
                actual_clip_dur = audio_dur  # fallback
            clip_paths.append(scene_clip_path)
            clip_durations.append(actual_clip_dur)
            job_log(job, f"   ✅ Scene {i+1}/{len(scenes)} তৈরি ({n_slots} slot, {actual_clip_dur:.2f}s)")

        if not clip_paths:
            raise RuntimeError("কোনো clip তৈরি হয়নি")

        # 6) Concat with transitions
        job["stage"] = "compose"
        job_log(job, f"🔗 Concat — transition: {transition}")
        composed_path = os.path.join(job_dir, "composed_video.mp4")
        concat_with_transitions(clip_paths, clip_durations, transition, composed_path, w, h)

        # 7) Background music
        final_path = os.path.join(job_dir, "final_video.mp4")
        if music_name != "none":
            job_log(job, f"🎵 Background music — {music_name} (vol:{music_vol})")
            mix_background_music(composed_path, music_name, music_vol, final_path, job_dir)
        else:
            shutil.copy(composed_path, final_path)

        job["status"] = "complete"; job["stage"] = "done"
        job["final_video"] = final_path
        job_log(job, "🎉 সম্পন্ন!")

    except Exception as e:
        job["status"] = "error"
        job_log(job, f"❌ ব্যর্থ: {e}", "error")

    save_job(job)


# ───────────────────────── Flask routes ─────────────────────────────────

def register_scenevideo_routes(app):

    @app.route("/scenevideo/start", methods=["POST"])
    def scenevideo_start():
        data = request.get_json(force=True)
        event_description = (data.get("event_description") or "").strip()
        text_api_key      = (data.get("text_api_key") or "").strip()
        tts_api_key       = (data.get("tts_api_key") or "").strip()
        voice             = data.get("voice", "Charon")
        se_voice          = data.get("se_voice", STREAMELEMENTS_VOICE)
        media_mode        = data.get("media_mode", "hybrid")
        visual_style      = data.get("visual_style", "realistic")
        resolution        = data.get("resolution", "720x1280")
        transition_style  = data.get("transition_style", "fade")
        background_music  = data.get("background_music", "none")
        music_volume      = float(data.get("music_volume", 0.1))
        pexels_key        = (data.get("pexels_key") or "").strip()
        pixabay_key       = (data.get("pixabay_key") or "").strip()
        unsplash_key      = (data.get("unsplash_key") or "").strip()

        if not event_description:
            return jsonify({"error": "event_description required"}), 400
        if not text_api_key:
            return jsonify({"error": "text_api_key (Gemini) দরকার"}), 400
        if media_mode in ("stock", "hybrid") and not any([pexels_key, pixabay_key, unsplash_key]):
            return jsonify({"error": "Stock mode-এ কমপক্ষে একটা stock API key দাও"}), 400
        if resolution not in RESOLUTIONS:
            resolution = "720x1280"

        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id, "status": "pending", "stage": "queued",
            "created_at": time.time(),
            "event_description": event_description,
            "text_api_key": text_api_key, "tts_api_key": tts_api_key,
            "voice": voice, "se_voice": se_voice,
            "media_mode": media_mode, "visual_style": visual_style,
            "resolution": resolution, "transition_style": transition_style,
            "background_music": background_music, "music_volume": music_volume,
            "pexels_key": pexels_key, "pixabay_key": pixabay_key, "unsplash_key": unsplash_key,
            "stop_requested": False, "logs": [], "scenes": [], "image_done": 0,
        }
        save_job(job)
        threading.Thread(target=_run_scenevideo_job, args=(job_id,), daemon=True).start()
        return jsonify({"job_id": job_id})

    @app.route("/scenevideo/status/<job_id>")
    def scenevideo_status(job_id):
        def generate():
            last_log_len = 0; last_stage = ""
            while True:
                job = load_job(job_id)
                if not job:
                    yield f"event: error\ndata: {json.dumps({'msg':'job not found'})}\n\n"; return
                status = job.get("status",""); stage = job.get("stage","")
                logs = job.get("logs", [])
                if len(logs) > last_log_len:
                    yield f"event: log\ndata: {json.dumps({'entries': logs[last_log_len:]}, ensure_ascii=False)}\n\n"
                    last_log_len = len(logs)
                if stage != last_stage or status in ("complete","error","stopped"):
                    payload = {
                        "stage": stage, "status": status,
                        "scene_count": job.get("scene_count", 0),
                        "image_done": job.get("image_done", 0),
                        "title": job.get("title",""),
                        "media_mode": job.get("media_mode",""),
                        "visual_style": job.get("visual_style",""),
                    }
                    yield f"event: progress\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    last_stage = stage
                if status in ("complete","error","stopped"):
                    yield f"event: done\ndata: {json.dumps({'status':status}, ensure_ascii=False)}\n\n"; return
                time.sleep(1.5)
        return Response(stream_with_context(generate()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.route("/scenevideo/job/<job_id>")
    def scenevideo_job_get(job_id):
        job = load_job(job_id)
        if not job: return jsonify({"error": "job not found"}), 404
        return jsonify({
            "id": job["id"], "status": job.get("status",""), "stage": job.get("stage",""),
            "title": job.get("title",""), "scene_count": job.get("scene_count",0),
            "image_done": job.get("image_done",0), "final_video": job.get("final_video",""),
            "media_mode": job.get("media_mode",""), "visual_style": job.get("visual_style",""),
        })

    @app.route("/scenevideo/stop/<job_id>", methods=["POST"])
    def scenevideo_stop(job_id):
        job = load_job(job_id)
        if not job: return jsonify({"error": "job not found"}), 404
        job["stop_requested"] = True; job["status"] = "stopped"
        save_job(job); return jsonify({"ok": True})

    @app.route("/scenevideo/video/<job_id>")
    def scenevideo_video(job_id):
        job = load_job(job_id)
        if not job or not job.get("final_video"):
            return jsonify({"error": "video not ready"}), 404
        return send_file(job["final_video"], mimetype="video/mp4")

    return app
