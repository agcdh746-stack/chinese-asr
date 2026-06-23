# -*- coding: utf-8 -*-
"""
scenevideo.py — Hybrid Scene Video (v2)
─────────────────────────────────────────────────────────────────────────
নতুন features:
  - Smart search query generation (visual_style aware — realistic/cartoon)
  - StreamElements TTS fallback (Gemini TTS না থাকলে free fallback)
  - Resolution: 16:9 (1920x1080, 1280x720) / 9:16 (720x1280)
  - Transition: fade (xfade) / slide (xfade) / none (direct concat)
    xfade offset = scene_duration - transition_duration (OpenMontage approach)
  - Background music: epic / upbeat / relaxing / none (free hosted)
  - Music volume control
  - Visual style: realistic / cartoon (stock search query-তে apply হয়)
  - Media mode: ai / stock / hybrid
  - Stock priority: Pexels video → Pixabay video → Pexels image → Unsplash image
─────────────────────────────────────────────────────────────────────────
"""

import os
import re
import json
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

# StreamElements TTS — সম্পূর্ণ ফ্রি, key লাগে না
# voices: Brian, Amy, Emma, Joanna, Justin, Matthew, Ivy, Kendra, Kimberly, Salli, Joey, Russell
STREAMELEMENTS_TTS_URL = "https://api.streamelements.com/kappa/v2/speech"
STREAMELEMENTS_VOICE   = "Brian"

# Pollinations — free AI image
POLLINATIONS_MODEL = "flux"

# Stock API endpoints
PEXELS_VIDEO_URL   = "https://api.pexels.com/videos/search"
PEXELS_IMAGE_URL   = "https://api.pexels.com/v1/search"
PIXABAY_VIDEO_URL  = "https://pixabay.com/api/videos/"
PIXABAY_IMAGE_URL  = "https://pixabay.com/api/"
UNSPLASH_IMAGE_URL = "https://api.unsplash.com/search/photos"

# Background music — free hosted tracks (same source as reference project)
MUSIC_TRACKS = {
    "epic":     "https://storage.googleapis.com/aistudio-hosting/music/cinematic-documentary.mp3",
    "upbeat":   "https://storage.googleapis.com/aistudio-hosting/music/corporate-upbeat.mp3",
    "relaxing": "https://storage.googleapis.com/aistudio-hosting/music/relaxing-ambient.mp3",
}

MIN_CLIP_DURATION  = 2.0
TRANSITION_DUR     = 0.5   # xfade duration (seconds)

# ───────────────────────── resolution helper ────────────────────────────

RESOLUTIONS = {
    "1920x1080": (1920, 1080),
    "1280x720":  (1280, 720),
    "720x1280":  (720, 1280),   # 9:16 vertical (Shorts)
    "1080x1920": (1080, 1920),  # 9:16 vertical (Reels)
}

def _res(job):
    key = job.get("resolution", "720x1280")
    return RESOLUTIONS.get(key, (720, 1280))

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
# KEY INSIGHT (reference project থেকে):
# Gemini-কে দিয়ে scene description → stock search query বানানো হয়,
# visual_style অনুযায়ী আলাদা prompt দেওয়া হয়।
# realistic: literal 2-5 word descriptive keyword
# cartoon:   keyword + "illustration"/"animation"/"cartoon" suffix
# আমরা আরো উন্নত করেছি: narration text + scene description দুটোই context হিসেবে দিচ্ছি
# এবং আলাদা করে stock_query, ai_prompt, pixabay_video_type return করছি।

SCENE_SYSTEM_PROMPT = """তুমি একজন বাংলা নিউজ ভিডিও স্ক্রিপ্ট রাইটার ও visual director।
ইউজার একটা বাস্তব ঘটনার বিবরণ দেবে। তোমার কাজ:

1. ঘটনাটাকে ৪-৮টা scene-এ ভাগ করো (কালানুক্রমিকভাবে)।
2. প্রতিটা scene-এ:
   - narration: বাংলা ১-২ বাক্য (সাংবাদিকতার ভাষায়, নিরপেক্ষ)
   - stock_query: ইংরেজি ২-৪ word, Pexels/Pixabay-তে search করার জন্য।
     VISUAL_STYLE placeholder অনুযায়ী:
       realistic → literal descriptive keyword (e.g. "ambulance highway accident")
       cartoon   → keyword + "illustration" বা "animation" (e.g. "ambulance illustration")
     নিয়ম: simple noun+adjective, কোনো বাংলা শব্দ না, কোনো proper noun না
   - ai_prompt: ইংরেজিতে Pollinations-এর জন্য detailed visual description।
     সবসময় "illustrated editorial news-style digital painting" style।
     graphic violence/gore/blood/realistic face দেওয়া যাবে না।

JSON ফরম্যাট (শুধু JSON, অন্য কিছু না):
{
  "title": "ভিডিওর শিরোনাম",
  "scenes": [
    {
      "narration": "বাংলা বাক্য",
      "stock_query": "english keywords",
      "ai_prompt": "English AI image description"
    }
  ]
}
"""

def generate_script_and_scenes(event_description, api_key, visual_style="realistic"):
    # visual_style অনুযায়ী prompt customize
    style_note = (
        "realistic → stock_query must be literal descriptive (e.g. 'fire truck emergency')"
        if visual_style == "realistic"
        else "cartoon → stock_query must include 'illustration' or 'animation' (e.g. 'fire truck illustration')"
    )
    prompt = (
        SCENE_SYSTEM_PROMPT.replace("VISUAL_STYLE placeholder অনুযায়ী:", f"visual style: {style_note}\n     ")
        + f"\n\nঘটনার বিবরণ:\n{event_description}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.7},
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    with httpx.Client(timeout=60) as client:
        resp = client.post(TEXT_ENDPOINT, headers=headers, json=payload)
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
# PRIMARY: Gemini TTS (key দিলে)
# FALLBACK: StreamElements Brian (free, no key) — reference project approach

BREAK_TAG = '<break time="700ms"/>'

def synthesize_full_audio_gemini(scenes, api_key, voice, out_wav_path):
    """Gemini TTS — একটামাত্র call, সব scene একসাথে"""
    full_text = f" {BREAK_TAG} ".join(s["narration"].strip() for s in scenes)
    payload = {
        "contents": [{"parts": [{"text": full_text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    with httpx.Client(timeout=120) as client:
        resp = client.post(TTS_ENDPOINT, headers=headers, json=payload)
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
    """StreamElements TTS — free, no key needed"""
    url = f"{STREAMELEMENTS_TTS_URL}?voice={voice}&text={urllib.parse.quote(text)}"
    with httpx.Client(timeout=30) as client:
        resp = client.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"StreamElements TTS failed ({resp.status_code})")
    return resp.content  # mp3 bytes


def synthesize_full_audio_streamelements(scenes, out_mp3_path, voice=STREAMELEMENTS_VOICE):
    """
    StreamElements TTS fallback — প্রতিটা scene আলাদা MP3 বানিয়ে concat করে।
    Reference project: 200 char chunks, retry 3x
    """
    tmp_dir = out_mp3_path + "_parts"
    os.makedirs(tmp_dir, exist_ok=True)
    part_files = []

    for idx, scene in enumerate(scenes):
        text = scene["narration"].strip()
        # 200 char chunks (reference project approach)
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
        # chunk গুলো concat করে scene MP3
        scene_mp3 = os.path.join(tmp_dir, f"scene_{idx:02d}.mp3")
        if len(chunk_files) == 1:
            shutil.copy(chunk_files[0], scene_mp3)
        else:
            lst = scene_mp3 + ".txt"
            with open(lst, "w") as f:
                for p in chunk_files:
                    f.write(f"file '{p}'\n")
            subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                            "-i", lst, "-c", "copy", scene_mp3], capture_output=True)
        part_files.append(scene_mp3)
        # scene-এর মাঝে 700ms silence যোগ করো (BREAK_TAG এর equivalent)
        if idx < len(scenes) - 1:
            silence_path = os.path.join(tmp_dir, f"silence_{idx:02d}.mp3")
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                "-t", "0.7", "-q:a", "9", "-acodec", "libmp3lame", silence_path
            ], capture_output=True)
            part_files.append(silence_path)

    # সব parts concat → final MP3
    lst_path = out_mp3_path + "_list.txt"
    with open(lst_path, "w") as f:
        for p in part_files:
            f.write(f"file '{p}'\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", lst_path, "-c", "copy", out_mp3_path], capture_output=True, check=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_mp3_path


def synthesize_audio(job, scenes, job_dir):
    """
    TTS entry point:
    - Gemini key আছে → Gemini TTS → WAV
    - নেই / fail → StreamElements → MP3 → WAV convert
    Returns: path to final WAV file
    """
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

    # StreamElements fallback
    job_log(job, "🎙️  StreamElements TTS (free fallback) দিয়ে audio বানানো হচ্ছে...")
    raw_mp3 = os.path.join(job_dir, "full_audio.mp3")
    se_voice = job.get("se_voice", STREAMELEMENTS_VOICE)
    synthesize_full_audio_streamelements(scenes, raw_mp3, se_voice)
    # MP3 → WAV convert
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
# KEY INSIGHT (reference project):
# visual_style="cartoon" → Pixabay-তে video_type=animation, image_type=illustration
# visual_style="realistic" → কোনো filter নেই, normal search
# stock_query-তে already cartoon keyword থাকে (Gemini-ই দিয়েছে)

def _pixabay_video_type(visual_style):
    return "animation" if visual_style == "cartoon" else "film"

def _pixabay_image_type(visual_style):
    return "illustration" if visual_style == "cartoon" else "photo"


def _search_pexels_video(query, api_key, target_dur, visual_style):
    try:
        params = {"query": query, "per_page": 10, "size": "medium"}
        # orientation: portrait for vertical video, landscape for horizontal
        params["orientation"] = "portrait"
        r = httpx.get(PEXELS_VIDEO_URL, headers={"Authorization": api_key},
                      params=params, timeout=15)
        if r.status_code != 200: return None
        videos = r.json().get("videos", [])
        if not videos: return None
        valid = [v for v in videos if v.get("duration", 0) >= MIN_CLIP_DURATION]
        if not valid: valid = videos
        gte = [v for v in valid if v.get("duration", 0) >= target_dur]
        chosen = gte[0] if gte else sorted(valid, key=lambda v: v.get("duration",0), reverse=True)[0]
        files = chosen.get("video_files", [])
        portrait = [f for f in files if f.get("width",9999) <= 1080]
        best = sorted(portrait or files, key=lambda f: f.get("width",0), reverse=True)[0] if (portrait or files) else None
        if not best: return None
        return {"url": best["link"], "duration": chosen["duration"], "kind": "video", "source": "pexels"}
    except Exception as e:
        print(f"[stock] Pexels video error: {e}", flush=True); return None


def _search_pixabay_video(query, api_key, target_dur, visual_style):
    try:
        params = {
            "key": api_key, "q": query, "per_page": 10,
            "video_type": _pixabay_video_type(visual_style),
            "safesearch": "true",
        }
        r = httpx.get(PIXABAY_VIDEO_URL, params=params, timeout=15)
        if r.status_code != 200: return None
        hits = r.json().get("hits", [])
        if not hits: return None
        valid = [h for h in hits if h.get("duration", 0) >= MIN_CLIP_DURATION]
        if not valid: valid = hits
        gte = [h for h in valid if h.get("duration", 0) >= target_dur]
        chosen = gte[0] if gte else sorted(valid, key=lambda h: h.get("duration",0), reverse=True)[0]
        vids = chosen.get("videos", {})
        for q in ("medium", "small", "large"):
            v = vids.get(q)
            if v and v.get("url"):
                return {"url": v["url"], "duration": chosen["duration"], "kind": "video", "source": "pixabay"}
        return None
    except Exception as e:
        print(f"[stock] Pixabay video error: {e}", flush=True); return None


def _search_pexels_image(query, api_key, visual_style):
    try:
        params = {"query": query, "per_page": 5, "orientation": "portrait"}
        r = httpx.get(PEXELS_IMAGE_URL, headers={"Authorization": api_key},
                      params=params, timeout=15)
        if r.status_code != 200: return None
        photos = r.json().get("photos", [])
        if not photos: return None
        src = photos[0].get("src", {})
        url = src.get("portrait") or src.get("large") or src.get("original")
        return {"url": url, "kind": "image", "source": "pexels"} if url else None
    except Exception as e:
        print(f"[stock] Pexels image error: {e}", flush=True); return None


def _search_pixabay_image(query, api_key, visual_style):
    try:
        params = {
            "key": api_key, "q": query, "per_page": 5,
            "image_type": _pixabay_image_type(visual_style),
            "orientation": "vertical", "safesearch": "true",
        }
        r = httpx.get(PIXABAY_IMAGE_URL, params=params, timeout=15)
        if r.status_code != 200: return None
        hits = r.json().get("hits", [])
        if not hits: return None
        url = hits[0].get("largeImageURL") or hits[0].get("webformatURL")
        return {"url": url, "kind": "image", "source": "pixabay"} if url else None
    except Exception as e:
        print(f"[stock] Pixabay image error: {e}", flush=True); return None


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
        print(f"[stock] Unsplash image error: {e}", flush=True); return None


def fetch_stock_media(query, target_dur, pexels_key, pixabay_key, unsplash_key, visual_style="realistic"):
    """Priority: Pexels video → Pixabay video → Pexels image → Pixabay image → Unsplash image"""
    if pexels_key:
        r = _search_pexels_video(query, pexels_key, target_dur, visual_style)
        if r: return r
    if pixabay_key:
        r = _search_pixabay_video(query, pixabay_key, target_dur, visual_style)
        if r: return r
    # image fallback
    if pexels_key:
        r = _search_pexels_image(query, pexels_key, visual_style)
        if r: return r
    if pixabay_key:
        r = _search_pixabay_image(query, pixabay_key, visual_style)
        if r: return r
    if unsplash_key:
        r = _search_unsplash_image(query, unsplash_key, visual_style)
        if r: return r
    return None


def download_media(url, out_path):
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        r = client.get(url)
    if r.status_code != 200:
        raise RuntimeError(f"download failed ({r.status_code}): {url}")
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path


# ───────────────────────── 5. AI Image (Pollinations) ───────────────────

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


# ───────────────────────── 6. Per-scene media (mode-aware) ──────────────

def get_scene_media(job, scene, scene_idx, media_dir,
                    mode, pexels_key, pixabay_key, unsplash_key, visual_style, w, h):
    audio_dur = scene.get("duration", 5.0)
    query = scene.get("stock_query", scene.get("ai_prompt", ""))[:80]

    def try_stock():
        result = fetch_stock_media(query, audio_dur, pexels_key, pixabay_key, unsplash_key, visual_style)
        if not result: return None
        ext = ".mp4" if result["kind"] == "video" else ".jpg"
        out_path = os.path.join(media_dir, f"scene_{scene_idx:02d}_stock{ext}")
        try:
            download_media(result["url"], out_path)
            return {"type": result["kind"], "path": out_path, "source": result["source"]}
        except Exception as e:
            job_log(job, f"⚠️  Stock download error scene {scene_idx}: {e}", "warn")
            return None

    def try_ai():
        out_path = os.path.join(media_dir, f"scene_{scene_idx:02d}_ai.png")
        for attempt in range(3):
            try:
                generate_scene_image(scene["ai_prompt"], out_path, w, h)
                return {"type": "image", "path": out_path, "source": "pollinations"}
            except Exception as e:
                time.sleep(3 * (attempt + 1))
        return None

    if mode == "ai":
        return try_ai() or {"type": "image", "path": None, "source": "failed"}

    if mode == "stock":
        r = try_stock()
        if r: return r
        job_log(job, f"⚠️  Stock miss scene {scene_idx}, AI fallback...", "warn")
        return try_ai() or {"type": "image", "path": None, "source": "failed"}

    if mode == "hybrid":
        stock_result = [None]; ai_result = [None]
        def _stock(): stock_result[0] = try_stock()
        def _ai(): ai_result[0] = try_ai()
        t1 = threading.Thread(target=_stock)
        t2 = threading.Thread(target=_ai)
        t1.start(); t2.start()
        t1.join(); t2.join()
        # stock video > stock image > ai image
        if stock_result[0] and stock_result[0]["type"] == "video":
            return stock_result[0]
        if stock_result[0]:
            return stock_result[0]
        if ai_result[0]:
            return ai_result[0]
        return {"type": "image", "path": None, "source": "failed"}

    return {"type": "image", "path": None, "source": "unknown_mode"}


# ───────────────────────── 7. Scene Clip Builder ────────────────────────

def _scale_filter(w, h):
    return (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1")


def _build_image_clip(image_path, audio_path, duration_sec, out_path, zoom_in, w, h):
    """Image → Ken Burns clip, audio duration master"""
    scale_w = int(w * 1.1); scale_h = int(h * 1.1)
    if zoom_in:
        vf = f"scale={scale_w}:{scale_h},crop={w}:{h}:0:0,setsar=1"
    else:
        ox = (scale_w-w)//2; oy = (scale_h-h)//2
        vf = f"scale={scale_w}:{scale_h},crop={w}:{h}:{ox}:{oy},setsar=1"
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", image_path, "-i", audio_path,
        "-vf", vf, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-pix_fmt", "yuv420p", "-threads", "1",
        "-c:a", "aac", "-b:a", "96k", "-t", str(duration_sec), "-shortest", out_path,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"image clip failed: {r.stderr[-300:]}")
    return out_path


def _build_video_clip(video_path, audio_path, audio_dur, out_path, w, h):
    """Stock video → audio duration-এ fit (trim বা tpad freeze)"""
    video_dur = _ffprobe_duration(video_path)
    if video_dur <= 0:
        raise RuntimeError(f"video duration পড়তে পারেনি: {video_path}")
    sf = _scale_filter(w, h)
    if video_dur >= audio_dur:
        vf = sf
        cmd = ["ffmpeg", "-y", "-i", video_path, "-i", audio_path,
               "-t", str(audio_dur), "-vf", vf,
               "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
               "-pix_fmt", "yuv420p", "-threads", "1",
               "-map", "0:v:0", "-map", "1:a:0",
               "-c:a", "aac", "-b:a", "96k", out_path]
    else:
        pad_dur = audio_dur - video_dur
        vf = f"{sf},tpad=stop_mode=clone:stop_duration={pad_dur:.3f}"
        cmd = ["ffmpeg", "-y", "-i", video_path, "-i", audio_path,
               "-t", str(audio_dur), "-vf", vf,
               "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
               "-pix_fmt", "yuv420p", "-threads", "1",
               "-map", "0:v:0", "-map", "1:a:0",
               "-c:a", "aac", "-b:a", "96k", out_path]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"video clip failed: {r.stderr[-300:]}")
    return out_path


def _make_black_frame(out_path, w, h):
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=black:size={w}x{h}:rate=1",
        "-frames:v", "1", out_path,
    ], capture_output=True)


# ───────────────────────── 8. Concat with transitions ───────────────────
# KEY INSIGHT (reference project):
# xfade offset = scene_duration - transition_duration
# transition_dur = 0 হলে direct concat demuxer

def concat_with_transitions(clip_paths, scene_durations, transition_style, out_path):
    """
    transition_style: "fade" | "slide" | "none"
    scene_durations: প্রতিটা clip-এর duration (xfade offset calculate করতে)
    """
    if len(clip_paths) == 1:
        shutil.copy(clip_paths[0], out_path)
        return out_path

    if transition_style == "none":
        # simple concat demuxer
        list_path = out_path + "_list.txt"
        with open(list_path, "w") as f:
            for p in clip_paths: f.write(f"file '{p}'\n")
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
               "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
               "-pix_fmt", "yuv420p", "-threads", "1",
               "-c:a", "aac", "-b:a", "96k", out_path]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
        finally:
            if os.path.exists(list_path): os.remove(list_path)
        return out_path

    # fade / slide → xfade filter
    # reference project approach: offset = cumulative_duration - TRANSITION_DUR * i
    td = TRANSITION_DUR
    xfade_name = "fade" if transition_style == "fade" else "slideleft"

    inputs = []
    for p in clip_paths:
        inputs += ["-i", p]

    # filter_complex build
    # [0:v][1:v]xfade=...:offset=O1[vt1]; [vt1][2:v]xfade=...:offset=O2[vt2]; ...
    # audio: [0:a][1:a][2:a]concat=n=N:v=0:a=1[outa]
    filter_parts = []
    cumulative = 0.0
    last_v = "0:v"
    last_a_inputs = "[0:a]"

    for i in range(1, len(clip_paths)):
        cumulative += scene_durations[i-1]
        offset = max(0.0, cumulative - td * i)
        cur_v  = f"{i}:v"
        out_v  = f"vt{i}"
        filter_parts.append(
            f"[{last_v}][{cur_v}]xfade=transition={xfade_name}:duration={td}:offset={offset:.3f}[{out_v}]"
        )
        last_v = out_v
        last_a_inputs += f"[{i}:a]"

    # audio concat
    n = len(clip_paths)
    filter_parts.append(f"{last_a_inputs}concat=n={n}:v=0:a=1[outa]")

    filter_complex = ";".join(filter_parts)
    cmd = (["ffmpeg", "-y"] + inputs +
           ["-filter_complex", filter_complex,
            "-map", f"[{last_v}]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-pix_fmt", "yuv420p", "-threads", "1",
            "-c:a", "aac", "-b:a", "96k", out_path])
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        # xfade fail হলে none fallback
        print(f"[scenevideo] xfade failed, falling back to concat: {r.stderr[-200:]}", flush=True)
        return concat_with_transitions(clip_paths, scene_durations, "none", out_path)
    return out_path


# ───────────────────────── 9. Background Music mix ──────────────────────

def mix_background_music(video_path, music_name, music_volume, out_path, job_dir):
    """
    Background music track overlay করো।
    music_name: "epic" | "upbeat" | "relaxing"
    music_volume: 0.0 - 1.0
    Reference project: amix=inputs=2:duration=first:dropout_transition=3:weights=1 {volume}
    """
    music_url = MUSIC_TRACKS.get(music_name)
    if not music_url:
        shutil.copy(video_path, out_path)
        return out_path

    music_path = os.path.join(job_dir, f"music_{music_name}.mp3")
    try:
        download_media(music_url, music_path)
    except Exception as e:
        print(f"[scenevideo] music download failed: {e}", flush=True)
        shutil.copy(video_path, out_path)
        return out_path

    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-i", music_path,
        "-filter_complex",
        f"[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=3:weights=1 {music_volume:.2f}[outa]",
        "-map", "0:v", "-map", "[outa]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        out_path,
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

        # 1) Script + scene breakdown (visual_style-aware query generation)
        job_log(job, "📝 Script ও scene breakdown বানানো হচ্ছে...")
        result = generate_script_and_scenes(
            job["event_description"], job["text_api_key"], visual_style
        )
        scenes = result["scenes"]
        job["title"] = result.get("title", "")
        job["scenes"] = scenes
        job["scene_count"] = len(scenes)
        job_log(job, f"✅ {len(scenes)}টা scene: {job['title']}")

        # 2) TTS (Gemini primary, StreamElements fallback)
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

        # 4) Media fetch (mode + visual_style aware)
        job["stage"] = "images"
        mode_label = {"ai": "AI", "stock": "Stock", "hybrid": "Hybrid"}
        job_log(job, f"🎨 Media fetch — {mode_label.get(mode,mode)}, style:{visual_style}")
        media_dir = os.path.join(job_dir, "scene_media")
        os.makedirs(media_dir, exist_ok=True)

        for idx, scene in enumerate(scenes):
            if job.get("stop_requested"): raise RuntimeError("ইউজার স্টপ করেছে")
            job_log(job, f"   Scene {idx+1}/{len(scenes)} — query: \"{scene.get('stock_query','')}\"")
            media = get_scene_media(
                job, scene, idx, media_dir, mode,
                pexels_key, pixabay_key, unsplash_key, visual_style, w, h
            )
            scene["media_type"]   = media["type"]
            scene["media_path"]   = media["path"]
            scene["media_source"] = media["source"]
            job["image_done"] = idx + 1
            job_log(job, f"   ✅ {media['type']} from {media['source']}")
            save_job(job)

        # 5) Build per-scene clips
        job["stage"] = "video"
        job_log(job, "🎬 Scene clip বানানো হচ্ছে...")
        clip_dir = os.path.join(job_dir, "scene_clips")
        os.makedirs(clip_dir, exist_ok=True)
        clip_paths = []
        clip_durations = []

        for i, scene in enumerate(scenes):
            if job.get("stop_requested"): raise RuntimeError("ইউজার স্টপ করেছে")
            clip_path  = os.path.join(clip_dir, f"clip_{i:02d}.mp4")
            audio_path = scene["audio_path"]
            audio_dur  = scene["duration"]
            media_type = scene.get("media_type", "image")
            media_path = scene.get("media_path")

            try:
                if media_type == "video" and media_path and os.path.exists(media_path):
                    job_log(job, f"   🎬 Clip {i+1}: stock video ({audio_dur:.1f}s)")
                    _build_video_clip(media_path, audio_path, audio_dur, clip_path, w, h)
                else:
                    if not media_path or not os.path.exists(media_path or ""):
                        job_log(job, f"   ⚠️  Clip {i+1}: media নেই, black frame", "warn")
                        media_path = os.path.join(media_dir, f"scene_{i:02d}_black.png")
                        _make_black_frame(media_path, w, h)
                    job_log(job, f"   🖼️  Clip {i+1}: image Ken Burns ({audio_dur:.1f}s)")
                    _build_image_clip(media_path, audio_path, audio_dur, clip_path, i%2==0, w, h)

                clip_paths.append(clip_path)
                clip_durations.append(audio_dur)
                job_log(job, f"   ✅ Clip {i+1}/{len(scenes)} তৈরি")
            except Exception as e:
                job_log(job, f"   ❌ Clip {i+1} ব্যর্থ: {e}", "error")

        if not clip_paths:
            raise RuntimeError("কোনো clip তৈরি হয়নি")

        # 6) Concat with transitions (xfade offset = duration - transition_dur)
        job["stage"] = "compose"
        job_log(job, f"🔗 Concat — transition: {transition}")
        composed_path = os.path.join(job_dir, "composed_video.mp4")
        concat_with_transitions(clip_paths, clip_durations, transition, composed_path)

        # 7) Background music mix
        final_path = os.path.join(job_dir, "final_video.mp4")
        if music_name != "none":
            job_log(job, f"🎵 Background music mix — {music_name} (vol:{music_vol})")
            mix_background_music(composed_path, music_name, music_vol, final_path, job_dir)
        else:
            shutil.copy(composed_path, final_path)

        job["status"] = "complete"; job["stage"] = "done"
        job["final_video"] = final_path
        job_log(job, "🎉 সম্পন্ন! Final video তৈরি।")

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
            return jsonify({"error": f"resolution must be one of: {list(RESOLUTIONS.keys())}"}), 400

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
