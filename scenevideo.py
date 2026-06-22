# -*- coding: utf-8 -*-
"""
scenevideo.py
─────────────────────────────────────────────────────────────────────────
নতুন ফিচার: "Scene Video" — একটা ঘটনার বিবরণ থেকে স্বয়ংক্রিয়ভাবে
script + scene breakdown + TTS (single-call, silence-split) + per-scene
image generation + Ken Burns zoom video — সবকিছু auto বানানো।

এই ফাইলটা app.py থেকে import হবে:
    from scenevideo import register_scenevideo_routes
    register_scenevideo_routes(app)

ডিজাইন সিদ্ধান্ত (আগের আলোচনা অনুযায়ী):
  - Script + scene breakdown: Gemini text model নিজেই বানাবে (ইউজার শুধু
    ঘটনার বিবরণ দেবে)।
  - TTS: পুরো script-কে scene-wise <break> ট্যাগ দিয়ে জোড়া লাগিয়ে
    *একটামাত্র* Gemini TTS call করা হয় (voice consistency বজায় রাখতে),
    তারপর ffmpeg silencedetect দিয়ে scene-wise অডিওতে কাটা হয়।
  - Image: প্রতিটা scene-এর জন্য আলাদা Gemini image generation call
    (multi-key pool দিয়ে rate-limit handle করা হয়)।
  - Video: প্রতিটা ছবিকে তার scene-audio duration অনুযায়ী zoompan
    (Ken Burns) দিয়ে animate করে, scene-থেকে-scene ছোট crossfade
    (xfade) দিয়ে জোড়া লাগানো হয়।
  - Architecture: পুরোপুরি standalone — existing /tts বা /am route
    স্পর্শ করা হয়নি। নতুন route: /scenevideo/*
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

import httpx
from flask import request, jsonify, Response, stream_with_context, send_file

# ───────────────────────── কনফিগ / পাথ ─────────────────────────────────

SCENEVIDEO_JOBS_DIR = os.path.join(os.getcwd(), "scenevideo_jobs")
os.makedirs(SCENEVIDEO_JOBS_DIR, exist_ok=True)

TTS_MODEL   = "gemini-2.5-flash-preview-tts"
TEXT_MODEL  = "gemini-2.5-flash"                # script/scene breakdown জেনারেশন

TTS_ENDPOINT  = f"https://generativelanguage.googleapis.com/v1beta/models/{TTS_MODEL}:generateContent"
TEXT_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{TEXT_MODEL}:generateContent"

# Pollinations image — key ছাড়া সম্পূর্ণ ফ্রি
POLLINATIONS_IMAGE_URL = "https://image.pollinations.ai/prompt/{prompt}"
POLLINATIONS_MODEL     = "flux"          # flux | flux-realism | turbo
POLLINATIONS_IMG_W     = 1080
POLLINATIONS_IMG_H     = 1920

VIDEO_W, VIDEO_H = 1080, 1920   # vertical (shorts/reels স্টাইল)
FPS = 30

# ───────────────────────── job ফাইল হেল্পার ────────────────────────────

def _job_path(job_id):
    return os.path.join(SCENEVIDEO_JOBS_DIR, f"{job_id}.json")

def load_job(job_id):
    p = _job_path(job_id)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_job(job):
    with open(_job_path(job["id"]), "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)

def job_log(job, msg, lvl="info"):
    entry = {"t": time.time(), "lvl": lvl, "msg": msg}
    job.setdefault("logs", []).append(entry)
    print(f"[scenevideo:{job['id']}] {msg}", flush=True)
    save_job(job)


# ───────────────────────── Gemini key pool (Chinese-ASR প্রজেক্ট থেকে reuse করা প্যাটার্ন) ─

def _init_key_pool(job, keys):
    job["key_pool"] = [
        {
            "key": k,
            "cooldown_until": 0.0,
            "rpm_count": 0,
            "rpm_window_start": time.time(),
            "last_status": "ready",
            "dead": False,
        }
        for k in keys if k and k.strip()
    ]

def _pick_healthy_key(job, now):
    pool = job.get("key_pool", [])
    candidates = [k for k in pool if not k["dead"] and k["cooldown_until"] <= now]
    if not candidates:
        return None
    candidates.sort(key=lambda k: k["rpm_count"])
    return candidates[0]

def _next_cooldown_eta(job, now):
    pool = [k for k in job.get("key_pool", []) if not k["dead"]]
    if not pool:
        return None
    return min(max(0.0, k["cooldown_until"] - now) for k in pool)

def _mark_key_cooldown(key_entry, seconds):
    key_entry["cooldown_until"] = time.time() + seconds
    key_entry["last_status"] = f"cooldown_{int(seconds)}s"

def _mark_key_dead(key_entry, reason):
    key_entry["dead"] = True
    key_entry["last_status"] = f"dead:{reason}"


# ───────────────────────── ১. Script + Scene Breakdown (Gemini text) ───

SCENE_SYSTEM_PROMPT = """তুমি একজন বাংলা নিউজ ভিডিও স্ক্রিপ্ট রাইটার। ইউজার একটা বাস্তব ঘটনার সংক্ষিপ্ত
বিবরণ দেবে। তোমার কাজ:

1. ঘটনাটাকে ৪ থেকে ৮ টা সংক্ষিপ্ত দৃশ্যে (scene) ভাগ করো, কালানুক্রমিকভাবে।
2. প্রতিটা scene-এর জন্য একটা ছোট বাংলা narration sentence লেখো (১-২ বাক্য,
   নিরপেক্ষ সাংবাদিকতার ভাষায়, অতিরিক্ত নাটকীয়তা ছাড়া)।
3. প্রতিটা scene-এর জন্য একটা ইংরেজি image-generation prompt লেখো যা সেই
   দৃশ্যের ভিজ্যুয়াল বর্ণনা করে — সবসময় "illustrated editorial news-style
   digital painting" বা "symbolic/silhouette style" রাখবে, কখনো realistic
   graphic violence/gore/blood/rক্তপাত বা কোনো বাস্তব ব্যক্তির চেহারা আঁকার
   prompt দেবে না। সহিংসতা থাকলেও সেটা suggestive/silhouette/distant-angle
   ভাবে বর্ণনা করবে, কখনো explicit/graphic না।
4. কোনো ভুক্তভোগী/অভিযুক্তের আসল নাম ব্যবহার করবে না যদি ইউজার নিজে না দেয়;
   সাধারণ বর্ণনা (যেমন "অ্যাম্বুলেন্স চালক", "উদ্ধারকারী দল") ব্যবহার করবে।

JSON আউটপুট ফরম্যাট (অন্য কিছু লিখবে না, শুধু এই JSON):
{
  "title": "ভিডিওর সংক্ষিপ্ত শিরোনাম",
  "scenes": [
    {"narration": "বাংলা বাক্য", "image_prompt": "English visual description"},
    ...
  ]
}
"""

def generate_script_and_scenes(event_description, api_key):
    """Gemini text model দিয়ে script + scene breakdown বানায়। JSON dict রিটার্ন করে।"""
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": SCENE_SYSTEM_PROMPT + "\n\nঘটনার বিবরণ:\n" + event_description}]}
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.7,
        },
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
        raise RuntimeError(f"unexpected script response shape: {data}") from e

    parsed = json.loads(text)
    if "scenes" not in parsed or not parsed["scenes"]:
        raise RuntimeError("Gemini script response-এ কোনো scene পাওয়া যায়নি")
    return parsed


# ───────────────────────── ২. একটামাত্র TTS call + silence-split ───────

BREAK_TAG = '<break time="700ms"/>'

def _build_full_script(scenes):
    """সব scene narration জোড়া দিয়ে <break> ট্যাগ সহ একটা স্ট্রিং বানায়।"""
    parts = [s["narration"].strip() for s in scenes]
    return f" {BREAK_TAG} ".join(parts)


def synthesize_full_audio(scenes, api_key, voice, out_wav_path):
    """পুরো script-কে একটামাত্র TTS call-এ পাঠায়, raw WAV ফাইলে সেভ করে।"""
    full_text = _build_full_script(scenes)

    payload = {
        "contents": [{"parts": [{"text": full_text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
            },
        },
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    with httpx.Client(timeout=120) as client:
        resp = client.post(TTS_ENDPOINT, headers=headers, json=payload)

    if resp.status_code != 200:
        raise RuntimeError(f"TTS call failed ({resp.status_code}): {resp.text[:500]}")

    data = resp.json()
    pcm_b64 = None
    try:
        for part in data["candidates"][0]["content"]["parts"]:
            if "inlineData" in part and part["inlineData"].get("data"):
                pcm_b64 = part["inlineData"]["data"]
                break
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"unexpected TTS response shape: {data}") from e

    if not pcm_b64:
        raise RuntimeError(f"TTS response-এ কোনো audio data পাওয়া যায়নি: {data}")

    pcm_data = base64.b64decode(pcm_b64)
    with wave.open(out_wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm_data)

    return out_wav_path


def _ffprobe_duration(path):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", path]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    return float(out)


def _run_silencedetect(wav_path, noise_db="-30dB", min_dur=0.3):
    cmd = ["ffmpeg", "-i", wav_path, "-af",
           f"silencedetect=noise={noise_db}:d={min_dur}", "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = result.stderr
    starts = [float(m) for m in re.findall(r"silence_start:\s*([\d.]+)", stderr)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*([\d.]+)", stderr)]
    return list(zip(starts, ends))


def split_audio_by_silence(wav_path, expected_scene_count, out_dir):
    """
    TTS audio-কে silence gap অনুযায়ী কেটে scene-wise WAV ফাইল বানায়।
    Returns: list of (scene_idx, wav_path, duration) — expected_scene_count
             না মিললে adaptive থ্রেশহোল্ড দিয়ে retry করে; শেষ পর্যন্ত না মিললে
             RuntimeError (caller-কে fallback/retry লজিক চালাতে হবে)।
    """
    os.makedirs(out_dir, exist_ok=True)
    total_dur = _ffprobe_duration(wav_path)

    attempts = [
        ("-30dB", 0.3), ("-25dB", 0.2), ("-22dB", 0.15), ("-35dB", 0.4),
    ]

    pairs = None
    for noise_db, min_dur in attempts:
        candidate = _run_silencedetect(wav_path, noise_db, min_dur)
        # ঠিক expected_scene_count - 1 টা gap দরকার, N scene-এর জন্য N-1 break
        if len(candidate) == expected_scene_count - 1:
            pairs = candidate
            break
        if pairs is None or abs(len(candidate) - (expected_scene_count - 1)) < abs(len(pairs) - (expected_scene_count - 1)):
            pairs = candidate  # সবচেয়ে কাছের ফলাফল রেখে দেওয়া fallback হিসেবে

    if pairs is None:
        pairs = []

    cut_points = [0.0]
    for s_start, s_end in pairs:
        cut_points.append(s_start)
        cut_points.append(s_end)
    cut_points.append(total_dur)

    scenes_out = []
    for i in range(0, len(cut_points) - 1, 2):
        start, end = cut_points[i], cut_points[i + 1]
        if end - start < 0.05:
            continue
        scenes_out.append((start, end))

    if len(scenes_out) != expected_scene_count:
        print(
            f"[scenevideo] ⚠️ silence-split mismatch: {len(scenes_out)} অংশ পাওয়া গেছে, "
            f"{expected_scene_count} দরকার ছিল। Auto-adjust করা হচ্ছে...", flush=True
        )
        if len(scenes_out) == 0:
            # কোনো silence নেই — পুরো audio সমান ভাগে ভাগ করো
            chunk = total_dur / expected_scene_count
            scenes_out = [(i * chunk, (i + 1) * chunk) for i in range(expected_scene_count)]
        elif len(scenes_out) > expected_scene_count:
            # বেশি অংশ — শেষেরগুলো merge করো যতক্ষণ না মেলে
            while len(scenes_out) > expected_scene_count:
                # সবচেয়ে ছোট দুটো পাশাপাশি segment merge করো
                min_idx = min(range(len(scenes_out) - 1),
                              key=lambda i: (scenes_out[i][1] - scenes_out[i][0]) +
                                            (scenes_out[i+1][1] - scenes_out[i+1][0]))
                merged = (scenes_out[min_idx][0], scenes_out[min_idx + 1][1])
                scenes_out = scenes_out[:min_idx] + [merged] + scenes_out[min_idx + 2:]
        else:
            # কম অংশ — সবচেয়ে বড় segment ভাগ করো যতক্ষণ না মেলে
            while len(scenes_out) < expected_scene_count:
                max_idx = max(range(len(scenes_out)),
                              key=lambda i: scenes_out[i][1] - scenes_out[i][0])
                s, e = scenes_out[max_idx]
                mid = (s + e) / 2
                scenes_out = scenes_out[:max_idx] + [(s, mid), (mid, e)] + scenes_out[max_idx + 1:]

    results = []
    for idx, (start, end) in enumerate(scenes_out):
        out_file = os.path.join(out_dir, f"scene_{idx:02d}.wav")
        cmd = ["ffmpeg", "-y", "-i", wav_path, "-ss", str(start), "-to", str(end),
               "-c", "copy", out_file]
        subprocess.run(cmd, capture_output=True)
        results.append((idx, out_file, end - start))

    return results


# ───────────────────────── ৩. প্রতি-scene image generation (Pollinations) ──

def generate_scene_image(prompt, out_path):
    """
    Pollinations.ai দিয়ে একটা scene-এর ছবি বানায়।
    key লাগে না — সম্পূর্ণ ফ্রি।
    anatomy ঠিক রাখতে tight suffix যোগ করা হয়।
    """
    anatomy_suffix = (
        "correct human anatomy, two arms, two hands, two legs, "
        "no extra limbs, no deformity, photorealistic, high quality"
    )
    full_prompt = f"{prompt.strip()}, {anatomy_suffix}"

    import urllib.parse
    encoded = urllib.parse.quote(full_prompt)
    url = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?model={POLLINATIONS_MODEL}"
        f"&width={POLLINATIONS_IMG_W}"
        f"&height={POLLINATIONS_IMG_H}"
        f"&nologo=true"
        f"&enhance=true"
    )

    with httpx.Client(timeout=120) as client:
        resp = client.get(url)

    if resp.status_code != 200:
        raise RuntimeError(f"Pollinations image failed ({resp.status_code}): {resp.text[:300]}")

    with open(out_path, "wb") as f:
        f.write(resp.content)
    return out_path


def generate_all_scene_images(job, scenes, out_dir):
    """
    সব scene-এর ছবি Pollinations দিয়ে বানায়।
    rate limit নেই — simple sequential loop।
    """
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    for idx, scene in enumerate(scenes):
        if job.get("stop_requested"):
            raise RuntimeError("ইউজার স্টপ করেছে")

        out_path = os.path.join(out_dir, f"scene_{idx:02d}.png")
        job_log(job, f"🖼️  Scene {idx+1}/{len(scenes)} ছবি বানানো হচ্ছে...")

        retry = 0
        while retry < 3:
            try:
                generate_scene_image(scene["image_prompt"], out_path)
                results[idx] = out_path
                job_log(job, f"✅ Scene {idx+1}/{len(scenes)} ছবি তৈরি হয়েছে")
                job["image_done"] = len(results)
                save_job(job)
                break
            except Exception as e:
                retry += 1
                job_log(job, f"⚠️  Scene {idx} retry {retry}/3 — {str(e)[:150]}", "warn")
                time.sleep(3 * retry)
        else:
            job_log(job, f"❌ Scene {idx} ছবি বানাতে ব্যর্থ — skip করা হলো", "error")

    return results


# ───────────────────────── ৪. Ken Burns video + crossfade compose ──────

def _zoompan_filter(duration_sec, zoom_in=True):
    """
    FFmpeg zoompan filter স্ট্রিং বানায় — static image-কে slow zoom/pan
    দিয়ে animate করে (Ken Burns effect)। ২-৫ সেকেন্ড movement গাইডলাইন
    অনুসরণ করে ধীর movement রাখা হয়েছে।
    """
    frames = max(int(duration_sec * FPS), 1)
    if zoom_in:
        zoom_expr = "zoom+0.0008"
        z_start, z_end = 1.0, 1.0 + 0.0008 * frames
    else:
        z_start, z_end = 1.15, 1.15 - 0.0008 * frames
        zoom_expr = "zoom-0.0008"

    return (
        f"scale={int(VIDEO_W*1.2)}:{int(VIDEO_H*1.2)},"
        f"zoompan=z='{zoom_expr}':d={frames}:s={VIDEO_W}x{VIDEO_H}:fps={FPS}"
    )


def build_scene_clip(image_path, audio_path, duration_sec, out_path, zoom_in=True):
    """একটা scene-এর ছবি+অডিও থেকে Ken Burns zoom সহ একটা video clip বানায়।"""
    vf = _zoompan_filter(duration_sec, zoom_in=zoom_in)
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", image_path,
        "-i", audio_path,
        "-vf", vf,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-t", str(duration_sec),
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return out_path


def concat_with_crossfade(clip_paths, out_path, fade_dur=0.4):
    """
    একাধিক scene clip-কে ছোট crossfade (xfade) দিয়ে জোড়া লাগায়।
    ffmpeg-এর xfade filter chain ব্যবহার করা হয়েছে (transition: fade)।
    """
    if len(clip_paths) == 1:
        shutil.copy(clip_paths[0], out_path)
        return out_path

    durations = [_ffprobe_duration(p) for p in clip_paths]

    inputs = []
    for p in clip_paths:
        inputs += ["-i", p]

    filter_parts = []
    audio_parts = []
    running_offset = 0.0
    prev_v = "0:v"
    prev_a = "0:a"

    for i in range(1, len(clip_paths)):
        offset = running_offset + durations[i - 1] - fade_dur
        v_label = f"v{i}"
        a_label = f"a{i}"
        filter_parts.append(
            f"[{prev_v}][{i}:v]xfade=transition=fade:duration={fade_dur}:offset={offset:.3f}[{v_label}]"
        )
        audio_parts.append(
            f"[{prev_a}][{i}:a]acrossfade=d={fade_dur}[{a_label}]"
        )
        prev_v = v_label
        prev_a = a_label
        running_offset = offset + fade_dur

    filter_complex = ";".join(filter_parts + audio_parts)

    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", filter_complex,
        "-map", f"[{prev_v}]", "-map", f"[{prev_a}]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return out_path


# ───────────────────────── মূল job runner (background thread) ──────────

def _run_scenevideo_job(job_id):
    job = load_job(job_id)
    if not job:
        return

    job_dir = os.path.join(SCENEVIDEO_JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    try:
        job["status"] = "running"
        job["stage"] = "script"
        job_log(job, "🚀 Job শুরু হলো")

        # ১) Script + scene breakdown
        job_log(job, "📝 Script ও scene breakdown বানানো হচ্ছে...")
        result = generate_script_and_scenes(job["event_description"], job["text_api_key"])
        scenes = result["scenes"]
        job["title"] = result.get("title", "")
        job["scenes"] = scenes
        job["scene_count"] = len(scenes)
        job_log(job, f"✅ {len(scenes)}টা scene পাওয়া গেছে: {job['title']}")

        # ২) একটামাত্র TTS call
        job["stage"] = "tts"
        job_log(job, "🎙️  TTS audio বানানো হচ্ছে (একটামাত্র call, সব scene একসাথে)...")
        raw_wav = os.path.join(job_dir, "full_audio.wav")
        synthesize_full_audio(scenes, job["tts_api_key"], job.get("voice", "Charon"), raw_wav)

        # ৩) Silence-split
        job["stage"] = "split"
        job_log(job, "✂️  Silence দিয়ে scene-wise audio কাটা হচ্ছে...")
        audio_out_dir = os.path.join(job_dir, "scene_audio")
        scene_audio = split_audio_by_silence(raw_wav, len(scenes), audio_out_dir)
        for idx, path, dur in scene_audio:
            scenes[idx]["audio_path"] = path
            scenes[idx]["duration"] = dur
        job_log(job, "✅ Audio scene-wise কাটা শেষ")
        save_job(job)

        # ৪) প্রতি-scene ছবি (Pollinations — key লাগে না)
        job["stage"] = "images"
        job_log(job, "🖼️  Scene ছবি বানানো শুরু হচ্ছে (Pollinations)...")
        image_out_dir = os.path.join(job_dir, "scene_images")
        images = generate_all_scene_images(job, scenes, image_out_dir)
        for idx, path in images.items():
            scenes[idx]["image_path"] = path
        save_job(job)

        # ৫) Ken Burns clip + crossfade compose
        job["stage"] = "video"
        job_log(job, "🎬 Ken Burns video clip বানানো হচ্ছে...")
        clip_dir = os.path.join(job_dir, "scene_clips")
        os.makedirs(clip_dir, exist_ok=True)
        clip_paths = []
        for i, scene in enumerate(scenes):
            clip_path = os.path.join(clip_dir, f"clip_{i:02d}.mp4")
            build_scene_clip(
                scene["image_path"], scene["audio_path"], scene["duration"],
                clip_path, zoom_in=(i % 2 == 0),
            )
            clip_paths.append(clip_path)
            job_log(job, f"   clip {i+1}/{len(scenes)} তৈরি হলো")

        job["stage"] = "compose"
        job_log(job, "🔗 সব clip জোড়া লাগানো হচ্ছে (crossfade)...")
        final_path = os.path.join(job_dir, "final_video.mp4")
        concat_with_crossfade(clip_paths, final_path)

        job["status"] = "complete"
        job["stage"] = "done"
        job["final_video"] = final_path
        job_log(job, "🎉 সম্পন্ন! ফাইনাল ভিডিও তৈরি হয়েছে।")

    except Exception as e:
        job["status"] = "error"
        job_log(job, f"❌ ব্যর্থ হলো: {e}", "error")

    save_job(job)


# ───────────────────────── Flask routes ─────────────────────────────────

def register_scenevideo_routes(app):

    @app.route("/scenevideo/start", methods=["POST"])
    def scenevideo_start():
        data = request.get_json(force=True)

        event_description = (data.get("event_description") or "").strip()
        text_api_key       = (data.get("text_api_key") or "").strip()
        tts_api_key        = (data.get("tts_api_key") or "").strip()
        voice              = data.get("voice", "Charon")

        if not event_description:
            return jsonify({"error": "event_description required"}), 400
        if not text_api_key or not tts_api_key:
            return jsonify({"error": "text_api_key এবং tts_api_key দুটোই দরকার"}), 400

        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "status": "pending",
            "stage": "queued",
            "created_at": time.time(),
            "event_description": event_description,
            "text_api_key": text_api_key,
            "tts_api_key": tts_api_key,
            "voice": voice,
            "stop_requested": False,
            "logs": [],
            "scenes": [],
            "image_done": 0,
        }
        save_job(job)
        threading.Thread(target=_run_scenevideo_job, args=(job_id,), daemon=True).start()
        return jsonify({"job_id": job_id})

    @app.route("/scenevideo/status/<job_id>")
    def scenevideo_status(job_id):
        def generate():
            last_log_len = 0
            last_stage = ""
            while True:
                job = load_job(job_id)
                if not job:
                    yield f"event: error\ndata: {json.dumps({'msg':'job not found'})}\n\n"
                    return

                status = job.get("status", "")
                stage  = job.get("stage", "")
                logs   = job.get("logs", [])

                if len(logs) > last_log_len:
                    new_logs = logs[last_log_len:]
                    last_log_len = len(logs)
                    yield f"event: log\ndata: {json.dumps({'entries': new_logs}, ensure_ascii=False)}\n\n"

                if stage != last_stage or status in ("complete", "error", "stopped"):
                    progress_payload = {
                        "stage": stage,
                        "status": status,
                        "scene_count": job.get("scene_count", 0),
                        "image_done": job.get("image_done", 0),
                        "title": job.get("title", ""),
                    }
                    yield f"event: progress\ndata: {json.dumps(progress_payload, ensure_ascii=False)}\n\n"
                    last_stage = stage

                if status in ("complete", "error", "stopped"):
                    yield (
                        f"event: done\n"
                        f"data: {json.dumps({'status': status}, ensure_ascii=False)}\n\n"
                    )
                    return

                time.sleep(1.5)

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/scenevideo/job/<job_id>")
    def scenevideo_job_get(job_id):
        job = load_job(job_id)
        if not job:
            return jsonify({"error": "job not found"}), 404
        return jsonify({
            "id": job["id"],
            "status": job.get("status", ""),
            "stage": job.get("stage", ""),
            "title": job.get("title", ""),
            "scene_count": job.get("scene_count", 0),
            "image_done": job.get("image_done", 0),
            "final_video": job.get("final_video", ""),
        })

    @app.route("/scenevideo/stop/<job_id>", methods=["POST"])
    def scenevideo_stop(job_id):
        job = load_job(job_id)
        if not job:
            return jsonify({"error": "job not found"}), 404
        job["stop_requested"] = True
        job["status"] = "stopped"
        save_job(job)
        return jsonify({"ok": True})

    @app.route("/scenevideo/video/<job_id>")
    def scenevideo_video(job_id):
        job = load_job(job_id)
        if not job or not job.get("final_video"):
            return jsonify({"error": "video not ready"}), 404
        return send_file(job["final_video"], mimetype="video/mp4")

    return app
