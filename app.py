import os, re, json, time, uuid, struct, asyncio, subprocess, tempfile, base64
import threading, hashlib, shutil
from pathlib import Path

import httpx
import edge_tts
from flask import Flask, request, jsonify, render_template, Response, stream_with_context, send_from_directory

app = Flask(__name__)

# â”€â”€â”€ Global JSON error handler (V6) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Frontend "Unexpected token '<'" error à¦¹à¦¯à¦¼ à¦¯à¦–à¦¨ route exception throw à¦•à¦°à§‡
# à¦à¦¬à¦‚ Flask default HTML error page à¦«à§‡à¦°à¦¤ à¦ªà¦¾à¦ à¦¾à¦¯à¦¼à¥¤ à¦à¦–à¦¨ à¦¸à¦¬ error JSON-à¦ à¦¯à¦¾à¦¬à§‡à¥¤
import traceback as _traceback
from werkzeug.exceptions import HTTPException as _HTTPException

@app.errorhandler(_HTTPException)
def _json_http_err(e):
    return jsonify({
        "error":  e.name,
        "code":   e.code,
        "detail": e.description,
    }), e.code

@app.errorhandler(Exception)
def _json_any_err(e):
    tb = _traceback.format_exc()
    print(f"[ERR] Unhandled exception: {e}\n{tb}", flush=True)
    return jsonify({
        "error":  "internal_error",
        "type":   type(e).__name__,
        "detail": str(e)[:500],
    }), 500
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


TEMP_DIR     = os.environ.get("TEMP_DIR", "/tmp/asr")
KS_API       = os.environ.get("KS_API", "http://localhost:5557")
KS_PROXY     = os.environ.get("KS_PROXY", "").strip()
KS_COOKIE    = os.environ.get("KS_COOKIE", "").strip()
COOKIES_FILE = os.environ.get("COOKIES_FILE", "/app/data/cookies/cookies.txt")
YTDLP_PROXY  = os.environ.get("YTDLP_PROXY", "").strip()

Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TTS JOB SYSTEM â€” zip à¦à¦° à¦¸à¦¬ features
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

JOBS_DIR  = os.path.join(TEMP_DIR, "tts_jobs")
CACHE_DIR = os.path.join(TEMP_DIR, "tts_cache")   # SHA256 cache
for _d in [JOBS_DIR, CACHE_DIR]:
    Path(_d).mkdir(parents=True, exist_ok=True)

GEMINI_TTS_MODEL        = "gemini-2.5-flash-preview-tts"
GEMINI_TTS_BACKUP_MODEL = "gemini-2.5-pro-preview-tts"   # zip reference fallback
GEMINI_TTS_URL_TMPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)

# â”€â”€ Job file helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _job_path(job_id):    return os.path.join(JOBS_DIR, f"{job_id}.json")
def _seg_wav_path(job_id, idx): return os.path.join(JOBS_DIR, f"{job_id}_seg{idx}.wav")

def load_job(job_id):
    p = _job_path(job_id)
    if not os.path.exists(p): return None
    try: return json.loads(open(p).read())
    except: return None

def save_job(job):
    open(_job_path(job["id"]), "w").write(json.dumps(job, ensure_ascii=False, indent=2))

def job_log(job, msg, lvl="info"):
    if "logs" not in job: job["logs"] = []
    job["logs"].append({"t": round(time.time(), 2), "msg": msg, "lvl": lvl})
    if len(job["logs"]) > 200: job["logs"] = job["logs"][-200:]

# â”€â”€ PCM â†’ WAV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def pcm_to_wav(pcm, sample_rate=24000):
    nc=1; bps=16; data_size=len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36+data_size, b"WAVE", b"fmt ", 16, 1,
        nc, sample_rate, sample_rate*nc*(bps//8), nc*(bps//8), bps,
        b"data", data_size,
    )
    return header + pcm

# â”€â”€ SHA256 Cache â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _cache_key(text, voice, language, model=None):
    data = json.dumps({"text":text,"voice":voice,"language":language,
                       "model": model or GEMINI_TTS_MODEL}, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()

def _cache_get(key):
    p = os.path.join(CACHE_DIR, f"{key}.wav")
    return p if os.path.exists(p) and os.path.getsize(p) > 100 else None

def _cache_set(key, wav_path):
    dst = os.path.join(CACHE_DIR, f"{key}.wav")
    try: shutil.copy2(wav_path, dst)
    except: pass

# â”€â”€ Silence removal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def remove_silence(wav_path):
    fd, tmp = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    try:
        shutil.copy2(wav_path, tmp)
        subprocess.run([
            "ffmpeg","-hide_banner","-y","-i",tmp,
            "-af","silenceremove=stop_periods=-1:stop_duration=0.1:stop_threshold=-50dB",
            wav_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except: pass
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

# â”€â”€ concat_close_transcripts (zip à¦à¦° à¦®à¦¤à§‹) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def concat_close_transcripts(segments, threshold=2.0, max_chunk_sec=15):
    """
    BUG FIX: à¦†à¦—à§‡ MAX_DUR=120s + factor=2 â†’ à¦¸à¦¬ segments 1-à¦ merge à¦¹à¦¯à¦¼à§‡ "1/1 done"à¥¤
    à¦à¦–à¦¨ max 15s, threshold doubling à¦¨à¦¾à¦‡à¥¤ Default-à¦ OFF (worker check à¦•à¦°à§‡)à¥¤

    Enhancement: TTS-à¦•à§‡ natural pause à¦¨à¦¿à¦¤à§‡ à¦¬à¦²à¦¾à¦° à¦œà¦¨à§à¦¯ merge-à¦
    "(pause for N seconds)" text inject à¦•à¦°à§‡ (zip reference à¦¥à§‡à¦•à§‡)à¥¤
    """
    if not segments: return []
    result = [dict(segments[0])]
    for curr in segments[1:]:
        prev = result[-1]
        diff = curr["start"] - prev["end"]
        merged_dur = curr["end"] - prev["start"]
        if diff <= threshold and merged_dur <= max_chunk_sec:
            pause_sec = max(0, int(round(diff)))
            prev["end"]  = curr["end"]
            # TTS-à¦•à§‡ pause à¦¨à¦¿à¦¤à§‡ à¦¬à¦²à§‹; à¦ªà¦¾à¦¶à¦¾à¦ªà¦¾à¦¶à¦¿ text-à¦“ à¦¯à§à¦•à§à¦¤ à¦•à¦°à§‹
            if pause_sec >= 1:
                prev["text"] += f"\n(pause for {pause_sec} seconds)\n{curr['text']}"
            else:
                prev["text"] += f" {curr['text']}"
        else:
            result.append(dict(curr))
    return result

# â”€â”€ Gemini TTS (style prompt à¦¸à¦¹) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
async def _gemini_tts_call(text, voice, api_key, language="Bengali", model=None):
    model = model or GEMINI_TTS_MODEL
    styled = (
        f"<style-instruction>\n"
        f"The following is a dubbed segment in {language} language.\n"
        f"Take pauses and intonate accordingly.\n"
        f"Read aloud in a calm, soothing, enthusiastic tone like David Attenborough:\n"
        f"</style-instruction>\n{text}"
    )
    payload = {
        "contents": [{"parts": [{"text": styled}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }
    url = GEMINI_TTS_URL_TMPL.format(model=model, key=api_key)
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(url, json=payload)
        if resp.status_code == 429:
            raise httpx.HTTPStatusError("429", request=resp.request, response=resp)
        resp.raise_for_status()
        b64 = (
            resp.json().get("candidates",[{}])[0]
            .get("content",{}).get("parts",[{}])[0]
            .get("inlineData",{}).get("data")
        )
        if not b64: raise ValueError("Gemini returned no audio data")
        return base64.b64decode(b64)

# â”€â”€ Edge-TTS fallback â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
async def _edge_tts_call(text, voice="bn-IN-TanishaaNeural", pitch="-5Hz", rate="+12%"):
    fd, mp3 = tempfile.mkstemp(suffix=".mp3"); os.close(fd)
    wav = None
    try:
        comm = edge_tts.Communicate(text, voice, pitch=pitch, rate=rate)
        await comm.save(mp3)
        fd2, wav = tempfile.mkstemp(suffix=".wav"); os.close(fd2)
        subprocess.run(
            ["ffmpeg","-y","-i",mp3,"-ar","24000","-ac","1","-sample_fmt","s16",wav],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        with open(wav,"rb") as f: return f.read()[44:]
    finally:
        for p in [mp3, wav]:
            if p and os.path.exists(p):
                try: os.unlink(p)
                except: pass

# â”€â”€ Gemini voice gender mapping (Edge fallback voice matching) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
GEMINI_MALE_VOICES = {"Schedar","Algenib","Charon","Fenrir","Iapetus","Orus","Achird",
                     "Alnilam","Gacrux","Achernar","Puck","Enceladus","Umbriel",
                     "Algieba","Rasalgethi","Sadaltager","Zubenelgenubi"}
GEMINI_FEMALE_VOICES = {"Aoede","Kore","Leda","Zephyr","Erinome","Callirhoe","Callirrhoe",
                       "Despina","Laomedeia","Autonoe","Pulcherrima","Vindemiatrix",
                       "Sadachbia","Sulafat"}
EDGE_VOICE_MALE   = "bn-IN-BashkarNeural"
EDGE_VOICE_FEMALE = "bn-IN-TanishaaNeural"

def pick_edge_voice(gemini_voice, user_choice="auto"):
    if user_choice and user_choice != "auto":
        return user_choice
    if gemini_voice in GEMINI_MALE_VOICES:   return EDGE_VOICE_MALE
    if gemini_voice in GEMINI_FEMALE_VOICES: return EDGE_VOICE_FEMALE
    return EDGE_VOICE_FEMALE


# â”€â”€ SMART KEY POOL: per-key cooldown timer, RPM counter, healthy-first â”€â”€
def _init_key_pool(job, n_keys):
    """Job dict-à¦ key pool state initialize à¦•à¦°à§‡ (idempotent)."""
    if "key_pool" not in job or len(job.get("key_pool", [])) != n_keys:
        job["key_pool"] = [
            {"idx": i, "cooldown_until": 0.0, "rpm_count": 0,
             "rpm_window_start": time.time(),
             "total_ok": 0, "total_429": 0, "total_err": 0,
             "consecutive_429": 0,    # for exponential backoff
             "permanently_dead": False, # 403/invalid key
             "last_status": "ready"}
            for i in range(n_keys)
        ]
    else:
        # Migrate older pool entries to ensure new fields exist
        for k in job["key_pool"]:
            k.setdefault("consecutive_429", 0)
            k.setdefault("permanently_dead", False)

def _pick_healthy_key(job, now):
    """Return key info of next healthy key (excludes dead keys), or None."""
    pool = job["key_pool"]
    # RPM window reset (60s)
    for k in pool:
        if now - k["rpm_window_start"] >= 60:
            k["rpm_window_start"] = now
            k["rpm_count"] = 0
    # Healthy = not dead AND cooldown expired
    healthy = [k for k in pool if not k.get("permanently_dead") and k["cooldown_until"] <= now]
    if not healthy:
        return None
    # Smart priority: fewer recent calls â†’ ones that have had successes â†’ lowest idx
    healthy.sort(key=lambda k: (k["rpm_count"], -k["total_ok"], k["idx"]))
    return healthy[0]

def _next_cooldown_eta(job, now):
    """à¦•à¦–à¦¨ à¦•à§‹à¦¨à§‹ (alive) key healthy à¦¹à¦¬à§‡ â€” à¦¸à¦¬à¦šà§‡à¦¯à¦¼à§‡ à¦•à¦¾à¦›à§‡à¦° timeà¥¤"""
    pool = job["key_pool"]
    alive = [k for k in pool if not k.get("permanently_dead")]
    if not alive:
        return -1  # all keys dead
    futures = [k["cooldown_until"] - now for k in alive if k["cooldown_until"] > now]
    return max(0, int(min(futures))) if futures else 0

def _alive_keys_count(job):
    return sum(1 for k in job.get("key_pool", []) if not k.get("permanently_dead"))


# â”€â”€ synthesize_segment: SMART POOL + INFINITE RETRY + KEY DELAY â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
async def synthesize_segment(*, text, voice, language="Bengali",
                              gemini_keys, edge_pitch="-5Hz", edge_rate="+12%",
                              edge_voice="auto", gemini_only=True,
                              key_idx_ref=None, job=None, seg_label="",
                              cooldown_sec=300, backup_after_rounds=3,
                              max_rounds=999, key_delay_sec=2.0,
                              rpm_limit=15):
    """
    V3 STRATEGY:
      â€¢ Smart pool: healthy keys first, 429 à¦¹à¦¿à¦Ÿ key 5min cooldown
      â€¢ Key-à¦à¦° à¦®à¦¾à¦à§‡ 2s delay (burst rate-limit à¦à¦¡à¦¼à¦¾à¦¨à§‹)
      â€¢ RPM tracking (default 15/min Gemini limit)
      â€¢ 429 â†’ à¦“à¦‡ key 300s cooldown à¦¤à§‡ à¦¯à¦¾à¦¬à§‡ â†’ à¦…à¦¨à§à¦¯ healthy key à¦šà§‡à¦·à§à¦Ÿà¦¾
      â€¢ à¦¸à¦¬ key cooling â†’ à¦¤à¦–à¦¨ cooldown ETA wait à¦•à¦°à§‡ retry
      â€¢ 3+ round-à¦ backup model (gemini-2.5-pro-preview-tts) alternately
      â€¢ gemini_only=True â†’ infinite retry, never Edge
    """
    def _log(msg, lvl="info"):
        if job is not None: job_log(job, msg, lvl)

    lbl = seg_label or "seg"
    ev  = pick_edge_voice(voice, edge_voice)

    # ---- Cache check (main + backup) ----
    if gemini_keys:
        for mdl in (GEMINI_TTS_MODEL, GEMINI_TTS_BACKUP_MODEL):
            ck = _cache_key(text, voice, language, mdl)
            cached = _cache_get(ck)
            if cached:
                _log(f"[{lbl}] ðŸ’¾ Cache hit â€” API skip", "success")
                with open(cached,"rb") as f: wav_bytes = f.read()
                return wav_bytes[44:], "cache"

    # ---- No keys â†’ Edge (or user explicitly chose Edge) ----
    if not gemini_keys:
        _log(f"[{lbl}] ðŸŽ™ Edge-TTS ({ev})")
        pcm = await _edge_tts_call(text, voice=ev, pitch=edge_pitch, rate=edge_rate)
        return pcm, f"edge({ev})"

    # Initialize key pool in job state
    _init_key_pool(job, len(gemini_keys))

    round_num = 0
    keys_tried_this_round = set()

    while True:
        if job and job.get("stop_requested"):
            raise InterruptedError("stopped")

        now = time.time()
        kinfo = _pick_healthy_key(job, now)

        if kinfo is None:
            alive_n = _alive_keys_count(job)
            if alive_n == 0:
                # à¦¸à¦¬ key DEAD â€” Gemini-only mode-à¦ à¦à¦Ÿà¦¾ fatal
                _log(f"[{lbl}] ðŸ’€ à¦¸à¦¬ {len(gemini_keys)}à¦Ÿà¦¾ key permanently dead â€” job à¦¥à¦¾à¦®à¦›à§‡", "error")
                raise RuntimeError("all_keys_dead")
            eta = _next_cooldown_eta(job, now)
            _log(f"[{lbl}] ðŸ”´ {alive_n}à¦Ÿà¦¿ alive key à¦¸à¦¬ cooldown-à¦ â€” {eta}s à¦…à¦ªà§‡à¦•à§à¦·à¦¾â€¦", "warn")
            for _ in range(max(1, eta)):
                if job and job.get("stop_requested"): raise InterruptedError("stopped")
                await asyncio.sleep(1)
            keys_tried_this_round.clear()
            round_num += 1
            continue

        ki   = kinfo["idx"]
        key  = gemini_keys[ki]
        klbl = f"key#{ki+1}"

        # Avoid trying same key twice in same round
        if ki in keys_tried_this_round:
            # All healthy keys exhausted this round â€” small pause, new round
            keys_tried_this_round.clear()
            round_num += 1
            await asyncio.sleep(0.5)
            continue
        keys_tried_this_round.add(ki)

        # Backup model alternation (round-based, not key-based)
        use_backup = (round_num >= backup_after_rounds and round_num % 2 == 1)
        cur_model  = GEMINI_TTS_BACKUP_MODEL if use_backup else GEMINI_TTS_MODEL
        mdl_lbl    = "backup" if use_backup else "main"

        # RPM check
        kinfo["rpm_count"] += 1
        _log(f"[{lbl}] ðŸ”‘ {klbl} ({mdl_lbl}) RPM:{kinfo['rpm_count']}/{rpm_limit} â†’ trying (r{round_num+1})...")
        kinfo["last_status"] = "trying"
        save_job(job)

        try:
            pcm = await _gemini_tts_call(text, voice, key, language, model=cur_model)
            # â”€â”€â”€ SUCCESS: reset cooldown + backoff counter â”€â”€â”€
            kinfo["total_ok"]        += 1
            kinfo["last_status"]      = "ok"
            kinfo["cooldown_until"]   = 0.0   # immediately healthy
            kinfo["consecutive_429"]  = 0     # reset exponential backoff
            _log(f"[{lbl}] âœ… {klbl} OK ({mdl_lbl}) â€” total_ok:{kinfo['total_ok']}, 429s:{kinfo['total_429']}", "success")
            save_job(job)

            # Cache save
            ck = _cache_key(text, voice, language, cur_model)
            fd, tw = tempfile.mkstemp(suffix=".wav"); os.close(fd)
            with open(tw,"wb") as f: f.write(pcm_to_wav(pcm))
            _cache_set(ck, tw)
            try: os.unlink(tw)
            except: pass

            return pcm, f"gemini-{mdl_lbl}({klbl})"

        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code == 429:
                # â”€â”€ ADAPTIVE EXPONENTIAL BACKOFF: 60 â†’ 120 â†’ 240 â†’ cap 300 â”€â”€
                kinfo["total_429"]       += 1
                kinfo["consecutive_429"] += 1
                cn = kinfo["consecutive_429"]
                adaptive_cd = min(300, 60 * (2 ** (cn - 1)))   # 60,120,240,300,300...
                kinfo["cooldown_until"]   = time.time() + adaptive_cd
                kinfo["last_status"]      = "limited"
                _log(f"[{lbl}] â³ {klbl} â†’ 429 â†’ {adaptive_cd}s cooldown "
                     f"(consecutive:{cn}, 429s:{kinfo['total_429']})", "warn")
            elif code in (401, 403):
                # â”€â”€ PERMANENT: invalid/forbidden API key â†’ remove from pool â”€â”€
                kinfo["total_err"]       += 1
                kinfo["permanently_dead"] = True
                kinfo["last_status"]      = f"DEAD_{code}"
                _log(f"[{lbl}] ðŸ’€ {klbl} HTTP {code} â†’ permanently removed from pool "
                     f"(invalid/forbidden key)", "error")
            elif code in (400, 404):
                # â”€â”€ Likely model/request issue â€” also remove from pool â”€â”€
                kinfo["total_err"]       += 1
                kinfo["permanently_dead"] = True
                kinfo["last_status"]      = f"DEAD_{code}"
                _log(f"[{lbl}] ðŸ’€ {klbl} HTTP {code} â†’ permanently removed (bad request)", "error")
            else:
                # 5xx / transient â€” short cooldown
                kinfo["total_err"]       += 1
                kinfo["cooldown_until"]   = time.time() + 30
                kinfo["last_status"]      = f"http_{code}"
                _log(f"[{lbl}] âŒ {klbl} HTTP {code} â†’ 30s cooldown", "error")
            save_job(job)
        except InterruptedError:
            raise
        except Exception as e:
            kinfo["total_err"] += 1
            kinfo["cooldown_until"] = time.time() + 15
            kinfo["last_status"] = "error"
            _log(f"[{lbl}] âŒ {klbl} error: {str(e)[:80]} â†’ 15s cooldown", "error")
            save_job(job)

        # Key delay before next key (burst prevention)
        if not (job and job.get("stop_requested")):
            await asyncio.sleep(key_delay_sec)


# â”€â”€ Background Audio Ducking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _probe_mean_volume_db(path):
    """ffmpeg volumedetect à¦¦à¦¿à¦¯à¦¼à§‡ mean volume (dB) extract à¦•à¦°à§‡à¥¤ silent à¦¹à¦²à§‡ -91dBà¥¤"""
    try:
        r = subprocess.run(
            ["ffmpeg","-hide_banner","-nostats","-i",path,
             "-af","volumedetect","-vn","-sn","-dn","-f","null","-"],
            capture_output=True, text=True
        )
        # stderr-à¦ "mean_volume: -25.3 dB" line à¦¥à¦¾à¦•à§‡
        m = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", r.stderr)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def apply_ducking(video_path, dubbed_wav, out_path,
                  bg_target_db=-22.0, voice_vol="2dB",
                  bg_floor_db=-50.0, max_bg_boost_db=6.0):
    """
    Enhanced merge_background_and_vocals (zip reference + amplitude check):

    Step 1: video-à¦à¦° audio mean volume probe à¦•à¦°à§‹
    Step 2: à¦¯à¦¦à¦¿ audio silent à¦¬à¦¾ floor-à¦à¦° à¦¨à¦¿à¦šà§‡ (-50dB), bg=0 (mute)
    Step 3: à¦¨à¦¯à¦¼à¦¤à§‹ bg_target_db (default -22dB)-à¦ normalize à¦•à¦°à¦¾à¦° à¦œà¦¨à§à¦¯ gain à¦¹à¦¿à¦¸à¦¾à¦¬ à¦•à¦°à§‹
            (à¦•à¦–à¦¨à¦“ max_bg_boost_db = +6dB à¦à¦° à¦¬à§‡à¦¶à¦¿ boost à¦•à¦°à¦¬ à¦¨à¦¾ â€” noise amplify à¦à¦¡à¦¼à¦¾à¦¨à§‹)
    Step 4: voice + normalized bg â†’ amix
    """
    mean_db = _probe_mean_volume_db(video_path)

    if mean_db is None or mean_db <= bg_floor_db:
        # Effectively silent original audio â€” just use voice
        bg_gain = -60.0
    else:
        # Normalize to target â€” but cap boost
        desired_gain = bg_target_db - mean_db
        bg_gain = min(max_bg_boost_db, desired_gain)
        # Don't make it louder than voice (cap at -3dB always)
        bg_gain = min(bg_gain, -3.0)

    fc = (
        f"[0:a]volume={bg_gain:.2f}dB[bg];"
        f"[1:a]volume={voice_vol}[voice];"
        f"[bg][voice]amix=inputs=2:duration=first:normalize=0[aout]"
    )
    subprocess.run([
        "ffmpeg","-hide_banner","-y",
        "-i",video_path,"-i",dubbed_wav,
        "-filter_complex",fc,
        "-map","0:v:0","-map","[aout]",
        "-c:v","copy","-c:a","aac","-b:a","192k",
        "-shortest","-movflags","+faststart",out_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# â”€â”€ Background worker â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _run_tts_job(job_id):
    job = load_job(job_id)
    if not job: return

    job["status"]     = "running"
    job["started_at"] = time.time()
    job["logs"]       = job.get("logs", [])
    job_log(job, f"ðŸš€ Job à¦¶à§à¦°à§ â€” {job['total']} segments")
    save_job(job)

    segments     = job["segments"]
    tts_provider = job.get("tts_provider", "gemini")
    keys         = [] if tts_provider == "edge" else job.get("gemini_keys", [])
    voice        = job.get("voice", "Charon")
    language     = job.get("language", "Bengali")
    pitch        = job.get("edge_pitch", "-5Hz")
    rate         = job.get("edge_rate", "+12%")
    edge_voice   = job.get("edge_voice", "auto")
    gemini_only  = bool(job.get("gemini_only", True))
    key_idx      = [job.get("_key_idx", 0)]
    do_concat    = bool(job.get("concat_segments", False))   # BUG FIX: default OFF
    do_silence   = job.get("remove_silence", True)

    # concat_close_transcripts (zip feature)
    if do_concat and not job.get("_concat_done"):
        orig_count = len(segments)
        segments = concat_close_transcripts(
            [{"start":s["start"],"end":s["end"],"text":s["text"]} for s in segments],
            threshold=2.0, max_chunk_sec=15
        )
        job["segments"] = [
            {"idx":i,"start":s["start"],"end":s["end"],
             "text":s["text"],"status":"pending"}
            for i,s in enumerate(segments)
        ]
        job["total"]       = len(segments)
        job["_concat_done"]= True
        job_log(job, f"ðŸ”— Concat: {orig_count} â†’ {len(segments)} segments (max 15s)")
        save_job(job)
        segments = job["segments"]

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        total_segs = len(segments)
        for i in range(total_segs):
            # â”€â”€ CRITICAL FIX: reload job AND re-bind seg via index â”€â”€
            # à¦ªà§à¦°à¦¨à§‹ version-à¦ seg local ref â†’ save_job() reload à¦•à¦°à¦¾à¦° à¦ªà¦°
            # status="done" hariye jetoà¥¤ index à¦¦à¦¿à¦¯à¦¼à§‡ fresh job["segments"][i] use à¦•à¦°à¦¿à¥¤
            job = load_job(job_id)
            if not job: return
            if job.get("stop_requested"):
                job_log(job, "ðŸ›‘ Stopped by user", "warn")
                job["status"] = "stopped"; save_job(job); return

            seg = job["segments"][i]
            wav_path = _seg_wav_path(job_id, i)

            # Resume: file already exists?
            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 100:
                seg["status"]   = "done"
                seg["wav_path"] = wav_path
                job["done"]     = sum(1 for s in job["segments"] if s.get("status")=="done")
                job_log(job, f"[seg{i+1}] â­ Resume â€” file exists")
                save_job(job); continue

            seg["status"]  = "processing"
            job["current"] = i
            job_log(job, f"[seg{i+1}/{job['total']}] ðŸ“ \"{seg['text'][:40]}...\"")
            save_job(job)

            try:
                pcm, used_prov = loop.run_until_complete(
                    synthesize_segment(
                        text        = seg["text"],
                        voice       = voice,
                        language    = language,
                        gemini_keys = keys,
                        edge_pitch  = pitch,
                        edge_rate   = rate,
                        edge_voice  = edge_voice,
                        gemini_only = gemini_only,
                        key_idx_ref = key_idx,
                        job         = job,
                        seg_label   = f"seg{i+1}",
                    )
                )

                wav_bytes = pcm_to_wav(pcm)
                with open(wav_path,"wb") as f: f.write(wav_bytes)

                # Silence removal (only if explicitly enabled)
                if do_silence and not used_prov.startswith("cache"):
                    before = os.path.getsize(wav_path)
                    remove_silence(wav_path)
                    after  = os.path.getsize(wav_path)
                    if before != after:
                        job_log(job, f"[seg{i+1}] âœ‚ï¸ Silence removed ({before//1024}â†’{after//1024}kb)")

                # â”€â”€ Re-fetch latest job + bulletproof persist â”€â”€
                # CRITICAL: synthesize_segment calls save_job many times for
                # logging/key_pool updates. To avoid races, we re-fetch
                # immediately AFTER synth and persist seg.status atomically.
                for _retry in range(3):
                    job = load_job(job_id)
                    if not job: return
                    seg = job["segments"][i]
                    seg["status"]   = "done"
                    seg["wav_path"] = wav_path
                    seg["provider"] = used_prov
                    job["done"]     = sum(1 for s in job["segments"] if s.get("status")=="done")
                    job["_key_idx"] = key_idx[0]
                    save_job(job)
                    # Verify it actually persisted
                    _v = load_job(job_id)
                    if _v and _v["segments"][i].get("status") == "done":
                        break
                    job_log(job, f"[seg{i+1}] âš ï¸ persist retry {_retry+1}/3", "warn")
                    time.sleep(0.05)

            except InterruptedError:
                job = load_job(job_id) or job
                job["status"] = "stopped"
                job_log(job, "ðŸ›‘ Stopped", "warn")
                save_job(job); return
            except Exception as e:
                # Re-fetch first
                job = load_job(job_id) or job
                seg = job["segments"][i]
                seg["status"] = "error"
                seg["error"]  = str(e)
                job_log(job, f"[seg{i+1}] âŒ {str(e)[:80]}", "error")

            # (done & _key_idx persisted atomically inside try block above)

        # â”€â”€ Final: AUTHORITATIVE done count from disk + wav file presence â”€â”€
        # SAFETY NET: even if a flag missed persisting, count wav files on disk
        job = load_job(job_id) or job
        real_done = 0
        for _i, _s in enumerate(job["segments"]):
            _wp = _seg_wav_path(job_id, _i)
            if os.path.exists(_wp) and os.path.getsize(_wp) > 100:
                if _s.get("status") != "done":
                    _s["status"]   = "done"
                    _s["wav_path"] = _wp
                    job_log(job, f"[seg{_i+1}] ðŸ”§ wav exists but flag was stale â€” fixed", "info")
                real_done += 1
        job["done"]   = real_done
        job["status"] = "complete"
        job_log(job, f"ðŸŽ‰ à¦¶à§‡à¦·! {real_done}/{job['total']} done", "success")
        save_job(job)

    except Exception as e:
        job["status"] = "error"; job["error"] = str(e)
        job_log(job, f"âŒ Fatal: {e}", "error")
        save_job(job)
    finally:
        loop.close()


# â”€â”€ TTS Routes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route("/tts/start", methods=["POST"])
def tts_start():
    data           = request.get_json(force=True)
    segments       = data.get("segments", [])
    gemini_keys    = data.get("gemini_keys", [])
    voice          = data.get("voice", "Charon")
    edge_pitch     = data.get("edge_pitch", "-5Hz")
    edge_rate      = data.get("edge_rate", "+12%")
    edge_voice     = data.get("edge_voice", "auto")
    tts_provider   = (data.get("tts_provider", "gemini") or "gemini").strip().lower()
    gemini_only    = bool(data.get("gemini_only", True))
    resume_id      = data.get("resume_id", "").strip()
    remove_silence    = bool(data.get("remove_silence", False))   # default OFF
    language          = data.get("language", "Bengali")
    concat_segments   = bool(data.get("concat_segments", False))   # default OFF

    if tts_provider == "edge":
        gemini_keys = []
        gemini_only = False

    # ---- Resume / Restart (BUG FIX) ----
    if resume_id:
        old_job = load_job(resume_id)
        if old_job:
            st = old_job.get("status", "")
            if st in ("running", "pending"):
                return jsonify({"job_id": resume_id, "resumed": True})
            old_job["stop_requested"] = False
            old_job["status"]         = "pending"
            old_job["gemini_keys"]    = gemini_keys or old_job.get("gemini_keys", [])
            old_job["tts_provider"]   = tts_provider
            old_job["gemini_only"]    = gemini_only
            old_job["edge_voice"]     = edge_voice
            old_job["edge_pitch"]     = edge_pitch
            old_job["edge_rate"]      = edge_rate
            old_job["voice"]          = voice or old_job.get("voice", "Charon")
            # KEY POOL RESET: resume-à¦ à¦¸à¦¬ key cooldown clear à¦•à¦°à§‹
            if "key_pool" in old_job:
                for k in old_job["key_pool"]:
                    k["cooldown_until"]    = 0.0
                    k["rpm_count"]         = 0
                    k["rpm_window_start"]  = time.time()
                    k["last_status"]       = "ready"
            for s in old_job.get("segments", []):
                # WAV FILE RECHECK: "done" segment à¦à¦° file actually à¦†à¦›à§‡ à¦•à¦¿à¦¨à¦¾ verify à¦•à¦°à§‹
                if s.get("status") == "done":
                    wp = s.get("wav_path", "")
                    if not wp or not os.path.exists(wp) or os.path.getsize(wp) <= 100:
                        s["status"]   = "pending"
                        s["wav_path"] = ""
                elif s.get("status") != "done":
                    s["status"] = "pending"
            old_job["done"] = sum(1 for s in old_job["segments"] if s.get("status") == "done")
            job_log(old_job, f"ðŸ”„ Resume â€” restart job (done={old_job['done']}/{old_job['total']})", "info")
            save_job(old_job)
            threading.Thread(target=_run_tts_job, args=(resume_id,), daemon=True).start()
            return jsonify({"job_id": resume_id, "resumed": True})

    if not segments:
        return jsonify({"error": "segments required"}), 400

    job_id = uuid.uuid4().hex[:12]
    job = {
        "id":               job_id,
        "status":           "pending",
        "created_at":       time.time(),
        "gemini_keys":      gemini_keys,
        "tts_provider":     tts_provider,
        "gemini_only":      gemini_only,
        "voice":            voice,
        "language":         language,
        "edge_pitch":       edge_pitch,
        "edge_rate":        edge_rate,
        "edge_voice":       edge_voice,
        "remove_silence":   remove_silence,
        "concat_segments":  concat_segments,
        "done":           0,
        "total":          len(segments),
        "current":        -1,
        "_key_idx":       0,
        "stop_requested": False,
        "logs":           [],
        "segments": [
            {"idx": i, "start": s["start"], "end": s["end"],
             "text": s["text"], "status": "pending"}
            for i, s in enumerate(segments)
        ],
    }
    save_job(job)
    threading.Thread(target=_run_tts_job, args=(job_id,), daemon=True).start()
    return jsonify({"job_id": job_id, "resumed": False})


@app.route("/tts/stop/<job_id>", methods=["POST"])
def tts_stop(job_id):
    job = load_job(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    job["stop_requested"] = True
    job_log(job, "ðŸ›‘ Stop requested by user", "warn")
    save_job(job)
    return jsonify({"ok": True})


@app.route("/tts/status/<job_id>")
def tts_status(job_id):
    # Allow client to resume log stream from a given index
    try:
        start_log_idx = int(request.args.get("log_from", "0"))
    except Exception:
        start_log_idx = 0

    def generate():
        last_done    = -1
        last_log_len = start_log_idx
        while True:
            job = load_job(job_id)
            if not job:
                yield f"event: error\ndata: {json.dumps({'msg': 'job not found'}, ensure_ascii=False)}\n\n"
                return

            done   = job.get("done", 0)
            total  = job.get("total", 1)
            curr   = job.get("current", -1)
            status = job.get("status", "")
            logs   = job.get("logs", [])

            # à¦¨à¦¤à§à¦¨ log entries à¦ªà¦¾à¦ à¦¾à¦“
            if len(logs) > last_log_len:
                new_logs = logs[last_log_len:]
                last_log_len = len(logs)
                yield (
                    f"event: log\n"
                    f"data: {json.dumps({'entries': new_logs}, ensure_ascii=False)}\n\n"
                )

            if done != last_done or status in ("complete", "error", "stopped"):
                seg_text = ""
                if 0 <= curr < len(job["segments"]):
                    seg_text = job["segments"][curr].get("text", "")[:30]

                # key reset time estimate
                key_reset_info = []
                keys = job.get("gemini_keys", [])
                for ki in range(len(keys)):
                    key_reset_info.append({
                        "key_num": ki + 1,
                        "label":   f"key#{ki+1}",
                    })

                yield (
                    f"event: progress\n"
                    f"data: {json.dumps({'done': done, 'total': total, 'current': curr, 'seg_text': seg_text, 'status': status}, ensure_ascii=False)}\n\n"
                )
                last_done = done

            if status in ("complete", "error", "stopped"):
                results = []
                for s in job["segments"]:
                    wav_path = s.get("wav_path", "")
                    if s.get("status") == "done" and wav_path and os.path.exists(wav_path):
                        with open(wav_path, "rb") as f:
                            wav_bytes = f.read()
                        results.append({
                            "idx":      s["idx"],
                            "start":    s["start"],
                            "end":      s["end"],
                            "pcm_b64":  base64.b64encode(wav_bytes[44:]).decode(),
                            "provider": s.get("provider", "?"),
                        })
                    else:
                        results.append({
                            "idx":   s["idx"],
                            "start": s["start"],
                            "end":   s["end"],
                            "error": s.get("error", "failed"),
                        })
                ok_n = sum(1 for r in results if r.get("pcm_b64"))

                # â”€â”€â”€ DIAGNOSTIC DUMP (just before video build trigger) â”€â”€â”€
                done_count   = sum(1 for s in job["segments"] if s.get("status") == "done")
                pending_count= sum(1 for s in job["segments"] if s.get("status") in ("pending","processing"))
                failed_count = sum(1 for s in job["segments"] if s.get("status") in ("error","failed"))
                _diag_lines = [
                    "â•â•â•â•â•â•â•â•â•â•â• DIAGNOSTIC DUMP (pre-build) â•â•â•â•â•â•â•â•â•â•â•",
                    f"job_id          = {job_id}",
                    f"job.status      = {status}",
                    f"job.done        = {job.get('done', 0)}",
                    f"job.total       = {job.get('total', 0)}",
                    f"done_count      = {done_count}",
                    f"pending_count   = {pending_count}",
                    f"failed_count    = {failed_count}",
                    f"results_ok      = {ok_n}/{len(results)}",
                ]
                for _i, _s in enumerate(job["segments"][:3]):
                    _wp  = _s.get("wav_path", "")
                    _wp_exists = bool(_wp and os.path.exists(_wp))
                    _wp_size   = os.path.getsize(_wp) if _wp_exists else 0
                    # find corresponding result
                    _r = next((r for r in results if r.get("idx") == _s.get("idx")), {})
                    _pcm = _r.get("pcm_b64", "")
                    _diag_lines.append(
                        f"seg[{_i}] status={_s.get('status','?'):10s} "
                        f"wav_path={_wp or '(none)'} "
                        f"wav_exists={_wp_exists} wav_size={_wp_size}B "
                        f"pcm_b64_exists={bool(_pcm)} pcm_b64_length={len(_pcm)}"
                    )
                _diag_lines.append("â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•")
                # Print to server stdout (Railway logs)
                for _ln in _diag_lines:
                    print(f"[DIAG] {_ln}", flush=True)
                # Also push into job logs (TTS Live Log panel)
                for _ln in _diag_lines:
                    job_log(job, _ln, "info")
                save_job(job)
                # And emit as a single SSE log event so the frontend shows it now (not after refresh)
                yield (
                    f"event: log\n"
                    f"data: {json.dumps({'entries': [{'t': round(time.time(),2), 'msg': _ln, 'lvl': 'info'} for _ln in _diag_lines]}, ensure_ascii=False)}\n\n"
                )

                yield (
                    f"event: done\n"
                    f"data: {json.dumps({'results': results, 'status': status, 'ok': ok_n, 'total': len(results)}, ensure_ascii=False)}\n\n"
                )
                return

            time.sleep(1.5)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Content-Type": "text/event-stream; charset=utf-8"},
    )


@app.route("/tts/active")
def tts_active():
    """List recent jobs (newest first) â€” used for page-refresh recovery."""
    out = []
    try:
        files = sorted(os.listdir(JOBS_DIR), key=lambda f: -os.path.getmtime(os.path.join(JOBS_DIR, f)))
        for fn in files[:20]:
            if not fn.endswith(".json"): continue
            try:
                j = json.loads(open(os.path.join(JOBS_DIR, fn)).read())
            except: continue
            out.append({
                "id":       j.get("id"),
                "status":   j.get("status"),
                "done":     j.get("done", 0),
                "total":    j.get("total", 0),
                "current":  j.get("current", -1),
                "created":  j.get("created_at", 0),
                "log_count": len(j.get("logs", [])),
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"jobs": out})


@app.route("/tts/job/<job_id>")
def tts_job_get(job_id):
    job = load_job(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    safe = {k: v for k, v in job.items() if k not in ("gemini_keys", "_key_idx")}
    return jsonify(safe)


@app.route("/tts/job/<job_id>", methods=["DELETE"])
def tts_job_delete(job_id):
    job = load_job(job_id)
    if job:
        for i in range(job.get("total", 0)):
            p = _seg_wav_path(job_id, i)
            try: os.unlink(p)
            except: pass
    p = _job_path(job_id)
    try: os.unlink(p)
    except: pass
    return jsonify({"ok": True})


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PLATFORM / PROXY / XRAY
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def get_platform(url):
    if re.search(r'instagram\.com', url, re.I):             return 'instagram'
    if re.search(r'tiktok\.com|vm\.tiktok\.com', url, re.I): return 'tiktok'
    if re.search(r'kuaishou\.com|v\.kuaishou\.com', url, re.I): return 'kuaishou'
    if re.search(r'youtube\.com|youtu\.be', url, re.I):     return 'youtube'
    return 'unknown'

XRAY_BIN    = os.environ.get("XRAY_BIN", "/usr/local/bin/xray")
XRAY_CONFIG = os.environ.get("XRAY_CONFIG", "/app/data/xray-config.json")
SOCKS_PORT  = int(os.environ.get("XRAY_SOCKS_PORT", "10808"))
LOCAL_PROXY = f"socks5://127.0.0.1:{SOCKS_PORT}"
CONFIG_FILE = os.environ.get("CONFIG_FILE", "/app/data/config.json")

_xray_proc = None
_xray_lock = threading.Lock()

def load_config():
    try:
        if os.path.exists(CONFIG_FILE):
            return json.loads(open(CONFIG_FILE).read())
    except: pass
    return {}

def save_config(obj):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    open(CONFIG_FILE, "w").write(json.dumps(obj, indent=2))

def _decode_vmess(link):
    import base64 as b64m
    raw = b64m.b64decode(link.replace("vmess://","").strip() + "==").decode()
    j = json.loads(raw)
    return dict(type="vmess", address=j["add"], port=int(j["port"]), uuid=j["id"],
                alterId=int(j.get("aid",0)), network=j.get("net","tcp"),
                security="tls" if j.get("tls")=="tls" else "none",
                host=j.get("host",j["add"]), path=j.get("path","/"),
                sni=j.get("sni",j.get("host",j["add"])))

def _decode_vless(link):
    import urllib.parse as up
    m = re.match(r"vless://([^@]+)@([^:/?]+):(\d+)(\?[^#]*)?", link, re.I)
    if not m: raise ValueError("bad vless link")
    uuid_,host,port,qs = m.groups()
    p = dict(up.parse_qsl((qs or "").lstrip("?")))
    return dict(type="vless", address=host, port=int(port), uuid=uuid_,
                network=p.get("type","tcp"), security=p.get("security","none"),
                host=p.get("host",host), path=p.get("path","/"),
                sni=p.get("sni",host), flow=p.get("flow",""))

def _build_xray_cfg(d):
    if d["type"] == "vmess":
        outbound = {"tag":"proxy","protocol":"vmess",
            "settings":{"vnext":[{"address":d["address"],"port":d["port"],
                "users":[{"id":d["uuid"],"alterId":d.get("alterId",0),"security":"auto"}]}]},
            "streamSettings":_stream(d)}
    else:
        outbound = {"tag":"proxy","protocol":"vless",
            "settings":{"vnext":[{"address":d["address"],"port":d["port"],
                "users":[{"id":d["uuid"],"encryption":"none","flow":d.get("flow","")}]}]},
            "streamSettings":_stream(d)}
    return {"log":{"loglevel":"warning"},
        "inbounds":[{"tag":"socks-in","listen":"127.0.0.1","port":SOCKS_PORT,
            "protocol":"socks","settings":{"auth":"noauth","udp":True}}],
        "outbounds":[outbound,{"tag":"direct","protocol":"freedom","settings":{}}]}

def _stream(d):
    ss = {"network": d.get("network","tcp")}
    if d.get("security") == "tls":
        ss["security"] = "tls"
        ss["tlsSettings"] = {"serverName": d.get("sni") or d.get("host") or d["address"]}
    if d.get("network") == "ws":
        ss["wsSettings"] = {"path": d.get("path","/"), "headers":{"Host": d.get("host","")}}
    return ss

def start_xray(link=None):
    global _xray_proc
    link = link or os.environ.get("VMESS_LINK","") or os.environ.get("PROXY_LINK","")
    if not link: return False
    link = link.strip()
    if re.match(r"socks5?://", link, re.I):
        os.environ["YTDLP_PROXY"] = link
        return True
    try:
        if re.match(r"vmess://", link, re.I):   d = _decode_vmess(link)
        elif re.match(r"vless://", link, re.I): d = _decode_vless(link)
        else: raise ValueError("unsupported scheme")
        cfg = _build_xray_cfg(d)
        os.makedirs(os.path.dirname(XRAY_CONFIG), exist_ok=True)
        open(XRAY_CONFIG,"w").write(json.dumps(cfg, indent=2))
        with _xray_lock:
            stop_xray()
            _xray_proc = subprocess.Popen([XRAY_BIN,"run","-c",XRAY_CONFIG],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.environ["YTDLP_PROXY"] = LOCAL_PROXY
        return True
    except Exception as e:
        print(f"[xray] start failed: {e}", flush=True)
        return False

def stop_xray():
    global _xray_proc
    if _xray_proc:
        try: _xray_proc.kill()
        except: pass
        _xray_proc = None

def _apply_saved_config():
    cfg = load_config()
    for k,v in cfg.items(): os.environ[k] = v
    link = cfg.get("VMESS_LINK","") or cfg.get("PROXY_LINK","")
    if link: start_xray(link)

_apply_saved_config()

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# YT-DLP
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

YT_STRATEGIES = [
    {"name": "web_embedded", "client": "web_embedded", "max_sec": 150},
    {"name": "mweb",         "client": "mweb",         "max_sec": 300},
    {"name": "ios",          "client": "ios",           "max_sec": 120},
    {"name": "android",      "client": "android",       "max_sec": 120},
    {"name": "web_safari",   "client": "web_safari",    "max_sec": 150},
    {"name": "tv_simply",    "client": "tv_simply",     "max_sec": 60},
]

def _build_ytdlp_base(is_yt=False, log_fn=None):
    args = [
        "--no-warnings","--progress","--newline","--no-playlist",
        "--retries","10","--fragment-retries","10","--retry-sleep","3",
        "--socket-timeout","60",
        "--user-agent","Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "--add-header","Accept-Language:en-US,en;q=0.9",
        "--geo-bypass","--hls-prefer-native",
        "--concurrent-fragments","4","-N","4","--http-chunk-size","10M",
    ]
    proxy   = os.environ.get("YTDLP_PROXY","").strip()
    cookies = os.environ.get("COOKIES_FILE", COOKIES_FILE)
    if is_yt:
        try:
            subprocess.run(["deno","--version"], capture_output=True, check=True)
            args += ["--extractor-args","youtube:jsruntime=deno"]
        except: pass
        if os.path.exists(cookies) and os.path.getsize(cookies) > 100:
            args += ["--cookies", cookies]
        if proxy: args += ["--proxy", proxy]
    return args

def ytdlp_download(url, out_dir, job_log_fn=None):
    platform = get_platform(url)
    is_yt    = platform == "youtube"
    fmt      = "b[height<=720][ext=mp4][protocol*=https]/bv*[height<=720][ext=mp4]+ba[ext=m4a]/bv*+ba/b[ext=mp4]/b"
    out_tpl  = os.path.join(out_dir, "dl_%(id)s.%(ext)s")
    strategies = YT_STRATEGIES if is_yt else [{"name":"direct","client":None,"max_sec":180}]
    errors = []

    for strat in strategies:
        try:
            for f in os.listdir(out_dir):
                if f.startswith("dl_"): os.unlink(os.path.join(out_dir, f))
        except: pass

        args = ["yt-dlp"] + _build_ytdlp_base(is_yt, job_log_fn)
        if is_yt and strat.get("client"):
            args += ["--extractor-args", f"youtube:player_client={strat['client']}"]
        if platform == "tiktok":
            args += ["--extractor-args","tiktok:api_hostname=api22-normal-c-useast2a.tiktokv.com"]
        args += ["-f", fmt, "--merge-output-format","mp4","-o",out_tpl, url]

        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=strat.get("max_sec",180))
        except subprocess.TimeoutExpired:
            errors.append(f"[{strat['name']}] timeout"); continue
        except Exception as e:
            errors.append(f"[{strat['name']}] {e}"); continue

        pick, pick_size = None, 0
        for f in os.listdir(out_dir):
            if not f.startswith("dl_") or f.endswith(".part"): continue
            sz = os.path.getsize(os.path.join(out_dir, f))
            if sz > pick_size: pick_size = sz; pick = f

        if pick and pick_size > 50*1024:
            final = os.path.join(out_dir, "downloaded.mp4")
            os.rename(os.path.join(out_dir, pick), final)
            return final

        errors.append(f"[{strat['name']}] no output (exit {getattr(result,'returncode','?')})")

    raise ValueError("All strategies failed:\n" + "\n".join(errors))

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# KUAISHOU
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

PHOTO_ID_RE = re.compile(r"/(?:short-video|video|photo)/([A-Za-z0-9_-]+)")

def extract_photo_id(url):
    m = PHOTO_ID_RE.search(url or "")
    return m.group(1) if m else None

async def wait_for_ks_api(max_wait_ms=10000):
    start = time.time()
    while (time.time()-start)*1000 < max_wait_ms:
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                if (await client.get(f"{KS_API}/docs")).status_code == 200:
                    return True
        except: pass
        await asyncio.sleep(1.5)
    return False

BLOCKED_KS_PATHS = ("/new-reco","/login","/captcha","/error","/404")

async def resolve_ks_url(url):
    if extract_photo_id(url): return url
    headers = {
        "User-Agent":"Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15",
        "Accept-Language":"zh-CN,zh;q=0.9","Referer":"https://www.kuaishou.com/",
    }
    try:
        async with httpx.AsyncClient(timeout=12, follow_redirects=True, headers=headers) as client:
            resp = await client.get(url)
            final_url = str(resp.url)
            from urllib.parse import urlparse
            path = urlparse(final_url).path
            if any(path.startswith(p) for p in BLOCKED_KS_PATHS):
                raise ValueError(f"Kuaishou blocked (redirected to {path})")
            return final_url
    except ValueError: raise
    except: return url

async def get_ks_video_url_via_api(raw_url):
    payload = {"text": raw_url}
    if KS_COOKIE: payload["cookie"] = KS_COOKIE
    if KS_PROXY:  payload["proxy"]  = KS_PROXY
    last_error = None
    for attempt in range(4):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(f"{KS_API}/detail/", json=payload)
                resp.raise_for_status()
                body = resp.json()
            data = body.get("data")
            if not data: raise ValueError(f"KS-API: {body.get('message','no data')}")
            downloads = data.get("download", [])
            if isinstance(downloads, str): downloads = downloads.split()
            if not downloads: raise ValueError("KS-API: no download URL")
            return downloads[0], data
        except Exception as e:
            last_error = e
            if attempt < 3: await asyncio.sleep(1.5)
    raise ValueError(str(last_error or "KS-API failed"))

async def get_ks_video_url_via_graphql(raw_url):
    resolved_url = await resolve_ks_url(raw_url)
    photo_id = extract_photo_id(resolved_url)
    if not photo_id: raise ValueError(f"Cannot extract photoId from {resolved_url}")
    payload = {
        "operationName":"visionVideoDetail",
        "variables":{"photoId":photo_id,"page":"detail"},
        "query":"query visionVideoDetail($photoId: String, $page: String) { visionVideoDetail(photoId: $photoId, page: $page) { photo { id caption photoUrl duration } } }",
    }
    headers = {
        "Content-Type":"application/json",
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer":f"https://www.kuaishou.com/short-video/{photo_id}",
        "Origin":"https://www.kuaishou.com","Accept":"*/*",
    }
    if KS_COOKIE: headers["Cookie"] = KS_COOKIE

    async def _try(proxy=None):
        kw = {"timeout":45}
        if proxy: kw["proxy"] = proxy
        async with httpx.AsyncClient(**kw) as client:
            resp = await client.post("https://www.kuaishou.com/graphql", headers=headers, json=payload)
            resp.raise_for_status()
            body = resp.json()
            video_url = body.get("data",{}).get("visionVideoDetail",{}).get("photo",{}).get("photoUrl")
            if not video_url:
                raise ValueError(f"GraphQL: {body.get('errors',[{}])[0].get('message','no photoUrl')}")
            photo = body.get("data",{}).get("visionVideoDetail",{}).get("photo",{})
            return video_url, {"photoId":photo.get("id") or photo_id,
                               "caption":photo.get("caption") or photo_id,
                               "duration":photo.get("duration") or 0}
    try: return await _try(None)
    except:
        if KS_PROXY: return await _try(KS_PROXY)
        raise

async def get_ks_video_url(raw_url):
    if await wait_for_ks_api(12000):
        try:
            video_url, meta = await get_ks_video_url_via_api(raw_url)
            return video_url, meta, "ks-downloader-api"
        except: pass
    video_url, meta = await get_ks_video_url_via_graphql(raw_url)
    return video_url, meta, "graphql-fallback"

async def download_video(video_url, out_path):
    headers = {
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer":"https://www.kuaishou.com/",
    }
    async with httpx.AsyncClient(timeout=180, follow_redirects=True) as client:
        async with client.stream("GET", video_url, headers=headers) as resp:
            resp.raise_for_status()
            with open(out_path,"wb") as f:
                async for chunk in resp.aiter_bytes(1024*64):
                    f.write(chunk)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# ASR
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def extract_audio(video_path, mp3_path):
    subprocess.run(
        ["ffmpeg","-y","-i",video_path,"-ar","16000","-ac","1","-b:a","128k","-vn",mp3_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

async def groq_transcribe(mp3_path, groq_keys, language="zh"):
    if not groq_keys: raise ValueError("No Groq API key provided")
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    retryable = {402,420,429}
    last_error = None
    for idx, key in enumerate(groq_keys, start=1):
        data = {"model":"whisper-large-v3","response_format":"verbose_json",
                "timestamp_granularities[]":["segment","word"]}
        if language and language != "auto": data["language"] = language
        try:
            async with httpx.AsyncClient(timeout=180) as client:
                with open(mp3_path,"rb") as f:
                    resp = await client.post(url, headers={"Authorization":f"Bearer {key}"},
                                             files={"file":("audio.mp3",f,"audio/mpeg")}, data=data)
            if resp.status_code in retryable:
                last_error = ValueError(f"key {idx} rate-limited ({resp.status_code})"); continue
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response else None
            if status in retryable or (status and status>=500):
                last_error = ValueError(f"key {idx} failed {status}"); continue
            raise
        except Exception as e:
            last_error = e
            if idx < len(groq_keys): continue
            raise
    raise ValueError(str(last_error or "All Groq keys failed"))

def has_chinese(text):
    return any('\u4e00'<=c<='\u9fff' for c in text)

def split_long_segments(segments, max_dur=8.0):
    result = []
    for seg in segments:
        start,end,text = seg["start"],seg["end"],seg["text"]
        dur = end-start
        if dur<=max_dur: result.append(seg); continue
        n = max(2, int(dur/max_dur)+1)
        words = text.split()
        time_step  = dur/n
        chunk_size = max(1, len(words)//n)
        for i in range(n):
            t_start = start+i*time_step
            t_end   = start+(i+1)*time_step if i<n-1 else end
            chunk_words = words[i*chunk_size:(i+1)*chunk_size] if i<n-1 else words[i*chunk_size:]
            chunk_text = " ".join(chunk_words).strip()
            if chunk_text:
                result.append({"start":round(t_start,3),"end":round(t_end,3),"text":chunk_text})
    return result

def parse_groq_keys(raw):
    keys = []
    for part in re.split(r"[\n,]+", raw or ""):
        part = part.strip()
        if part and part not in keys: keys.append(part)
    return keys

def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

def transcribe_stream(url, groq_keys_raw, language="zh"):
    job_id     = f"asr_{int(time.time())}"
    work_dir   = os.path.join(TEMP_DIR, job_id)
    video_path = os.path.join(TEMP_DIR, f"{job_id}.mp4")
    mp3_path   = os.path.join(TEMP_DIR, f"{job_id}.mp3")
    groq_keys  = parse_groq_keys(groq_keys_raw)
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    try:
        platform = get_platform(url)
        strategy = "yt-dlp"
        yield sse("log",{"msg":f"ðŸ” Platform: {platform.upper()}"})
        if platform=='kuaishou':
            yield sse("log",{"msg":"â³ Getting Kuaishou video URL..."})
            video_url,meta,strategy = asyncio.run(get_ks_video_url(url))
            caption=(meta or {}).get("caption") or (meta or {}).get("photoId") or "Kuaishou"
            yield sse("log",{"msg":f"âœ… Video URL found via {strategy}"})
            yield sse("log",{"msg":f"ðŸŽ¬ Source: {caption}"})
            yield sse("log",{"msg":"â¬‡ Downloading video..."})
            asyncio.run(download_video(video_url, video_path))
        else:
            yield sse("log",{"msg":f"â¬‡ Downloading via yt-dlp ({platform})..."})
            video_path = ytdlp_download(url, work_dir)
        size_mb = os.path.getsize(video_path)/1024/1024
        yield sse("log",{"msg":f"âœ… Downloaded ({size_mb:.1f} MB)"})
        yield sse("log",{"msg":"ðŸ”Š Extracting audio..."})
        extract_audio(video_path, mp3_path)
        yield sse("log",{"msg":"âœ… Audio extracted"})
        yield sse("log",{"msg":f"ðŸ¤– Sending to Groq Whisper ({language})..."})
        result = asyncio.run(groq_transcribe(mp3_path, groq_keys, language))
        yield sse("log",{"msg":f"âœ… Transcription done! ({result.get('duration',0):.1f}s)"})

        segments = []
        raw_segments = result.get("segments")
        if raw_segments:
            for seg in raw_segments:
                segments.append({"start":seg.get("start",0),"end":seg.get("end",0),
                                  "text":(seg.get("text") or "").strip()})
        else:
            words=result.get("words",[])
            current_words=[]; current_start=None
            for w in words:
                word_text=w.get("word","").strip()
                if not word_text: continue
                if current_start is None: current_start=w["start"]
                current_words.append(word_text)
                if word_text[-1] in 'à¥¤.?!\n':
                    joined="".join(current_words) if has_chinese(word_text) else " ".join(current_words)
                    segments.append({"start":round(current_start,3),"end":round(w["end"],3),"text":joined})
                    current_words=[]; current_start=None
            if current_words:
                joined="".join(current_words) if has_chinese(words[-1]["word"]) else " ".join(current_words)
                segments.append({"start":round(current_start,3),"end":round(words[-1]["end"],3),"text":joined})

        segments = split_long_segments(segments, max_dur=8.0)
        yield sse("done",{"job_id":job_id,"text":result.get("text",""),
                           "segments":segments,"duration":result.get("duration",0),
                           "strategy":strategy,"language":language})
    except Exception as e:
        yield sse("error",{"msg":str(e)})
    finally:
        try: os.unlink(video_path)
        except: pass

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# FLASK ROUTES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe")
def transcribe():
    url       = request.args.get("url","").strip()
    groq_keys = request.args.get("groq_keys","").strip() or request.args.get("groq_key","").strip()
    language  = request.args.get("language","zh").strip() or "zh"
    if not url:       return jsonify({"error":"url required"}),400
    if not groq_keys: return jsonify({"error":"groq_keys required"}),400
    return Response(stream_with_context(transcribe_stream(url,groq_keys,language)),
                    mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.route("/synthesize", methods=["POST"])
def synthesize():
    data     = request.get_json(force=True)
    text     = data.get("text","").strip()
    provider = data.get("provider","edge-tts")
    if not text: return jsonify({"error":"text required"}),400
    mp3_path=None; wav_path=None
    try:
        fd,mp3_path=tempfile.mkstemp(suffix=".mp3"); os.close(fd)
        if provider=="gemini_audio":
            gemini_key=data.get("gemini_key","").strip()
            job_id=data.get("job_id","").strip()
            start=float(data.get("start",0)); end=float(data.get("end",0))
            voice_name=data.get("voice","Gacrux")
            target_language=data.get("target_language","").strip()
            if not gemini_key: raise ValueError("gemini_key required")
            orig_audio_path=os.path.join(TEMP_DIR,f"{job_id}.mp3")
            if not os.path.exists(orig_audio_path): raise ValueError("Original audio not found")
            fd_chunk,chunk_path=tempfile.mkstemp(suffix=".mp3"); os.close(fd_chunk)
            subprocess.run(["ffmpeg","-y","-i",orig_audio_path,"-ss",str(start),"-to",str(end),
                            "-c","copy",chunk_path],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            with open(chunk_path,"rb") as f: audio_bytes=f.read()
            try: os.unlink(chunk_path)
            except: pass
            audio_b64=base64.b64encode(audio_bytes).decode("utf-8")
            lang_hint=f" Speak in {target_language}." if target_language else ""
            prompt=(f"Listen to the emotion, tone, pace, and prosody of this audio clip carefully."
                    f" Now synthesize the following text preserving the exact same emotion.{lang_hint}"
                    f" Text: '{text}'")
            payload={"contents":[{"parts":[
                {"inline_data":{"mime_type":"audio/mp3","data":audio_b64}},
                {"text":prompt}
            ]}],"generationConfig":{
                "responseModalities":["AUDIO"],
                "speechConfig":{"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":voice_name}}}
            }}
            gemini_url=(f"https://generativelanguage.googleapis.com/v1beta/models/"
                        f"gemini-2.0-flash:generateContent?key={gemini_key}")
            async def _call_gemini():
                async with httpx.AsyncClient(timeout=60) as client:
                    resp=await client.post(gemini_url,json=payload)
                    if resp.status_code==429: raise ValueError("429")
                    resp.raise_for_status()
                    return resp.json()
            loop=asyncio.new_event_loop(); asyncio.set_event_loop(loop)
            try: result=loop.run_until_complete(_call_gemini())
            finally: loop.close()
            b64_audio=(result.get("candidates",[{}])[0].get("content",{})
                       .get("parts",[{}])[0].get("inlineData",{}).get("data"))
            if not b64_audio: raise ValueError("Gemini returned no audio data")
            with open(mp3_path,"wb") as f: f.write(base64.b64decode(b64_audio))
        else:
            voice=data.get("voice","bn-IN-TanishaaNeural")
            pitch=data.get("pitch","-5Hz"); rate=data.get("rate","+12%")
            async def _synth():
                comm=edge_tts.Communicate(text,voice,pitch=pitch,rate=rate)
                await comm.save(mp3_path)
            loop=asyncio.new_event_loop(); asyncio.set_event_loop(loop)
            try: loop.run_until_complete(_synth())
            finally: loop.close()
        if not os.path.exists(mp3_path) or os.path.getsize(mp3_path)<100:
            raise ValueError("TTS returned empty output")
        fd,wav_path=tempfile.mkstemp(suffix=".wav"); os.close(fd)
        subprocess.run(["ffmpeg","-y","-i",mp3_path,"-ar","24000","-ac","1","-sample_fmt","s16",wav_path],
                       check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        with open(wav_path,"rb") as f: wav_bytes=f.read()
        return jsonify({"pcm_b64":base64.b64encode(wav_bytes[44:]).decode(),"sample_rate":24000})
    except Exception as e:
        return jsonify({"error":str(e)}),500
    finally:
        for p in (mp3_path,wav_path):
            if p and os.path.exists(p):
                try: os.unlink(p)
                except: pass

@app.route("/dub", methods=["POST"])
def dub():
    data        = request.get_json(force=True)
    video_url   = (data.get("video_url") or "").strip()
    segments    = data.get("segments",[])
    use_ducking = data.get("ducking", False)   # Background Audio Ducking

    # â”€â”€â”€ DIAGNOSTIC DUMP at /dub entry point â”€â”€â”€
    _ok_segs   = [s for s in segments if s.get("pcm_b64")]
    _fail_segs = [s for s in segments if not s.get("pcm_b64")]
    print("[DIAG] â•â•â•â•â•â•â•â•â•â•â• /dub ENTRY DUMP â•â•â•â•â•â•â•â•â•â•â•", flush=True)
    print(f"[DIAG] total_segments      = {len(segments)}", flush=True)
    print(f"[DIAG] segments_with_pcm   = {len(_ok_segs)}", flush=True)
    print(f"[DIAG] segments_without    = {len(_fail_segs)}", flush=True)
    print(f"[DIAG] use_ducking         = {use_ducking}", flush=True)
    print(f"[DIAG] video_url           = {video_url[:80]}", flush=True)
    for _i, _s in enumerate(segments[:3]):
        _pcm = _s.get("pcm_b64", "")
        print(
            f"[DIAG] seg[{_i}] "
            f"start={_s.get('start',0):.2f} end={_s.get('end',0):.2f} "
            f"pcm_b64_exists={bool(_pcm)} pcm_b64_length={len(_pcm)}",
            flush=True
        )
    print("[DIAG] â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•", flush=True)

    if not video_url: return jsonify({"error":"video_url required"}),400
    if not segments:  return jsonify({"error":"segments required"}),400
    job_id  = f"dub_{uuid.uuid4().hex[:8]}"
    job_dir = os.path.join(TEMP_DIR,job_id)
    Path(job_dir).mkdir(parents=True,exist_ok=True)
    video_path=os.path.join(job_dir,"original.mp4"); out_path=os.path.join(job_dir,"dubbed.mp4")
    try:
        platform=get_platform(video_url)
        if platform=='kuaishou':
            video_dl_url,_,_=asyncio.run(get_ks_video_url(video_url))
            asyncio.run(download_video(video_dl_url,video_path))
        else:
            video_path=ytdlp_download(video_url,job_dir)
        probe=subprocess.run(["ffprobe","-v","quiet","-print_format","json","-show_format",video_path],
                              capture_output=True,text=True,check=True)
        duration=float(json.loads(probe.stdout)["format"]["duration"])
        base_audio=os.path.join(job_dir,"base.wav")
        subprocess.run(["ffmpeg","-y","-f","lavfi","-i",f"anullsrc=r=24000:cl=mono",
                        "-t",str(duration),base_audio],
                       check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        seg_files=[]
        for i,seg in enumerate(segments):
            start=float(seg["start"]); end=float(seg["end"])
            pcm_b64=seg.get("pcm_b64",""); sr=int(seg.get("sample_rate",24000))
            target_dur=max(0.2,end-start)
            if not pcm_b64: continue
            pcm_bytes=base64.b64decode(pcm_b64)
            num_channels=1; bits_per_sample=16; data_size=len(pcm_bytes)
            byte_rate=sr*num_channels*(bits_per_sample//8)
            block_align=num_channels*(bits_per_sample//8); chunk_size=36+data_size
            header=struct.pack("<4sI4s4sIHHIIHH4sI",b"RIFF",chunk_size,b"WAVE",b"fmt ",16,1,
                               num_channels,sr,byte_rate,block_align,bits_per_sample,b"data",data_size)
            raw_wav=os.path.join(job_dir,f"seg_{i}_raw.wav")
            with open(raw_wav,"wb") as f: f.write(header+pcm_bytes)
            probe2=subprocess.run(["ffprobe","-v","quiet","-print_format","json","-show_streams",raw_wav],
                                   capture_output=True,text=True)
            try: tts_dur=float(json.loads(probe2.stdout)["streams"][0]["duration"])
            except: tts_dur=target_dur
            ratio=max(0.5,min(2.0,tts_dur/target_dur))
            atempo=[]; r=ratio
            while r>2.0: atempo.append("atempo=2.0"); r/=2.0
            while r<0.5: atempo.append("atempo=0.5"); r/=0.5
            atempo.append(f"atempo={r:.4f}")
            fit_wav=os.path.join(job_dir,f"seg_{i}_fit.wav")
            subprocess.run(["ffmpeg","-y","-i",raw_wav,"-af",",".join(atempo),fit_wav],
                           check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            seg_files.append({"path":fit_wav,"start":start})
        if seg_files:
            filter_parts=[]; inputs=["-i",base_audio]
            for idx,sf in enumerate(seg_files):
                inputs+=["-i",sf["path"]]
                delay_ms=int(sf["start"]*1000)
                filter_parts.append(f"[{idx+1}]adelay={delay_ms}|{delay_ms}[d{idx}]")
            mixed_labels="[0]"+"".join(f"[d{i}]" for i in range(len(seg_files)))
            filter_parts.append(f"{mixed_labels}amix=inputs={len(seg_files)+1}:normalize=0[aout]")
            mixed_audio=os.path.join(job_dir,"mixed.wav")
            subprocess.run(["ffmpeg","-y"]+inputs+
                           ["-filter_complex",";".join(filter_parts),"-map","[aout]",mixed_audio],
                           check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        else:
            mixed_audio=base_audio
        if use_ducking:
            # Background Audio Ducking: original audio + dubbed audio mix
            apply_ducking(video_path, mixed_audio, out_path)
        else:
            subprocess.run(["ffmpeg","-y","-i",video_path,"-i",mixed_audio,
                            "-c:v","copy","-c:a","aac","-b:a","128k",
                            "-map","0:v:0","-map","1:a:0","-shortest",
                            "-movflags","+faststart",out_path],
                           check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        return jsonify({"video_url":f"/dub_video/{job_id}/dubbed.mp4","job_id":job_id})
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route("/dub_video/<job_id>/<filename>")
def serve_dub_video(job_id,filename):
    return send_from_directory(os.path.join(TEMP_DIR,job_id),filename)

@app.route("/health")
def health():
    return jsonify({"ok":True})

ALLOWED_CONFIG_KEYS={"VMESS_LINK","YTDLP_PROXY","COOKIES_FILE","GROQ_API_KEY","KS_COOKIES","KS_PROXY"}

@app.route("/setup/status")
def setup_status():
    cookies=os.environ.get("COOKIES_FILE",COOKIES_FILE)
    cookie_exists=os.path.exists(cookies) and os.path.getsize(cookies)>100
    return jsonify({"cookies":{"exists":cookie_exists,"size":os.path.getsize(cookies) if cookie_exists else 0},
                    "proxy":{"configured":bool(os.environ.get("YTDLP_PROXY")),"url":os.environ.get("YTDLP_PROXY","")},
                    "vmess":{"set":bool(os.environ.get("VMESS_LINK"))},"xray_bin":os.path.exists(XRAY_BIN)})

@app.route("/setup/config",methods=["GET"])
def setup_config_get():
    cfg=load_config()
    if "VMESS_LINK" in cfg: cfg["VMESS_LINK"]=cfg["VMESS_LINK"][:20]+"â€¦"
    return jsonify(cfg)

@app.route("/setup/config",methods=["POST"])
def setup_config_post():
    incoming=request.get_json(force=True) or {}
    current=load_config(); xray_changed=False
    for k,v in incoming.items():
        if k not in ALLOWED_CONFIG_KEYS: continue
        if not v: current.pop(k,None); os.environ.pop(k,None)
        else: current[k]=v.strip(); os.environ[k]=v.strip()
        if k in ("VMESS_LINK","YTDLP_PROXY"): xray_changed=True
    save_config(current)
    if xray_changed:
        link=current.get("VMESS_LINK","") or current.get("YTDLP_PROXY","")
        if link: start_xray(link)
        else: stop_xray()
    return jsonify({"ok":True})

@app.route("/setup/cookies",methods=["POST"])
def setup_cookies():
    data=request.get_json(force=True) or {}
    content=data.get("cookies","").strip()
    if not content: return jsonify({"error":"cookies required"}),400
    cookies_path=os.environ.get("COOKIES_FILE",COOKIES_FILE)
    os.makedirs(os.path.dirname(cookies_path),exist_ok=True)
    if not content.startswith("# Netscape"): content="# Netscape HTTP Cookie File\n"+content
    open(cookies_path,"w").write(content)
    return jsonify({"ok":True,"size":len(content)})

@app.route("/setup/cookies",methods=["DELETE"])
def setup_cookies_delete():
    cookies_path=os.environ.get("COOKIES_FILE",COOKIES_FILE)
    try: os.unlink(cookies_path)
    except: pass
    return jsonify({"ok":True})

@app.route("/setup/proxy-test",methods=["POST"])
def setup_proxy_test():
    proxy=os.environ.get("YTDLP_PROXY","")
    if not proxy: return jsonify({"ok":False,"error":"No proxy configured"})
    try:
        r=subprocess.run(["curl","-x",proxy,"-s","--max-time","15",
                          "-o","/dev/null","-w","%{http_code}","https://api.ipify.org"],
                         capture_output=True,text=True,timeout=20)
        ok=r.returncode==0 and r.stdout.strip().startswith("2")
        ip=None
        if ok:
            r2=subprocess.run(["curl","-x",proxy,"-s","--max-time","10","https://api.ipify.org"],
                              capture_output=True,text=True,timeout=15)
            ip=r2.stdout.strip()
        return jsonify({"ok":ok,"ip":ip,"proxy":proxy})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)})

if __name__=="__main__":
    port=int(os.environ.get("PORT",3000))
    app.run(host="0.0.0.0",port=port,debug=False,threaded=True)
