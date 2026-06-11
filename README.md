# 🎬 KwaiDub AI —  Video Dubbing Toolkit

**Download → Transcribe → Translate → Dub (Single-Person Voice, Emotion Mimic)**  
🔥 *Paste any video link*(YouTube,Facebook,titok,Instagram,kuaishou . Get a dubbed version in Bengali/English(any language) with Gemini api and Edge tts fallback.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

<p align="center">
  <img src="https://github.com/user-attachments/assets/a0bc7d2d-7d84-4561-b131-59aa9fd2928a" alt="Main Interface" width="280"/>
  <img src="https://github.com/user-attachments/assets/4c04862e-62d0-439c-aaad-4495479d1648" alt="Segment View" width="280"/>
  <img src="https://github.com/user-attachments/assets/71dc9af7-c6ad-414f-a911-4ff1695a465c" alt="Setup Tab" width="280"/>
</p>

---

## 🤯 What Is This?

You find a **Chinese/TikTok/YouTube video** – but you don't understand the language.  
You want to **dub it in Bengali/English/Hindi** — and make the voice sound **natural, emotional**, not robotic.

**KwaiDub AI does exactly that — in one click.**

- ✅ **YouTube, TikTok, Instagram, Kuaishou** – just paste the URL
- ✅ **Word‑level subtitles** via Groq Whisper Large v3
- ✅ **Auto‑translate** Chinese → Bengali/English (Groq LLaMA 3.3)
- ✅ **Three TTS engines** – including **Emotion Mimic** that copies the original speaker's tone
- ✅ **Free TTS** – Edge‑TTS needs zero API keys
- ✅ **YouTube bot bypass** – built‑in Xray/V2Ray local SOCKS5 proxy

📹 **See it in action:**  
[![Demo Video](https://img.shields.io/badge/Watch_Demo-Video-red?style=for-the-badge)](https://github.com/user-attachments/assets/4d56b61d-6d93-49f3-a07f-1fdf60e6db89)

---

## 🔥 THE KILLER FEATURE: Emotion Mimic

<p align="center">
  <img src="https://github.com/user-attachments/assets/1968a3ed-ca86-47ca-9360-cba571018d93" alt="Dubbing Progress with Emotion Mimic" width="300"/>
</p>

**Normal AI dubs** sound flat and robotic.  
**KwaiDub's Emotion Mimic** sends the **original audio clip** to Gemini and says:  
*"Copy the emotion, pace, and energy of this voice. Now speak THIS new text."*

> A Chinese video with **angry shouting** → the Bengali dub also sounds **angry and intense**.  
> A **slow, emotional speech** → the dub preserves that **exact mood**.

It's not just TTS. It's **emotional voice cloning for video dubbing**.

---

## 🚀 Quick Start (Docker only)

```bash
docker pull your-dockerhub/kwaidub-ai:latest
docker run -d -p 3000:3000 --name kwaidub your-dockerhub/kwaidub-ai:latest
```

Then open http://localhost:3000.
No Python, no pip, no requirements.txt – the whole app is inside the container.

---

🔑 How to Set Up API Keys

📝 Groq API Key (for Transcription & Translation)

1. Sign up at console.groq.com
2. Create an API key (gsk_...)
3. Paste it in the "Groq API Keys" box (multiple keys allowed, separated by newline)

⚠️ Important: For translation, always use Groq (LLaMA 3.3 70B) – NOT Gemini.
Gemini's free tier often blocks translation requests from Bangladesh. Groq works perfectly for Chinese→Bengali/English.

🎙️ Gemini API Keys (for TTS – Optional if you use Edge‑TTS)

To avoid rate limits, you'll need 13‑14 different Google accounts and one API key from each.

1. Create 13‑14 Gmail accounts (yes, it's worth it).
2. For each account, visit Google AI Studio
3. Click "Get API key" → "Create API key"
4. Copy every key (starts with AIza...)
5. Paste all keys into the "Gemini API Keys" box – one key per line.

Why 13‑14?
Gemini TTS free tier gives only ~15 requests per minute per key.
A 2‑minute video may have 30‑40 segments.
With 13 keys rotating, you get ~195 requests/minute – smooth dubbing without 429 errors.

---

🛡️ YouTube Bot Bypass (V2Ray / Xray)

YouTube may block automated downloads from datacenter IPs.
KwaiDub uses Xray to create a local SOCKS5 proxy through any vmess/vless node.

Get a free V2Ray node:

1. Go to www.v2nodes.com
2. Copy any vmess:// or vless:// link
3. In the app: Setup tab → paste the link → click "Save & Start"
4. Hit "Test Proxy" to verify

The proxy runs inside the container and only routes yt‑dlp traffic through it.

deploy it in Railway.com server.it provide free $5 Token,which is enough for this tools. creat a account with tempmail then install Railway CLI in your terminal(termux). then login bia railway login --browserless and deploy it..every month do it one time. Thats enough.  
---

🎨 Features

Feature Status Details
🌐 Multi‑platform ✅ YouTube, TikTok, Instagram, Kuaishou
🎯 Word‑level Transcript ✅ Groq Whisper Large v3
🌍 Auto Translate ✅ Chinese → Bengali/English (Groq LLaMA)
🎙️ TTS Providers ✅ Gemini Normal / Emotion Mimic / Edge‑TTS (FREE)
🎭 Emotion Mimic ✅ Copies original speaker's emotion & tone
🔑 Key Rotation ✅ Auto‑rotates 13+ Gemini keys
🛡️ YouTube Bypass ✅ Built‑in Xray/V2Ray local SOCKS5 proxy
🍪 YouTube Cookies ✅ For age‑restricted or login‑required videos
🎛️ Pitch & Speed ✅ Adjust Edge‑TTS voice (−5% pitch, +12% speed)
📥 One‑click Download ✅ Final dubbed video in mp4
🐳 Docker Ready ✅ No manual dependency installs
👤 Single‑Person Voice ✅ Current version (multi‑person coming soon)

---

🎭 TTS Engine Comparison

 Gemini Normal Gemini Emotion Mimic 🔥 Edge‑TTS (FREE)
Cost Free tier (15 RPM) Free tier (15 RPM) Completely FREE
Voice Quality Good Excellent + Emotional Very Good
Emotion Copy ❌ ✅ Copies original! ❌
Languages English, Hindi… English, Hindi… Bengali, Chinese, 50+
API Key Required (13+ for smooth) Required (13+ for smooth) NOT needed!

---

📸 Screenshots

<p align="center">
  <img src="https://github.com/user-attachments/assets/a0bc7d2d-7d84-4561-b131-59aa9fd2928a" alt="Main UI" width="250"/>
  <img src="https://github.com/user-attachments/assets/4c04862e-62d0-439c-aaad-4495479d1648" alt="Transcription Result" width="250"/>
  <img src="https://github.com/user-attachments/assets/71dc9af7-c6ad-414f-a911-4ff1695a465c" alt="Setup Tab" width="250"/>
  <img src="https://github.com/user-attachments/assets/196eabf4-38d2-4090-89a0-497a3118548f" alt="Dubbing Preview" width="250"/>
  <img src="https://github.com/user-attachments/assets/1968a3ed-ca86-47ca-9360-cba571018d93" alt="Emotion Mimic Progress" width="250"/>
</p>

---

🧠 How Emotion Mimic Works (Technical)

1. Transcribe the video → the original audio is saved as {job_id}.mp3
2. For each subtitle segment:
   · Cut the corresponding audio chunk (ffmpeg -ss start -to end)
   · Convert the chunk to base64
   · Send it to Gemini Multimodal with a prompt:
     "Listen to this audio. Copy the emotion, tone, and pace. Now speak THIS text in Bengali."
   · Gemini returns a new audio file with the same emotional characteristics
3. Replace the video's original audio track with the new segments (ffmpeg amix)

---

🛠️ Tech Stack

· Backend: Python 3.12, Flask, asyncio
· Container: Docker (all dependencies included)
· Transcription: Groq API (Whisper Large v3)
· Translation: Groq LLaMA 3.3 70B (better than Gemini for Bangladeshi users)
· TTS: Gemini TTS, Edge‑TTS (Microsoft)
· Audio Processing: FFmpeg, ffprobe
· Video Download: yt‑dlp, httpx
· Proxy: Xray core (vmess/vless → local SOCKS5)

---

🚧 Roadmap

· Single‑person voice dubbing
· Emotion Mimic (Gemini multimodal)
· YouTube/TikTok/Kuaishou support
· V2Ray/Xray local proxy
· Multi‑person voice separation (coming soon!)
· Speaker diarization
· More TTS engines (Azure, ElevenLabs)
· Web UI dark/light themes

---

🌟 Why u should use it.!?

1. Free & Open Source – Edge‑TTS needs zero keys
2. Emotion Mimic – first open‑source tool with emotional dubbing
3. Any platform – YouTube, TikTok, Instagram, Kuaishou
4. Any language pair – Chinese → Bengali/English/Hindi
5. One click – paste URL, get dubbed video
6. YouTube bypass built‑in – no more geo‑blocks
7. Docker only – runs anywhere

---

⚠️ Disclaimer

This tool is intended for educational and personal use only.
Please respect copyright laws. Do not distribute dubbed videos without proper rights.

---

⭐ Star This Repo!

If this project helped you dub even one video, give it a star – it fuels the development of multi‑person voice separation!

<p align="center">
  <b>Made with ❤️ by a developer who just wanted to watch Chinese videos in Bengali</b><br>
  <sub>Single‑person now, multi‑person soon 🚀</sub>
</p>
