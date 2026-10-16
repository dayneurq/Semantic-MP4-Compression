import os, requests, io, base64, json, sys, re, time
from openai import OpenAI
import fal_client
from pydub import AudioSegment

# 🔑 Credentials
os.environ["FAL_KEY"] = ""
OPENAI_KEY = ""
client = OpenAI(api_key=OPENAI_KEY)

def encode_audio(segment):
    buf = io.BytesIO()
    segment.export(buf, format="wav")
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def clean_ai_response(text):
    """Strips all conversational filler for Gen-AI reconstruction."""
    if not text: return ""
    # Fixed Regex for Python 3.11+ compliance
    unwanted = [
        r"(?i)^sure.*?:", r"(?i)^here is.*?:", r"(?i)^the description is:", 
        r"(?i)^based on the audio,", r"(?i)^I can hear", r"(?i)^this clip features",
        r"(?i)^prompt:", r"\""
    ]
    for pattern in unwanted:
        text = re.sub(pattern, "", text).strip()
    return text.strip()

def call_openai_with_retry(messages, model="gpt-4o-audio-preview"):
    """Handles RateLimitError 429 by waiting and retrying."""
    while True:
        try:
            res = client.chat.completions.create(model=model, messages=messages)
            return res.choices[0].message.content
        except Exception as e:
            if "429" in str(e):
                print("⏳ Rate limit reached. Waiting 60 seconds...")
                time.sleep(60)
            else:
                raise e

def run_production_blueprint(input_file):
    vox_p, mus_p, sfx_p = "dialog_stem.wav", "music_stem.wav", "sfx_stem.wav"
    transcript_p = "transcript_cache.json"
    blueprint_p = "master_blueprint_v11.txt"

    # --- 1. STEM SEPARATION (Uses existing if found) ---
    if not os.path.exists(vox_p):
        print("📤 Running Demucs High-Isolation Stem Separation...")
        file_url = fal_client.upload_file(input_file)
        job = fal_client.subscribe("fal-ai/demucs", {"audio_url": file_url})
        with open(vox_p, "wb") as f: f.write(requests.get(job["vocals"]["url"]).content)
        with open(mus_p, "wb") as f: f.write(requests.get(job["bass"]["url"]).content) 
        with open(sfx_p, "wb") as f: f.write(requests.get(job["other"]["url"]).content)
        print("✅ Stems separated.")

    v_stem, m_stem, s_stem = AudioSegment.from_file(vox_p), AudioSegment.from_file(mus_p), AudioSegment.from_file(sfx_p)

    # --- 2. DIARIZATION ---
    if os.path.exists(transcript_p):
        with open(transcript_p, "r") as f: segments = json.load(f)
    else:
        print("🎙️ Running Diarization...")
        with open(input_file, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="gpt-4o-transcribe-diarize", 
                file=f, 
                response_format="diarized_json",
                chunking_strategy="auto" 
            )
            segments = [{"start": s.start, "end": s.end, "text": s.text, "speaker": getattr(s, 'speaker', '0')} for s in resp.segments]
        with open(transcript_p, "w") as f: json.dump(segments, f)

    # --- 3. SECTION 1: SPEAKER PROFILES ---
    sec1 = ["=== SECTION 1: SPEAKER GEN-AI PROFILES ===\n"]
    unique_ids = sorted(list(set(seg['speaker'] for seg in segments)))
    for sid in unique_ids:
        s_list = [s for s in segments if s['speaker'] == sid]
        sample = max(s_list, key=lambda x: (x['end'] - x['start']))
        clip = v_stem[int(sample['start']*1000):int(sample['end']*1000)]
        content = call_openai_with_retry([
            {"role": "system", "content": "Output ONLY a comma-separated list of Gen-AI voice tags: Age, Gender, Texture, Pitch, Resonance."},
            {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": encode_audio(clip), "format": "wav"}}]}
        ])
        sec1.append(f"SPEAKER {sid}: {clean_ai_response(content)}\n")

    # --- 4. SECTION 2: DIALOG RECONSTRUCTION ---
    sec2 = ["\n=== SECTION 2: DIALOG RECONSTRUCTION ===\n"]
    for seg in segments:
        t1, t2 = int(seg['start']*1000), int(seg['end']*1000)
        content = call_openai_with_retry([
            {"role": "system", "content": "Output ONLY technical descriptions for a TTS engine: emotional intensity, speed, and prosody."},
            {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": encode_audio(v_stem[t1:t2]), "format": "wav"}}]}
        ])
        sec2.append(f"[{seg['start']}-{seg['end']}] | SPK:{seg['speaker']} | MANNER: {clean_ai_response(content)} | TEXT: {seg['text']}\n")

    # --- 5. SECTION 3: MUSIC ---
    sec3 = ["\n=== SECTION 3: MUSIC (Gen-AI Prompting) ===\n"]
    ptr, step = 0, 15000 # Increased step to 15s to save API requests
    while ptr < len(m_stem):
        clip = m_stem[ptr:ptr+step]
        if clip.dBFS > -42:
            content = call_openai_with_retry([
                {"role": "system", "content": "Output ONLY a technical prompt for a Music Generation AI. Include instruments, tempo, and mood. If no music, say 'SILENCE'."},
                {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": encode_audio(clip), "format": "wav"}}]}
            ])
            desc = clean_ai_response(content)
            if "SILENCE" not in desc.upper():
                sec3.append(f"[{ptr/1000}-{(ptr+len(clip))/1000}] | PROMPT: {desc}\n")
        ptr += step

    # --- 6. SECTION 4: SFX ---
    sec4 = ["\n=== SECTION 4: SFX (Gen-AI Prompting) ===\n"]
    ptr, step = 0, 10000 # Increased step to 10s to save API requests
    while ptr < len(s_stem):
        clip = s_stem[ptr:ptr+step]
        if clip.dBFS > -48:
            content = call_openai_with_retry([
                {"role": "system", "content": "Output ONLY a prompt for an SFX Generation AI. List objects and environment. If no sfx, say 'SILENCE'."},
                {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": encode_audio(clip), "format": "wav"}}]}
            ])
            desc = clean_ai_response(content)
            if "SILENCE" not in desc.upper():
                sec4.append(f"[{ptr/1000}-{(ptr+len(clip))/1000}] | PROMPT: {desc}\n")
        ptr += step

    with open(blueprint_p, "w", encoding="utf-8") as f:
        f.writelines(sec1 + sec2 + sec3 + sec4)
    print(f"✅ Full Production Blueprint Saved: {blueprint_p}")

if __name__ == "__main__":
    run_production_blueprint("bluevelvet.mp3")