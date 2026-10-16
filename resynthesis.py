import os, requests, io, fal_client
from pydub import AudioSegment

os.environ["FAL_KEY"] = ""

def fetch_seg(url):
    r = requests.get(url)
    return AudioSegment.from_file(io.BytesIO(r.content)) if r.status_code == 200 else None

def run_precision_resynthesis(blueprint_file, output="resynthesized_master.mp3"):
    with open(blueprint_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    final_mix = AudioSegment.silent(duration=0)
    voice_ids = {} # Key: s_id, Value: custom_voice_id

    for line in lines:
        # Parse: 0:start, 1:end, 2:s_id, 3:v_desc, 4:text, 5:manner, 6:env_desc, 7:v_db, 8:m_db, 9:s_db
        p = line.strip().split("|")
        start, end, s_id, v_desc, text, manner, env_desc = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
        v_db, m_db, s_db = float(p[7]), float(p[8]), float(p[9])
        dur_ms = int((float(end) - float(start)) * 1000)

        print(f"🔨 Synthesizing {start}s - {end}s...")

        # 1. Voice Design (Speaker Identity)
        if s_id not in voice_ids:
            design = fal_client.subscribe("fal-ai/minimax/voice-design", arguments={"prompt": v_desc})
            voice_ids[s_id] = design["custom_voice_id"]

        # 2. Dialogue Generation (Performance Manner)
        vox_job = fal_client.subscribe("fal-ai/minimax/text-to-speech", arguments={
            "text": text,
            "voice_id": voice_ids[s_id],
            "prompt": f"Deliver this line in a {manner} manner."
        })

        # 3. Environment Generation (Music & SFX)
        env_job = fal_client.subscribe("fal-ai/stable-audio-25/text-to-audio", arguments={
            "prompt": env_desc,
            "seconds_total": float(end) - float(start)
        })

        # 4. Download and Match Stems
        v_seg = fetch_seg(vox_job["audio"]["url"])
        e_seg = fetch_seg(env_job["audio"]["url"])

        if v_seg and e_seg:
            # Match Volume Dynamics
            v_seg = v_seg.apply_gain(v_db - v_seg.dBFS)
            e_seg = e_seg.apply_gain(((m_db + s_db) / 2) - e_seg.dBFS) # Average background volume

            # Precise Trimming to prevent drift
            e_seg = e_seg[:dur_ms]
            
            # Layer and Stitch
            combined = e_seg.overlay(v_seg)
            final_mix += combined

    print("💾 Exporting...")
    final_mix.export(output, format="mp3", bitrate="320k")
    print(f"🏁 DONE: {output}")

if __name__ == "__main__":
    run_precision_resynthesis("master_blueprint.txt")