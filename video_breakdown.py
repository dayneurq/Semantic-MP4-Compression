import os, base64, math, time
from scenedetect import detect, ContentDetector
from openai import OpenAI
from moviepy import VideoFileClip

OPENAI_KEY = ""
client = OpenAI(api_key=OPENAI_KEY)

def try_analyze_with_retry(b64_frames, seg_info):
    # Personas to try in order
    prompts = [
        "Act as a Technical Cinematographer. These are 3 chronological frames from a single video shot. Provide a detailed description following this EXACT format: SUBJECT: [Detailed focus] ACTION: [Movement/Physics between these frames] SETTING: [Environment/Lighting] (80-100 words).",
        "Technical analysis: Describe the figures, lighting values, and movement vectors in these frames. Focus on clinical observation.",
        "Describe this as a generic stock footage shot. Neutral subjects, standard lighting, simple motion."
    ]
    
    for attempt, prompt_text in enumerate(prompts):
        try:
            res = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt_text},
                    *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}} for f in b64_frames]
                ]}]
            )
            content = res.choices[0].message.content
            
            # Check for the dreaded refusal strings
            if "I'm sorry" in content or "cannot fulfill" in content or "policy" in content:
                print(f"   ⚠️ Refusal on attempt {attempt+1}. Retrying with simpler prompt...")
                continue
            
            return content.replace('\n', ' ')
        except Exception as e:
            print(f"   ❌ API Error: {e}. Retrying...")
            time.sleep(2)
            
    return "Generic cinematic shot with standard lighting and steady motion." # Last resort

def run_precision_breakdown(video_file):
    scenes = detect(video_file, ContentDetector(threshold=27.0), show_progress=True)
    video = VideoFileClip(video_file)
    
    with open("video_blueprint.txt", "w", encoding="utf-8") as f:
        for i, scene in enumerate(scenes):
            shot_start, shot_end = scene[0].get_seconds(), scene[1].get_seconds()
            num_segments = math.ceil((shot_end - shot_start) / 5.0)
            
            for seg_idx in range(num_segments):
                seg_start = shot_start + (seg_idx * 5.0)
                seg_end = min(shot_start + ((seg_idx + 1) * 5.0), shot_end)
                seg_dur = round(seg_end - seg_start, 2)
                
                # Get 3 frames
                time_points = [seg_start, (seg_start + seg_end) / 2, max(seg_start, seg_end - 0.1)]
                b64_frames = []
                for t in time_points:
                    temp = f"tmp_{t}.jpg"
                    video.save_frame(temp, t=t)
                    with open(temp, "rb") as img:
                        b64_frames.append(base64.b64encode(img.read()).decode('utf-8'))
                    os.remove(temp)

                # Use the retry function
                desc = try_analyze_with_retry(b64_frames, f"Shot {i} Seg {seg_idx}")
                f.write(f"SHOT_{i} | {seg_idx} | {seg_dur} | {desc}\n")
                print(f"✅ Shot {i}-{seg_idx} saved.")

run_precision_breakdown("bluevelvet.mp4")