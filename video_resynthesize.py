import os, requests, fal_client, base64
from moviepy import VideoFileClip, concatenate_videoclips

os.environ["FAL_KEY"] = ""

def run_chain_resynthesis(blueprint_file):
    with open(blueprint_file, "r", encoding="utf-8", errors="ignore") as f:
    	lines = [l for l in f.readlines() if l.startswith("SHOT_")]

    final_clips = []
    last_clip_filename = None

    for line in lines:
        parts = line.split(" | ")
        shot_id, seg_idx, target_dur, prompt = parts[0], int(parts[1]), float(parts[2]), parts[3]
        
        # 1. GENERATE
        if seg_idx == 0:
            res = fal_client.subscribe("fal-ai/wan-t2v", arguments={
                "prompt": prompt, "resolution": "480p", "turbo_mode": True
            })
        else:
            # Get the bridge frame from the previous segment
            prev_clip = VideoFileClip(last_clip_filename)
            bridge_path = "bridge.jpg"
            prev_clip.save_frame(bridge_path, t=prev_clip.duration - 0.05)
            with open(bridge_path, "rb") as b:
                img_b64 = base64.b64encode(b.read()).decode('utf-8')
            
            res = fal_client.subscribe("fal-ai/wan-i2v", arguments={
                "prompt": prompt, "image_url": f"data:image/jpeg;base64,{img_b64}", "resolution": "480p"
            })

        # 2. DOWNLOAD & PRECISION CUT
        video_url = res['video']['url']
        fname = f"raw_{shot_id}_{seg_idx}.mp4"
        with open(fname, "wb") as f: f.write(requests.get(video_url).content)
        
        # Use the "Flag" to cut the 5s AI clip down to the original duration
        raw_clip = VideoFileClip(fname)
        if raw_clip.duration > target_dur:
            print(f"✂️ Clipping {shot_id} from 5s down to {target_dur}s")
            final_clip = raw_clip.subclipped(0, target_dur)
        else:
            final_clip = raw_clip
            
        final_clips.append(final_clip)
        last_clip_filename = fname

    # 3. STITCH
    final = concatenate_videoclips(final_clips, method="compose")
    final.write_videofile("remastered_10min.mp4")

run_chain_resynthesis("video_blueprint.txt")