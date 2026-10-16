import os
from moviepy import VideoFileClip

def extract_mp3_from_mp4(input_video, output_audio=None):
    # If no output name is given, just swap the extension
    if not output_audio:
        output_audio = os.path.splitext(input_video)[0] + ".mp3"

    print(f"🎬 Loading video: {input_video}")
    try:
        # Load the video clip
        video = VideoFileClip(input_video)
        
        # Extract the audio track
        audio = video.audio
        
        print(f"🎵 Extracting audio to: {output_audio}...")
        # Write the audio file at high quality
        # 320k bitrate is best for AI analysis
        audio.write_audiofile(output_audio, bitrate="320k")
        
        # Close clips to free up system memory
        audio.close()
        video.close()
        print("✅ Extraction Complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Change 'input.mp4' to your actual file name
    extract_mp3_from_mp4("bluevelvet.mp4")