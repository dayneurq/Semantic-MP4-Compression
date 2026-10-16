from audio_compressor import analyze_audio, regenerate_audio, config

# Set your API keys first!
config.openai_api_key = ""
config.fal_api_key = ""
config.huggingface_token = ""

# Step 1: Compress
print("Step 1: Analyzing and compressing audio...")
analyze_audio("bluevelvet.mp3", "compressed.json")
print("✓ Compression complete! Check compressed.json")
