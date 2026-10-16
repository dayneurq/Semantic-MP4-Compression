from audio_compressor import regenerate_audio, config

config.openai_api_key = ""
config.fal_api_key = ""

regenerate_audio("compressed.json", "output.wav")