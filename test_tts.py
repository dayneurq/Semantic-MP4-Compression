from audio_compressor import config
import requests

config.openai_api_key = ""


url = "https://api.openai.com/v1/audio/speech"
headers = {
    "Authorization": f"Bearer {config.openai_api_key}",
    "Content-Type": "application/json"
}

payload = {
    "model": "tts-1",
    "input": "Hello world, this is a test of the text to speech system.",
    "voice": "onyx"
}

print("Sending request...")
response = requests.post(url, headers=headers, json=payload)

print(f"Status: {response.status_code}")
print(f"Content-Type: {response.headers.get('Content-Type')}")
print(f"Transfer-Encoding: {response.headers.get('Transfer-Encoding')}")
print(f"Response content length: {len(response.content)} bytes")

if response.status_code != 200:
    print(f"Error: {response.text}")
else:
    # Read the full response
    full_content = b""
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            full_content += chunk
    
    print(f"Total bytes after reading chunks: {len(full_content)}")
    
    with open("test_speech.mp3", "wb") as f:
        f.write(full_content)
    print("Saved to test_speech.mp3")