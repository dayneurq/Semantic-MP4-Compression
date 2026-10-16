"""
Quick test to check which TTS methods work with your API key
"""
import requests
import os

# Replace with your actual API key
API_KEY = ""  # <-- UPDATE THIS

def test_basic_tts():
    """Test basic tts-1 model"""
    print("\n1. Testing tts-1 (basic Speech API)...")
    response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": "tts-1",
            "input": "Hello, this is a test.",
            "voice": "alloy"
        }
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print("   ✅ tts-1 works!")
        return True
    else:
        print(f"   ❌ Error: {response.text}")
        return False

def test_gpt4o_mini_audio():
    """Test gpt-4o-mini with audio modality"""
    print("\n2. Testing gpt-4o-mini with audio output...")
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4o-mini",
            "modalities": ["text", "audio"],
            "audio": {"voice": "alloy", "format": "wav"},
            "messages": [
                {"role": "user", "content": "Say hello"}
            ]
        }
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        has_audio = "choices" in result and len(result["choices"]) > 0 and "audio" in result["choices"][0].get("message", {})
        if has_audio:
            print("   ✅ gpt-4o-mini audio works!")
            return True
        else:
            print("   ❌ Response has no audio data")
            print(f"   Response: {result}")
            return False
    else:
        print(f"   ❌ Error: {response.text}")
        return False

def test_gpt4o_audio_preview():
    """Test gpt-4o-audio-preview"""
    print("\n3. Testing gpt-4o-audio-preview...")
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4o-audio-preview",
            "modalities": ["text", "audio"],
            "audio": {"voice": "alloy", "format": "wav"},
            "messages": [
                {"role": "user", "content": "Say hello"}
            ]
        }
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        has_audio = "choices" in result and len(result["choices"]) > 0 and "audio" in result["choices"][0].get("message", {})
        if has_audio:
            print("   ✅ gpt-4o-audio-preview works!")
            return True
        else:
            print("   ❌ Response has no audio data")
            return False
    else:
        print(f"   ❌ Error: {response.text}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("OpenAI TTS API Compatibility Test")
    print("="*60)
    
    if API_KEY == "your-key-here":
        print("\n⚠️  Please update API_KEY in the script first!")
        exit(1)
    
    test_basic_tts()
    test_gpt4o_mini_audio()
    test_gpt4o_audio_preview()
    
    print("\n" + "="*60)
    print("Recommendation:")
    print("- If only tts-1 works: Set use_delivery_control=False")
    print("- If gpt-4o-audio-preview works: Use that (more expensive)")
    print("- If gpt-4o-mini audio works: Current config is fine")
    print("="*60)