"""
Semantic Audio Compression & Regeneration System
Analyzes audio into structured descriptions and regenerates from those descriptions
v3.1 - NOW WITH REAL DELIVERY CONTROL!
- Uses gpt-4o-mini-tts with 'instructions' parameter for true delivery control
- Passes detailed delivery instructions (tone, pacing, emotion) to TTS
- Much more affordable than gpt-4o-audio-preview
- Actually works (unlike Chat Completions audio which ignores instructions)
"""

import json
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Any
import requests
from dataclasses import dataclass
import subprocess
import os
import tempfile
import io

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration for the audio compression system"""
    openai_api_key: str
    fal_api_key: str
    huggingface_token: str = ""  # Optional - for speaker diarization
    sample_rate: int = 44100
    whisper_model: str = "whisper-1"
    tts_model: str = "gpt-4o-mini-tts"  # Options: "tts-1", "tts-1-hd", "gpt-4o-mini-tts" (with delivery control!)
    use_delivery_control: bool = True  # Enable to use delivery instructions with gpt-4o-mini-tts
    music_model: str = "fal-ai/stable-audio"
    use_diarization: bool = True  # Enable proper speaker diarization
    use_singing_generation: bool = False  # Use music AI for singing (experimental)
    singing_duration_threshold: float = 5.0  # Segments longer than this are considered singing
    
config = Config(
    openai_api_key="your-openai-key-here",
    fal_api_key="your-fal-key-here",
    huggingface_token="your-huggingface-token-here"  # Get from hf.co/settings/tokens
)

# ============================================================================
# PHASE 1: AUDIO ANALYSIS
# ============================================================================

class AudioAnalyzer:
    """Analyzes audio and creates semantic descriptions"""
    
    def __init__(self, audio_path: str, config: Config):
        self.audio_path = audio_path
        self.config = config
        print(f"Loading audio file: {audio_path}")
        self.audio, self.sr = librosa.load(audio_path, sr=config.sample_rate, mono=False)
        if self.audio.ndim > 1:
            self.audio = librosa.to_mono(self.audio)
        self.duration = librosa.get_duration(y=self.audio, sr=self.sr)
        self.speaker_profiles = {}
        print(f"Loaded {self.duration:.2f} seconds of audio at {self.sr} Hz")
        
    def separate_stems_advanced(self, output_dir: str = "separated") -> Dict[str, str]:
        """Separate audio into dialogue, music, and SFX stems using Demucs"""
        audio_name = Path(self.audio_path).stem
        stem_dir = Path(output_dir) / "htdemucs" / audio_name
        vocals_path = stem_dir / "vocals.wav"
        no_vocals_path = stem_dir / "no_vocals.wav"
        music_path = stem_dir / "music.wav"
        sfx_path = stem_dir / "sfx.wav"
        
        # If all stems exist, skip Demucs
        if vocals_path.exists() and music_path.exists() and sfx_path.exists():
            print(f"✓ Found existing stems in {stem_dir}")
            print("  Skipping Demucs separation (delete 'separated' folder to re-run)")
            return {
                "dialogue": str(vocals_path),
                "music": str(music_path),
                "sfx": str(sfx_path)
            }
        
        Path(output_dir).mkdir(exist_ok=True)
        
        print("Separating audio into stems with Demucs...")
        print(f"This will take 5-10 minutes for a 10-minute file...")
        
        cmd = ["demucs", "--two-stems=vocals", "-o", output_dir, self.audio_path]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✓ Demucs completed")
            
            if not vocals_path.exists() or not no_vocals_path.exists():
                print(f"ERROR: Stem files not found!")
                return {"full": self.audio_path}
            
            print(f"✓ Vocals separated: {vocals_path}")
            print(f"✓ Instrumental separated: {no_vocals_path}")
            
            print("Separating music and sound effects...")
            no_vocals_audio, sr = librosa.load(str(no_vocals_path), sr=self.config.sample_rate)
            harmonic, percussive = librosa.effects.hpss(no_vocals_audio, margin=3.0)
            
            sf.write(str(music_path), harmonic, sr)
            sf.write(str(sfx_path), percussive, sr)
            
            print(f"✓ Music separated: {music_path}")
            print(f"✓ SFX separated: {sfx_path}")
            print(f"\n✓ Successfully separated into 3 stems!")
            
            return {
                "dialogue": str(vocals_path),
                "music": str(music_path),
                "sfx": str(sfx_path)
            }
        except Exception as e:
            print(f"Stem separation error: {e}")
            return {"full": self.audio_path}
    
    def diarize_speakers(self, vocals_path: str) -> Dict:
        """Use pyannote.audio for accurate speaker diarization"""
        print("Running speaker diarization with pyannote.audio...")
        
        try:
            from pyannote.audio import Pipeline
            import torch
            
            # Initialize pipeline (requires HuggingFace token)
            if not self.config.huggingface_token:
                print("WARNING: No HuggingFace token provided, skipping diarization")
                return {}
            
            # Try with updated parameter name
            try:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=self.config.huggingface_token
                )
            except TypeError:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.config.huggingface_token
                )
            
            print("  This may take a few minutes...")
            
            # Load audio with librosa to avoid torchcodec issues
            audio_array, sr = librosa.load(vocals_path, sr=16000, mono=True)
            
            # Convert to pyannote format: {'waveform': tensor, 'sample_rate': int}
            waveform = torch.from_numpy(audio_array[np.newaxis, :]).float()
            audio_dict = {
                "waveform": waveform,
                "sample_rate": sr
            }
            
            # Run diarization on the preloaded audio
            diarization = pipeline(audio_dict)
            
            # Convert to speaker segments
            speaker_segments = {}
            
            # New pyannote API - use speaker_diarization property
            speaker_timeline = diarization.speaker_diarization
            
            # Iterate through the timeline
            for segment, _, speaker in speaker_timeline.itertracks(yield_label=True):
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                speaker_segments[speaker].append({
                    "start": float(segment.start),
                    "end": float(segment.end)
                })
            
            print(f"✓ Detected {len(speaker_segments)} speakers")
            return speaker_segments
            
        except ImportError:
            print("WARNING: pyannote.audio not installed. Install with: pip install pyannote.audio")
            return {}
        except Exception as e:
            print(f"Diarization error: {e}")
            return {}
    
    def analyze_voice_characteristics(self, audio_segment: np.ndarray, sr: int) -> Dict:
        """Analyze voice characteristics"""
        if len(audio_segment) < sr * 0.1:
            return {"pitch": "medium-pitched", "tone": "neutral", "pace": "normal", "avg_pitch_hz": 150.0}
        
        pitches, magnitudes = librosa.piptrack(y=audio_segment, sr=sr)
        pitch_values = [pitches[magnitudes[:, t].argmax(), t] for t in range(pitches.shape[1]) if pitches[magnitudes[:, t].argmax(), t] > 0]
        avg_pitch = np.mean(pitch_values) if pitch_values else 150
        
        if avg_pitch < 120:
            pitch_desc = "deep, low-pitched"
        elif avg_pitch < 180:
            pitch_desc = "medium-pitched"
        elif avg_pitch < 250:
            pitch_desc = "higher-pitched"
        else:
            pitch_desc = "high, bright"
        
        onset_frames = librosa.onset.onset_detect(y=audio_segment, sr=sr)
        pace = len(onset_frames) / (len(audio_segment) / sr)
        pace_desc = "slow, deliberate" if pace < 3 else "moderate pace" if pace < 6 else "fast, energetic"
        
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sr).mean()
        tone_desc = "warm, smooth" if spectral_centroid < 1500 else "balanced, clear" if spectral_centroid < 2500 else "bright, crisp"
        
        return {
            "pitch": pitch_desc,
            "pace": pace_desc,
            "tone": tone_desc,
            "avg_pitch_hz": float(avg_pitch)
        }
    
    def analyze_segment_with_api(self, audio_segment: np.ndarray, sr: int) -> Dict:
        """
        Upload audio segment to OpenAI API for detailed transcription, 
        delivery analysis, and voice gender detection
        """
        try:
            # Save segment to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(temp_path, audio_segment, sr)
            
            # Transcribe with Whisper API
            print(f"    Transcribing segment...")
            with open(temp_path, 'rb') as audio_file:
                transcript_response = requests.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.config.openai_api_key}"},
                    files={"file": audio_file},
                    data={"model": "whisper-1"}
                )
            
            if transcript_response.status_code != 200:
                print(f"    Transcription failed: {transcript_response.text}")
                os.unlink(temp_path)
                return {
                    "text": "[transcription failed]",
                    "delivery": "neutral conversational tone",
                    "voice_gender": "neutral"
                }
            
            transcription = transcript_response.json().get("text", "")
            
            # Now use GPT-4 with the audio for detailed analysis
            print(f"    Analyzing delivery style...")
            
            # Convert audio to base64 for GPT-4o Audio
            with open(temp_path, 'rb') as f:
                audio_data = f.read()
            
            import base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Use GPT-4 to analyze the audio and text
            analysis_response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-audio-preview",
                    "modalities": ["text"],
                    "audio": {"voice": "alloy", "format": "wav"},
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": audio_base64,
                                        "format": "wav"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": f"""Analyze this audio clip and provide:

1. **Voice Gender**: Is this a male voice, female voice, or ambiguous/neutral? Just answer with one word: "male", "female", or "neutral"

2. **Detailed Delivery Description**: Describe HOW this line is being delivered in rich detail (3-5 sentences). Include:
   - Emotional tone and intensity (e.g., angry, sad, excited, calm, panicked, sarcastic, warm, cold)
   - Volume and vocal intensity (e.g., shouting, whispering, normal speaking volume, strained)
   - Pacing and rhythm (e.g., rushed, slow and deliberate, hesitant, confident)
   - Vocal quality and characteristics (e.g., breathy, clear, raspy, smooth, shaky, firm)
   - Any notable emphasis, inflection, or vocal mannerisms

The transcription is: "{transcription}"

Format your response EXACTLY as:
GENDER: [male/female/neutral]
DELIVERY: [Your detailed 3-5 sentence description here]"""
                                }
                            ]
                        }
                    ]
                }
            )
            
            # Clean up temp file
            os.unlink(temp_path)
            
            if analysis_response.status_code != 200:
                print(f"    Analysis failed: {analysis_response.text}")
                return {
                    "text": transcription,
                    "delivery": "neutral conversational tone",
                    "voice_gender": "neutral"
                }
            
            analysis_text = analysis_response.json()['choices'][0]['message']['content']
            
            # Parse the response
            voice_gender = "neutral"
            delivery = "neutral conversational tone"
            
            lines = analysis_text.strip().split('\n')
            for line in lines:
                if line.startswith("GENDER:"):
                    voice_gender = line.replace("GENDER:", "").strip().lower()
                elif line.startswith("DELIVERY:"):
                    delivery = line.replace("DELIVERY:", "").strip()
            
            print(f"    ✓ Transcribed: {transcription[:50]}...")
            print(f"    ✓ Gender: {voice_gender}")
            print(f"    ✓ Delivery: {delivery[:80]}...")
            
            return {
                "text": transcription,
                "delivery": delivery,
                "voice_gender": voice_gender
            }
            
        except Exception as e:
            print(f"    API analysis error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "text": "[analysis failed]",
                "delivery": "neutral conversational tone",
                "voice_gender": "neutral"
            }
    
    def assign_speaker_to_segment(self, segment_start: float, segment_end: float, speaker_segments: Dict) -> str:
        """Assign a speaker to a dialogue segment based on diarization"""
        if not speaker_segments:
            return "Unknown"
        
        # Find which speaker has the most overlap with this segment
        max_overlap = 0
        best_speaker = "Unknown"
        
        for speaker, segments in speaker_segments.items():
            overlap = 0
            for seg in segments:
                # Calculate overlap
                overlap_start = max(segment_start, seg["start"])
                overlap_end = min(segment_end, seg["end"])
                if overlap_end > overlap_start:
                    overlap += (overlap_end - overlap_start)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = speaker
        
        return best_speaker
    
    def transcribe_dialogue_stem(self, vocals_path: str) -> List[Dict]:
        """Transcribe dialogue stem into individual lines with detailed API analysis"""
        print(f"Transcribing dialogue from: {vocals_path}")
        
        # Load vocals
        vocals_audio, sr = librosa.load(vocals_path, sr=self.config.sample_rate)
        
        # Detect speech segments (non-silent regions)
        # Use higher top_db to be less sensitive to brief pauses
        intervals = librosa.effects.split(vocals_audio, top_db=40)
        
        print(f"Detected {len(intervals)} raw speech segments")
        
        # Merge nearby segments into longer chunks
        # This prevents splitting on brief pauses within sentences
        merged_intervals = []
        MIN_SILENCE_GAP = 0.5  # Merge segments separated by less than 0.5 seconds
        MIN_SEGMENT_DURATION = 0.5  # Skip segments shorter than 0.5 seconds
        
        for start_sample, end_sample in intervals:
            duration = (end_sample - start_sample) / sr
            
            # Skip very short segments (likely noise)
            if duration < MIN_SEGMENT_DURATION:
                continue
            
            # If we have previous segments, check if we should merge
            if merged_intervals:
                prev_start, prev_end = merged_intervals[-1]
                gap = (start_sample - prev_end) / sr
                
                # Merge if gap is small (brief pause within sentence)
                if gap < MIN_SILENCE_GAP:
                    # Extend the previous segment to include this one
                    merged_intervals[-1] = (prev_start, end_sample)
                    continue
            
            # Add as new segment
            merged_intervals.append((start_sample, end_sample))
        
        print(f"Merged into {len(merged_intervals)} longer segments")
        
        # Get speaker diarization if enabled
        speaker_segments = {}
        if self.config.use_diarization:
            speaker_segments = self.diarize_speakers(vocals_path)
        
        dialogue_lines = []
        
        for i, (start_sample, end_sample) in enumerate(merged_intervals):
            start_time = start_sample / sr
            end_time = end_sample / sr
            duration = end_time - start_time
            
            print(f"  Segment {i+1}/{len(merged_intervals)}: {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s)")
            
            segment_audio = vocals_audio[start_sample:end_sample]
            
            # Use API to analyze this segment
            analysis = self.analyze_segment_with_api(segment_audio, sr)
            
            # Assign speaker based on diarization
            speaker_id = self.assign_speaker_to_segment(start_time, end_time, speaker_segments)
            
            # Build speaker profile if we haven't seen this speaker yet
            if speaker_id not in self.speaker_profiles:
                voice_chars = self.analyze_voice_characteristics(segment_audio, sr)
                self.speaker_profiles[speaker_id] = {
                    "voice_type": analysis["voice_gender"],
                    "characteristics": voice_chars,
                    "first_appearance": float(start_time)
                }
            
            # Create dialogue entry
            # Use RMS (root mean square) for proper volume calculation
            rms_value = np.sqrt(np.mean(segment_audio**2))
            volume_db = float(20 * np.log10(rms_value + 1e-10))
            
            dialogue_lines.append({
                "start": float(start_time),
                "end": float(end_time),
                "duration": float(duration),
                "text": analysis["text"],
                "speaker": speaker_id,
                "delivery": analysis["delivery"],
                "volume_db": volume_db
            })
        
        print(f"✓ Transcribed {len(dialogue_lines)} dialogue lines")
        return dialogue_lines
    
    def analyze_music_stem(self, music_path: str) -> List[Dict]:
        """Analyze music characteristics with better segmentation and merging"""
        print(f"Analyzing music from: {music_path}")
        music_audio, sr = librosa.load(music_path, sr=self.config.sample_rate)
        
        # Use original sensitivity - not too aggressive
        intervals = librosa.effects.split(music_audio, top_db=40)
        
        music_sections = []
        for start_sample, end_sample in intervals:
            start_time = start_sample / sr
            end_time = end_sample / sr
            duration = end_time - start_time
            
            # Keep segments >1.0s for stability
            if duration < 1.0:
                continue
            
            segment = music_audio[start_sample:end_sample]
            
            # Analyze musical characteristics
            try:
                tempo, _ = librosa.beat.beat_track(y=segment, sr=sr)
                # Fix tempo 0 issue - use default if detection fails
                if tempo <= 0 or np.isnan(tempo):
                    tempo = 90.0  # Default moderate tempo
            except:
                tempo = 90.0  # Fallback tempo
            
            spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr).mean()
            rms = librosa.feature.rms(y=segment).mean()
            
            # More nuanced mood detection
            if spectral_centroid < 800:
                mood = "deep, dark bass"
            elif spectral_centroid < 1500:
                mood = "warm, atmospheric"
            elif spectral_centroid < 2500:
                mood = "balanced, mid-range"
            elif spectral_centroid < 4000:
                mood = "bright, energetic"
            else:
                mood = "crisp, high energy"
            
            # Better tempo categorization
            if tempo < 60:
                tempo_desc = "very slow, ambient"
            elif tempo < 90:
                tempo_desc = "slow, relaxed"
            elif tempo < 110:
                tempo_desc = "moderate"
            elif tempo < 130:
                tempo_desc = "upbeat"
            else:
                tempo_desc = "fast, energetic"
            
            description = f"{tempo_desc}, {mood}"
            
            music_sections.append({
                "start": float(start_time),
                "end": float(end_time),
                "duration": float(duration),
                "description": description,
                "tempo": float(tempo),
                "spectral_centroid": float(spectral_centroid),
                "volume_db": float(20 * np.log10(rms + 1e-10))
            })
        
        # MERGE consecutive segments with similar characteristics
        # Only merge if VERY similar to avoid over-merging
        merged_sections = []
        i = 0
        while i < len(music_sections):
            current = music_sections[i]
            
            # Look ahead to merge similar consecutive segments
            while i + 1 < len(music_sections):
                next_seg = music_sections[i + 1]
                
                # Check if segments are consecutive (gap < 0.3s - tighter threshold)
                gap = next_seg["start"] - current["end"]
                
                # Check if characteristics are VERY similar (stricter)
                tempo_similar = abs(current["tempo"] - next_seg["tempo"]) < 10  # Stricter
                spectral_similar = abs(current["spectral_centroid"] - next_seg["spectral_centroid"]) < 300  # Stricter
                desc_similar = current["description"] == next_seg["description"]
                
                # Only merge if consecutive AND description matches exactly
                if gap < 0.3 and desc_similar:
                    print(f"  Merging music segments: {current['start']:.1f}-{current['end']:.1f} + {next_seg['start']:.1f}-{next_seg['end']:.1f}")
                    # Extend current segment
                    current["end"] = next_seg["end"]
                    current["duration"] = current["end"] - current["start"]
                    # Average the volume
                    current["volume_db"] = (current["volume_db"] + next_seg["volume_db"]) / 2
                    i += 1  # Skip the merged segment
                else:
                    break
            
            merged_sections.append(current)
            i += 1
        
        print(f"✓ Found {len(music_sections)} music sections, merged to {len(merged_sections)}")
        return merged_sections
    
    def analyze_sfx_stem(self, sfx_path: str) -> List[Dict]:
        """Analyze sound effects"""
        print(f"Analyzing sound effects from: {sfx_path}")
        sfx_audio, sr = librosa.load(sfx_path, sr=self.config.sample_rate)
        
        onset_frames = librosa.onset.onset_detect(y=sfx_audio, sr=sr, backtrack=True)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        sfx_events = []
        for i, timestamp in enumerate(onset_times):
            start_sample = onset_frames[i]
            end_sample = onset_frames[i+1] if i+1 < len(onset_frames) else len(sfx_audio)
            
            segment = sfx_audio[start_sample:end_sample]
            duration = (end_sample - start_sample) / sr
            
            if duration < 0.05 or duration > 3.0:
                continue
            
            spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr).mean()
            rms = librosa.feature.rms(y=segment).mean()
            
            if spectral_centroid > 3000:
                effect_type = "high-pitched transient"
            elif spectral_centroid > 1000:
                effect_type = "mid-range impact"
            else:
                effect_type = "low rumble"
            
            sfx_events.append({
                "timestamp": float(timestamp),
                "duration": float(duration),
                "type": effect_type,
                "volume_db": float(20 * np.log10(rms + 1e-10))
            })
        
        print(f"✓ Found {len(sfx_events)} sound effects")
        return sfx_events
    
    def create_semantic_description(self, stems: Dict[str, str]) -> Dict:
        """Create complete semantic description of audio"""
        print("\n=== Creating Semantic Description ===")
        
        description = {
            "total_duration": float(self.duration),
            "sample_rate": self.config.sample_rate,
            "dialogue": [],
            "music": [],
            "sound_effects": [],
            "speakers": {}
        }
        
        # Analyze dialogue
        if "dialogue" in stems:
            description["dialogue"] = self.transcribe_dialogue_stem(stems["dialogue"])
            description["speakers"] = convert_to_serializable(self.speaker_profiles)
        
        # Analyze music
        if "music" in stems:
            description["music"] = self.analyze_music_stem(stems["music"])
        
        # Analyze SFX
        if "sfx" in stems:
            description["sound_effects"] = self.analyze_sfx_stem(stems["sfx"])
        
        return description

# ============================================================================
# PHASE 2: AUDIO REGENERATION
# ============================================================================

class AudioRegenerator:
    """Regenerates audio from semantic descriptions"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def smooth_time_stretch(self, audio: np.ndarray, target_duration: float, sample_rate: int) -> np.ndarray:
        """
        High-quality time stretching with smoothing to reduce artifacts
        """
        target_samples = int(target_duration * sample_rate)
        current_duration = len(audio) / sample_rate
        
        if current_duration <= 0:
            return np.zeros(target_samples)
        
        # Calculate stretch factor
        stretch_factor = current_duration / target_duration
        
        # Only apply time stretching if the difference is significant (>2%)
        # Small differences can be handled with simple resampling
        if abs(stretch_factor - 1.0) < 0.02:
            # Very minor adjustment - just resample directly
            if len(audio) < target_samples:
                audio = np.pad(audio, (0, target_samples - len(audio)))
            else:
                audio = audio[:target_samples]
            return audio
        
        # Apply high-quality time stretching with phase vocoder
        # n_fft controls frequency resolution (higher = smoother but slower)
        # hop_length controls time resolution
        stretched_audio = librosa.effects.time_stretch(
            audio, 
            rate=stretch_factor,
            n_fft=2048,  # Higher FFT for better quality
            hop_length=512  # Good balance of quality and smoothness
        )
        
        # Apply gentle low-pass filter to reduce high-frequency artifacts from stretching
        # Only for significant time changes (>10% speed change)
        if abs(stretch_factor - 1.0) > 0.1:
            # Smooth out artifacts with a gentle filter
            from scipy import signal
            # Butterworth filter - very smooth, no ripples
            nyquist = sample_rate / 2
            cutoff = nyquist * 0.95  # Gentle high-frequency rolloff
            b, a = signal.butter(4, cutoff / nyquist, btype='low')
            stretched_audio = signal.filtfilt(b, a, stretched_audio)
        
        # Final trim/pad to exact sample count
        if len(stretched_audio) < target_samples:
            stretched_audio = np.pad(stretched_audio, (0, target_samples - len(stretched_audio)))
        else:
            stretched_audio = stretched_audio[:target_samples]
        
        return stretched_audio
    
    def generate_dialogue(self, text: str, speaker_id: str, speaker_profile: Dict, 
                         delivery: str, duration: float) -> np.ndarray:
        """Generate dialogue using TTS with delivery instructions"""
        
        if self.config.use_delivery_control:
            # Use gpt-4o-mini or gpt-4o-audio-preview with delivery control
            return self.generate_dialogue_with_delivery(text, speaker_profile, delivery, duration)
        else:
            # Use basic tts-1 without delivery control
            return self.generate_dialogue_basic(text, speaker_profile, duration)
    
    def generate_dialogue_basic(self, text: str, speaker_profile: Dict, duration: float) -> np.ndarray:
        """Generate dialogue using basic Speech API (tts-1 or tts-1-hd)"""
        voice_type = speaker_profile.get("voice_type", "neutral")
        
        # Map to OpenAI voices for Speech API
        voice_map = {
            "male": "onyx",
            "female": "nova",
            "neutral": "alloy"
        }
        voice = voice_map.get(voice_type, "alloy")
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/audio/speech",
                headers={"Authorization": f"Bearer {self.config.openai_api_key}"},
                json={
                    "model": self.config.tts_model,  # Use configured model (tts-1, tts-1-hd, etc)
                    "input": text,
                    "voice": voice
                }
            )
            
            if response.status_code == 200:
                temp_path = "temp_tts.mp3"
                with open(temp_path, "wb") as f:
                    f.write(response.content)
                
                audio, sr = librosa.load(temp_path, sr=self.config.sample_rate)
                os.remove(temp_path)
                
                # Apply high-quality time-stretch
                audio = self.smooth_time_stretch(audio, duration, self.config.sample_rate)
                
                return audio
        except Exception as e:
            print(f"      TTS error: {e}")
            import traceback
            traceback.print_exc()
        
        return np.zeros(int(duration * self.config.sample_rate))
    
    def generate_dialogue_with_delivery(self, text: str, speaker_profile: Dict, 
                                       delivery: str, duration: float) -> np.ndarray:
        """Generate dialogue using gpt-4o-mini-tts with REAL delivery control via instructions parameter"""
        voice_type = speaker_profile.get("voice_type", "neutral")
        
        # Map to gpt-4o-mini-tts voices (expanded voice options)
        voice_map = {
            "male": "onyx",      # Deep, authoritative male
            "female": "nova",    # Warm, friendly female
            "neutral": "alloy"   # Balanced, versatile
        }
        voice = voice_map.get(voice_type, "alloy")
        
        print(f"      Model: {self.config.tts_model}, Voice: {voice}")
        print(f"      Delivery: {delivery[:80]}...")
        
        try:
            # Use the SPEECH API with instructions parameter
            # This is the CORRECT way to get delivery control!
            response = requests.post(
                "https://api.openai.com/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {self.config.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.config.tts_model,  # gpt-4o-mini-tts
                    "input": text,
                    "voice": voice,
                    "instructions": delivery,  # THIS is where delivery control happens!
                    "response_format": "mp3"
                }
            )
            
            if response.status_code == 200:
                # Save audio to temp file
                temp_path = "temp_tts.mp3"
                with open(temp_path, "wb") as f:
                    f.write(response.content)
                
                # Load and process audio
                audio, sr = librosa.load(temp_path, sr=self.config.sample_rate)
                os.remove(temp_path)
                
                print(f"      Generated {len(audio)/sr:.2f}s of audio")
                
                # Apply high-quality time-stretch to match exact duration
                audio = self.smooth_time_stretch(audio, duration, self.config.sample_rate)
                
                return audio
            else:
                print(f"      ERROR {response.status_code}: {response.text}")
                    
        except Exception as e:
            print(f"      TTS error: {e}")
            import traceback
            traceback.print_exc()
        
        return np.zeros(int(duration * self.config.sample_rate))
    
    def generate_music(self, description: str, tempo: float, duration: float) -> np.ndarray:
        """Generate music using FAL.AI"""
        print(f"    Generating music: {description} ({duration:.1f}s)")
        
        url = "https://fal.run/fal-ai/stable-audio"
        headers = {
            "Authorization": f"Key {self.config.fal_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "prompt": description,
            "duration": min(duration, 47)
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                audio_url = result.get("audio_file", {}).get("url")
                
                if audio_url:
                    audio_response = requests.get(audio_url)
                    temp_path = "temp_music.wav"
                    with open(temp_path, "wb") as f:
                        f.write(audio_response.content)
                    
                    audio, sr = librosa.load(temp_path, sr=self.config.sample_rate)
                    os.remove(temp_path)
                    
                    # Adjust to target duration
                    target_samples = int(duration * self.config.sample_rate)
                    if len(audio) < target_samples:
                        audio = np.tile(audio, int(np.ceil(target_samples / len(audio))))[:target_samples]
                    else:
                        audio = audio[:target_samples]
                    
                    return audio
        except Exception as e:
            print(f"      Music generation error: {e}")
        
        return np.zeros(int(duration * self.config.sample_rate))
    
    def generate_sfx(self, effect_type: str, duration: float) -> np.ndarray:
        """Generate sound effect (procedural)"""
        sr = self.config.sample_rate
        samples = int(duration * sr)
        t = np.linspace(0, duration, samples)
        
        if "high" in effect_type:
            freq = 2000
            decay = np.exp(-t * 15)
            audio = np.sin(2 * np.pi * freq * t) * decay
        elif "rumble" in effect_type or "low" in effect_type:
            freq = 80
            decay = np.exp(-t * 5)
            audio = np.sin(2 * np.pi * freq * t) * decay
        else:
            freq = 200
            decay = np.exp(-t * 10)
            audio = np.sin(2 * np.pi * freq * t) * decay
            audio += np.random.randn(samples) * 0.1 * decay
        
        return audio * 0.3
    
    def regenerate_from_description(self, description_path: str, output_path: str):
        """Regenerate audio from semantic description"""
        print(f"\n=== Regenerating Audio ===")
        print(f"Loading description from {description_path}...")
        
        with open(description_path, 'r') as f:
            desc = json.load(f)
        
        total_duration = desc["total_duration"]
        sample_rate = desc.get("sample_rate", self.config.sample_rate)
        
        total_samples = int(total_duration * sample_rate)
        dialogue_track = np.zeros(total_samples)
        music_track = np.zeros(total_samples)
        sfx_track = np.zeros(total_samples)
        
        speakers = desc.get("speakers", {})
        dialogue_segments = desc.get("dialogue", [])
        music_segments = desc.get("music", [])
        sfx_segments = desc.get("sound_effects", [])
        
        print(f"Regenerating {len(dialogue_segments)} dialogue lines...")
        print(f"Regenerating {len(music_segments)} music sections...")
        print(f"Regenerating {len(sfx_segments)} sound effects...")
        
        # Generate dialogue
        for i, seg in enumerate(dialogue_segments):
            if i % 5 == 0:
                print(f"  Dialogue {i+1}/{len(dialogue_segments)}")
            
            speaker_profile = speakers.get(seg["speaker"], {})
            audio = self.generate_dialogue(
                seg["text"],
                seg["speaker"],
                speaker_profile,
                seg["delivery"],
                seg["duration"]
            )
            
            # Place in track using the ORIGINAL volume from analysis with a minimum threshold
            start_sample = int(seg["start"] * sample_rate)
            end_sample = min(start_sample + len(audio), total_samples)
            audio_samples = end_sample - start_sample
            
            if audio_samples > 0:
                # CRITICAL: Dialogue must always be clear and prominent
                # Professional mixing: dialogue at -6dB to -3dB, music at -15dB to -10dB
                MIN_DIALOGUE_DB = -10  # Minimum for audibility
                REFERENCE_DIALOGUE_DB = -6  # Target "normal" dialogue level (clear and prominent)
                MAX_DIALOGUE_DB = -3   # Maximum (loud dialogue/shouting)
                
                # Smart dialogue volume adjustment
                if seg["volume_db"] < MIN_DIALOGUE_DB:
                    # Too quiet - boost to reference
                    adjusted_volume_db = REFERENCE_DIALOGUE_DB
                elif seg["volume_db"] > MAX_DIALOGUE_DB:
                    # Too loud - cap it
                    adjusted_volume_db = MAX_DIALOGUE_DB
                else:
                    # In acceptable range - use original
                    adjusted_volume_db = seg["volume_db"]
                
                volume_linear = librosa.db_to_amplitude(adjusted_volume_db)
                dialogue_track[start_sample:end_sample] += audio[:audio_samples] * volume_linear
        
        # Generate music  
        for i, seg in enumerate(music_segments):
            print(f"  Music {i+1}/{len(music_segments)}: {seg['start']:.1f}s-{seg['end']:.1f}s")
            
            audio = self.generate_music(seg["description"], seg["tempo"], seg["duration"])
            
            start_sample = int(seg["start"] * sample_rate)
            end_sample = min(start_sample + len(audio), total_samples)
            audio_samples = end_sample - start_sample
            
            if audio_samples > 0:
                # CRITICAL: Music must be audible but never overpower dialogue
                # Set volume levels based on typical professional mixing standards
                MUSIC_TARGET_DB = -15     # Target music level (clear but background)
                MUSIC_ABSOLUTE_MIN = -20  # Absolute floor - anything quieter gets boosted
                MUSIC_MAX_DB = -10        # Maximum (for sections without dialogue)
                
                original_volume = seg["volume_db"]
                
                # Smart volume adjustment
                if original_volume < MUSIC_ABSOLUTE_MIN:
                    # Very quiet music - boost significantly
                    adjusted_music_db = MUSIC_TARGET_DB
                    print(f"      Music BOOST: {original_volume:.1f}dB → {adjusted_music_db:.1f}dB (was too quiet)")
                elif original_volume < MUSIC_TARGET_DB:
                    # Somewhat quiet - boost to target
                    adjusted_music_db = MUSIC_TARGET_DB
                elif original_volume > MUSIC_MAX_DB:
                    # Too loud - reduce
                    adjusted_music_db = MUSIC_MAX_DB
                    print(f"      Music REDUCE: {original_volume:.1f}dB → {adjusted_music_db:.1f}dB (was too loud)")
                else:
                    # In good range
                    adjusted_music_db = original_volume
                
                volume_linear = librosa.db_to_amplitude(adjusted_music_db)
                music_track[start_sample:end_sample] += audio[:audio_samples] * volume_linear
        
        # Generate SFX
        for i, seg in enumerate(sfx_segments):
            if i % 10 == 0:
                print(f"  SFX {i+1}/{len(sfx_segments)}")
            
            audio = self.generate_sfx(seg["type"], seg["duration"])
            
            start_sample = int(seg["timestamp"] * sample_rate)
            end_sample = min(start_sample + len(audio), total_samples)
            audio_samples = end_sample - start_sample
            
            if audio_samples > 0:
                volume_linear = librosa.db_to_amplitude(seg["volume_db"])
                sfx_track[start_sample:end_sample] += audio[:audio_samples] * volume_linear
        
        # Mix tracks
        print("Mixing tracks...")
        mixed_audio = dialogue_track + music_track + sfx_track
        
        # Peak normalize to ensure good volume without clipping
        # Use 0.9 instead of 0.95 for more headroom
        max_val = np.abs(mixed_audio).max()
        if max_val > 0:
            # Normalize to -1dB (0.9 linear) for professional headroom
            mixed_audio = mixed_audio / max_val * 0.9
            
            # Apply gentle compression if peaks are still too quiet
            # This ensures the output has good overall volume
            target_peak = 0.8
            current_peak = np.abs(mixed_audio).max()
            if current_peak < 0.3:  # If mix is very quiet
                # Boost gently to bring up to reasonable level
                boost_factor = target_peak / (current_peak + 1e-10)
                boost_factor = min(boost_factor, 3.0)  # Cap at 3x boost
                mixed_audio = mixed_audio * boost_factor
                # Re-normalize to prevent any clipping from boost
                max_val = np.abs(mixed_audio).max()
                if max_val > 0.9:
                    mixed_audio = mixed_audio / max_val * 0.9
        
        sf.write(output_path, mixed_audio, sample_rate)
        print(f"✓ Saved to {output_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def analyze_audio(input_path: str, output_json: str):
    """Analyze audio and save semantic description"""
    analyzer = AudioAnalyzer(input_path, config)
    stems = analyzer.separate_stems_advanced()
    description = analyzer.create_semantic_description(stems)
    
    with open(output_json, 'w') as f:
        json.dump(convert_to_serializable(description), f, indent=2)
    
    print(f"\n✓ Semantic description saved to {output_json}")
    return description

def regenerate_audio(description_json: str, output_path: str):
    """Regenerate audio from semantic description"""
    regenerator = AudioRegenerator(config)
    regenerator.regenerate_from_description(description_json, output_path)

if __name__ == "__main__":
    print("Semantic Audio Compression System v3.1 - gpt-4o-mini-tts Edition")
    print("Now with REAL delivery control via instructions parameter!")
    print("")
    print("Usage: analyze_audio('input.mp3', 'description.json')")
    print("       regenerate_audio('description.json', 'output.wav')")