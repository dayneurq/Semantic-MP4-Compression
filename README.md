# Semantic-MP4-Compression

**What if you stored what a scene *means* instead of what it looks and sounds like?**

Conventional codecs (H.264, AAC) compress the raw signal. This project takes a
different approach: it describes a video in text using audio and vision
models, then regenerates the video and soundtrack from that text with
generative models.

> **Result:** a 10-minute, 190 MB film clip became **142.1 KB of text**
> (92.7 KB video blueprint + 49.4 KB audio blueprint), a compression ratio of
> roughly **1,300x**, then was regenerated from that text alone.

📺 **Video walkthrough:** https://youtu.be/odjmAKvikHM

```
                ┌─► audio ─► ENCODE: stems, diarization, transcription,   ─► compressed.json (49.4 KB)
 MP4 (190 MB) ──┤            delivery analysis, music/SFX features              │
                └─► video ─► ENCODE: shot detection + vision-language     ─► video_blueprint.txt (92.7 KB)
                             descriptions                                       │
                                                                                ▼
                                  DECODE: TTS + music generation + text/image-to-video
                                                                                ▼
                                                              regenerated audio + video
```

The output is not the original footage. It is a new rendering of the same
scene. The compressed form keeps meaning (who is there, what they do and say,
how it sounds and looks) and discards the exact signal.

## Why it's interesting

- **Extreme ratios.** The compressed form of a 190 MB clip is smaller than a
  single JPEG.
- **Readable and editable.** The blueprints are plain text and JSON. Rewrite a
  line of dialogue, a delivery note, or a shot description, then regenerate.
- **A different kind of lossy.** Classic codecs lose fidelity and keep
  identity. This keeps meaning and loses identity: faces, voices, and music are
  re-created, not preserved. How much meaning survives is the open question.

## Pipeline

### Audio (`audio_compressor.py`)

**Encoder**
1. Separate vocals from the rest with Demucs, then split the remainder into
   harmonic ("music") and percussive ("SFX") stems with librosa HPSS
2. Diarize speakers with pyannote.audio
3. Segment dialogue by silence, transcribe each line with Whisper, and have
   `gpt-4o-audio-preview` describe its delivery (emotion, volume, pacing, vocal
   quality) and the speaker's voice gender
4. Summarize music sections (tempo, spectral brightness) and SFX events
   (onset detection) into short text tags
5. Write everything to `compressed.json`

**Decoder**
1. Speak each line with `gpt-4o-mini-tts`, passing the recorded delivery
   description as the TTS `instructions`, then time-stretch to the original
   line length
2. Generate music sections with Stable Audio (via fal.ai)
3. Synthesize placeholder SFX, level-match each stem, mix, and export a WAV

### Video (`video_breakdown.py`, `video_resynthesize.py`)

**Encoder**
1. Detect shot boundaries with PySceneDetect
2. Split long shots into segments of 5 s or less and sample 3 frames from each
3. Describe each segment with GPT-4o in a fixed SUBJECT / ACTION / SETTING
   format, with fallback prompts if the model refuses

**Decoder**
1. Generate the first segment of each shot from text (Wan text-to-video)
2. Chain later segments from the previous segment's final frame (Wan
   image-to-video) to keep continuity within a shot
3. Trim each clip to its original duration and stitch

## Setup

```bash
pip install moviepy numpy librosa soundfile scipy requests openai fal-client \
            scenedetect[opencv] demucs pyannote.audio torch
export OPENAI_API_KEY=...
export FAL_API_KEY=...
export HF_TOKEN=...   # for pyannote speaker diarization
```
Requires `ffmpeg`. **No audio or video is included in this repo.** Supply your
own clip. Filenames are currently hard-coded in the scripts (`bluevelvet.mp4`
and so on), so edit them for your input.

## Usage

```bash
python extract_audio.py          # MP4 -> MP3
python run_compression.py        # audio encode: MP3 -> compressed.json
python video_breakdown.py        # video encode: MP4 -> video_blueprint.txt

python regenerate_only.py        # audio decode: compressed.json -> output.wav
python video_resynthesize.py     # video decode: video_blueprint.txt -> remastered_10min.mp4

# combine the regenerated streams
ffmpeg -i remastered_10min.mp4 -i output.wav -c:v copy -c:a aac -shortest final.mp4
```
      between original and regenerated clips, human preference tests
- [ ] Map the tradeoff curve: how small can the blueprint get before the scene
      stops feeling the same?
