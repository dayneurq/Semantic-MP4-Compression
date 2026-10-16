#!/usr/bin/env python3
"""
Cost Estimator for Audio Regeneration
Calculates approximate costs for using gpt-4o-audio-preview
"""

import json
import sys

# Pricing as of January 2025 (check https://openai.com/api/pricing/ for current rates)
# gpt-4o-audio-preview pricing:
AUDIO_INPUT_COST_PER_TOKEN = 0.000100  # $0.10 per 1M tokens
AUDIO_OUTPUT_COST_PER_TOKEN = 0.000600  # $0.60 per 1M tokens

# Rough token estimates
SYSTEM_PROMPT_TOKENS = 150  # System prompt with delivery instructions
USER_PROMPT_TOKENS = 20     # "Please read this line: {text}"
TOKENS_PER_WORD = 1.3       # Average tokens per word for English text

def estimate_dialogue_cost(dialogue_segments):
    """Estimate cost for regenerating dialogue"""
    total_input_tokens = 0
    total_output_tokens = 0
    
    for seg in dialogue_segments:
        text = seg.get("text", "")
        words = len(text.split())
        
        # Input tokens: system prompt + user prompt + text
        input_tokens = SYSTEM_PROMPT_TOKENS + USER_PROMPT_TOKENS + (words * TOKENS_PER_WORD)
        total_input_tokens += input_tokens
        
        # Output tokens: roughly proportional to words spoken
        # Audio models generate ~50-100 tokens per second of speech
        # Assuming average speaking rate of 2-3 words per second
        duration = seg.get("duration", 0)
        output_tokens = duration * 75  # Conservative estimate: 75 tokens per second
        total_output_tokens += output_tokens
    
    input_cost = (total_input_tokens / 1_000_000) * AUDIO_INPUT_COST_PER_TOKEN * 1_000_000
    output_cost = (total_output_tokens / 1_000_000) * AUDIO_OUTPUT_COST_PER_TOKEN * 1_000_000
    total_cost = input_cost + output_cost
    
    return {
        "input_tokens": int(total_input_tokens),
        "output_tokens": int(total_output_tokens),
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

def estimate_music_cost(music_segments):
    """Estimate cost for music generation (if using FAL.AI)"""
    # FAL.AI Stable Audio pricing is roughly $0.01-0.05 per generation
    # depending on duration (up to 47 seconds per generation)
    total_cost = len(music_segments) * 0.03  # Average estimate
    return total_cost

def main():
    if len(sys.argv) < 2:
        print("Usage: python estimate_cost.py <description.json>")
        print("Example: python estimate_cost.py compressed.json")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        sys.exit(1)
    
    dialogue_segments = data.get("dialogue", [])
    music_segments = data.get("music", [])
    total_duration = data.get("total_duration", 0)
    
    print("="*70)
    print("AUDIO REGENERATION COST ESTIMATE")
    print("="*70)
    print(f"\nFile: {json_path}")
    print(f"Total Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"Dialogue Lines: {len(dialogue_segments)}")
    print(f"Music Sections: {len(music_segments)}")
    
    # Estimate dialogue cost
    print("\n" + "-"*70)
    print("DIALOGUE REGENERATION (gpt-4o-audio-preview)")
    print("-"*70)
    
    dialogue_cost = estimate_dialogue_cost(dialogue_segments)
    
    print(f"Estimated Input Tokens:  {dialogue_cost['input_tokens']:,}")
    print(f"Estimated Output Tokens: {dialogue_cost['output_tokens']:,}")
    print(f"\nInput Cost:  ${dialogue_cost['input_cost']:.4f}")
    print(f"Output Cost: ${dialogue_cost['output_cost']:.4f}")
    print(f"Total Dialogue Cost: ${dialogue_cost['total_cost']:.4f}")
    
    # Estimate music cost
    if music_segments:
        print("\n" + "-"*70)
        print("MUSIC REGENERATION (FAL.AI Stable Audio)")
        print("-"*70)
        music_cost = estimate_music_cost(music_segments)
        print(f"Estimated Music Cost: ${music_cost:.4f}")
        print(f"({len(music_segments)} generations × ~$0.03 each)")
    else:
        music_cost = 0
    
    # Total
    print("\n" + "="*70)
    total_cost = dialogue_cost['total_cost'] + music_cost
    print(f"ESTIMATED TOTAL COST: ${total_cost:.4f}")
    print("="*70)
    
    # Breakdown
    print("\nCost Breakdown:")
    print(f"  - Dialogue (gpt-4o-audio-preview): ${dialogue_cost['total_cost']:.4f} ({dialogue_cost['total_cost']/total_cost*100:.1f}%)")
    if music_cost > 0:
        print(f"  - Music (FAL.AI):                   ${music_cost:.4f} ({music_cost/total_cost*100:.1f}%)")
    
    # Alternatives
    print("\n" + "-"*70)
    print("COST COMPARISON WITH ALTERNATIVES")
    print("-"*70)
    
    # tts-1-hd estimate (much cheaper but no delivery control)
    tts_hd_cost = len(dialogue_segments) * 0.001  # Very rough: $0.001 per line
    print(f"tts-1-hd (no delivery control):  ${tts_hd_cost:.4f}")
    print(f"Savings vs gpt-4o-audio-preview:  ${dialogue_cost['total_cost'] - tts_hd_cost:.4f}")
    
    print("\n" + "="*70)
    print("NOTE: These are ESTIMATES. Actual costs may vary by ±30%")
    print("Check current pricing at: https://openai.com/api/pricing/")
    print("="*70)

if __name__ == "__main__":
    main()