#!/usr/bin/env python3
"""
Simple test script for voice cloning with F5TTS and Whisper.
"""

import os
import subprocess
import whisper
import argparse
from whisper_utils import setup_whisper, transcribe_audio


os.chdir("../../F5-TTS")

# Hardcoded parameters
AUDIO_FILE = "/home/ubuntu/Final-Project-11/Code/Audio-Samples/local_audio/SOME_WAVE_FILE.wav"  # Change this to your audio file path
OUTPUT_FILE = "/home/ubuntu/Final-Project-11/Code/Audio-Samples/local_audio/cloned_output.wav"  # Change this to your desired output path
TEXT_FILE = "/home/ubuntu/Final-Project-11/Code/Audio-Samples/custom_text.txt"
# CUSTOM_TEXT = "This is a test of voice cloning with F5TTS. The voice should sound like mine, but saying different words."  # Optional: remove this or set to None to use transcription

def load_text_from_file(file_path):
    try:
        with open(file_path, "r") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading text file: {e}")
        return None
CUSTOM_TEXT = load_text_from_file(TEXT_FILE)
    
def test_voice_cloning(audio_file, output_file=None, custom_text=None):
    """Test voice cloning with F5TTS."""
    # Set default output path if not provided
    if not output_file:
        dir_name = os.path.dirname(os.path.abspath(audio_file))
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        output_file = os.path.join(dir_name, f"{base_name}_cloned.wav")
    
    # Load Whisper model
    whisper_model = setup_whisper("base")
    
    # Transcribe the reference audio
    ref_text = transcribe_audio(whisper_model, audio_file)
    
    # Determine text to generate
    gen_text = custom_text if custom_text else ref_text
    if custom_text:
        print(f"Using custom text: {gen_text}")
    
    # Run F5TTS for voice cloning
    print("Running F5TTS...")
    cmd = [
        "f5-tts_infer-cli",
        "--model", "E2TTS_Base", #F5TTS_v1_Base
        "--ref_audio", audio_file,
        "--ref_text", ref_text,
        "--gen_text", gen_text,
        "--output_file", output_file
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Voice cloning successful! Output saved to: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error during voice cloning: {e}")
        return None

def main():
    # parser = argparse.ArgumentParser(description="Test voice cloning with F5TTS")
    # parser.add_argument("--audio", required=True, help="Path to reference audio file")
    # parser.add_argument("--output", help="Path for output audio file")
    # parser.add_argument("--text", help="Custom text to generate (default: use transcription)")
    
    # args = parser.parse_args()
    
    # Run the test
    #USE IF WE WANT TO PASS AT COMMAND LINE
    # output_path = test_voice_cloning(
    #     audio_file=args.audio,
    #     output_file=args.output,
    #     custom_text=args.text
    # )
    output_path = test_voice_cloning(
        audio_file=AUDIO_FILE,
        output_file=OUTPUT_FILE,
        custom_text=CUSTOM_TEXT #want to use the whisper base output
    )
    
    if output_path:
        print(f"\nTest completed successfully. Generated file: {output_path}")
    else:
        print("\nTest failed.")

if __name__ == "__main__":
    main()