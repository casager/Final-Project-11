#!/usr/bin/env python3
"""
Simple test script for voice cloning with F5TTS and Fine-tuned Whisper.
"""

import os
import subprocess
import whisper
import argparse
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import torch
from whisper_utils import setup_whisper, transcribe_audio
import soundfile as sf  # Add this import at the top with your other imports
import pandas as pd
from pydub import AudioSegment


# Change to F5-TTS directory
os.chdir("../../F5-TTS")
FILE_PREFIX="24fa_000"
# FILE_PREFIX="37m_012"

# Hardcoded parameters
AUDIO_FILE = f"/home/ubuntu/TimeStamped-wav/combined_wavs/{FILE_PREFIX}_combined.wav"  # Using original wave with stutters for audio ref.
# AUDIO_FILE = f"/home/ubuntu/TimeStamped-wav/original_wavs/{FILE_PREFIX}.wav"
REF_CSV = f"/home/ubuntu/TimeStamped/combined_csvs/{FILE_PREFIX}_combined.csv"
OUTPUT_FILE = f"/home/ubuntu/Final-Project-11/Code/Audio-Samples/project_audio/{FILE_PREFIX}_cloned.wav"
TEXT_FILE = "/home/ubuntu/Final-Project-11/Code/Audio-Samples/custom_text.txt"
FINE_TUNED_MODEL_PATH = "/home/ubuntu/Final-Project-11/Code/whisper-fine-tuned-stuttering-final-4"  # Path to your fine-tuned model

def load_text_from_file(file_path):
    try:
        with open(file_path, "r") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading text file: {e}")
        return None
    
CUSTOM_TEXT = load_text_from_file(TEXT_FILE)

def append_silence_to_audio(file_prefix, silence_duration_ms=1000):
    """Appends silence to the end of the given audio and saves to silenced_wavs."""
    input_path = f"/home/ubuntu/TimeStamped-wav/combined_wavs/{file_prefix}_combined.wav"
    output_path = f"/home/ubuntu/TimeStamped-wav/silenced_wavs/{file_prefix}_combined_silenced.wav"

    audio = AudioSegment.from_wav(input_path)
    silence = AudioSegment.silent(duration=silence_duration_ms)
    audio_with_silence = audio + silence

    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure output dir exists
    audio_with_silence.export(output_path, format="wav")

    return output_path

def load_fine_tuned_whisper(model_path):
    """Load the fine-tuned Whisper model."""
    print(f"Loading fine-tuned Whisper model from {model_path}...")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    return processor, model

def transcribe_with_base_model(model, audio_file):
    """
    Transcribe an audio file using Whisper.
    
    Args:
        model: The loaded Whisper model
        audio_file: Path to the audio file
    
    Returns:
        The transcribed text
    """
    # Transcribe the audio file
    result = model.transcribe(audio_file)
    return result["text"]

def transcribe_with_fine_tuned_whisper(processor, model, audio_file):
    """Transcribe audio using the fine-tuned Whisper model."""
    # Load and preprocess audio
    audio, sr = librosa.load(audio_file, sr=16000)
    
    # Process audio for input
    input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
    
    # Clear the forced_decoder_ids and set language and task
    model.generation_config.forced_decoder_ids = None

    # Generate transcription
    predicted_ids = model.generate(
        input_features,
        language='en',
        task='transcribe',
        use_cache=True
    )
    
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def test_voice_cloning(reference_audio, output_file=None, custom_text=None):
    """Test voice cloning with F5TTS using fine-tuned Whisper."""
    # Set default output path if not provided
    if not output_file:
        dir_name = os.path.dirname(os.path.abspath(reference_audio))
        base_name = os.path.splitext(os.path.basename(reference_audio))[0]
        output_file = os.path.join(dir_name, f"{base_name}_cloned.wav")
    
    # Load fine-tuned Whisper model
    fine_tuned_processor, fine_tuned_model = load_fine_tuned_whisper(FINE_TUNED_MODEL_PATH)
    
    # Step 1: Transcribe the reference audio for voice cloning using fine-tuned model
    print("Transcribing with fine-tuned Whisper model...")
    fluent_text = transcribe_with_fine_tuned_whisper(fine_tuned_processor, fine_tuned_model, reference_audio)
    print(f"Fine-tuned transcription (fluent): {fluent_text}")
    
    # Step 2: For comparison, also get standard Whisper transcription
    standard_whisper = setup_whisper("base")
    standard_text = transcribe_audio(standard_whisper, reference_audio)
    print(f"Standard Whisper transcription: {standard_text}")
    
    # For F5TTS, we'll use the fluent (fine-tuned) transcription as reference
    # ref_text = fluent_text
    ref_text = ' ' + fluent_text.strip().rstrip('.') + '.'
    # ref_text = standard_text #this is using the base model transcription...

    # df = pd.read_csv(REF_CSV) #takes the actual stuttering csv for reference
    # ref_text = ' '.join(df['word'].astype(str)).strip().rstrip('.') + '.'
    
    # Determine text to generate
    # gen_text = custom_text if custom_text else ref_text #THIS NEEDS TO CHANGE BASED ON MODEL 
    gen_text = ' ' + fluent_text.strip().rstrip('.') + '.' #uses the fluent transcription for generating
    if custom_text:
        print(f"Using custom text: {gen_text}")
    else:
        print(f"Using fluent transcription for generation: {gen_text}")
    
    # Run F5TTS for voice cloning
    print("Running F5TTS...")
    cmd = [
        "f5-tts_infer-cli",
        "--model", "F5TTS_v1_Base",  # or E2TTS_Base
        "--ref_audio", reference_audio,
        "--ref_text", ref_text,  
        "--gen_text", gen_text, # Using fluent transcription
        "--output_file", output_file
        # "--speed", "0.85" 
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
    # Run the test with hardcoded parameters

    AUDIO_FILE_WITH_SILENCE = append_silence_to_audio(FILE_PREFIX)

    output_path = test_voice_cloning(
        reference_audio=AUDIO_FILE_WITH_SILENCE,
        output_file=OUTPUT_FILE
    )

    # output_path = test_voice_cloning(
    #     reference_audio=AUDIO_FILE,
    #     output_file=OUTPUT_FILE
    # )
    
    if output_path:
        print(f"\nTest completed successfully. Generated file: {output_path}")
    else:
        print("\nTest failed.")


if __name__ == "__main__":
    main()