# Install F5TTS - comment this out later
# os.system("git clone https://github.com/SWivid/F5-TTS.git")
# os.chdir("F5-TTS")
# os.system("pip install -e .")

import subprocess
import os
from pathlib import Path

def setup_f5tts():
    """
    Setup the F5TTS environment.
    """
    # Check if the model directory exists, if not create it
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # Download the F5TTS_v1_Base model if not already present
    if not os.path.exists("models/F5TTS_v1_Base"):
        print("Downloading F5TTS model...")
        # You would need to manually download the model from the official source
        # This is just a placeholder - the actual command depends on where the model is hosted
        # os.system("wget [model_url] -O models/F5TTS_v1_Base.zip")
        # os.system("unzip models/F5TTS_v1_Base.zip -d models/")
    
    return True


def clone_voice_and_generate_speech_optimized(reference_audio, fine_tuned_whisper_model, output_path="output.wav"):
    """
    Clone a voice from a reference audio file and generate fluent speech using a fine-tuned Whisper model.
    
    Args:
        reference_audio: Path to the reference audio file for voice cloning
        fine_tuned_whisper_model: The fine-tuned Whisper model for stuttering
        output_path: Path to save the generated audio
    
    Returns:
        Path to the generated audio file and the fluent text
    """
    try:
        # Step 1: Transcribe the reference audio for voice cloning
        # We use standard Whisper for this to get the actual spoken content
        standard_whisper = setup_whisper("small")
        ref_text = transcribe_audio(standard_whisper, reference_audio)
        
        # Step 2: Generate fluent text from the same audio using fine-tuned model
        # This is the key optimization - we use our stuttering-aware model
        fluent_text = transcribe_audio(fine_tuned_whisper_model, reference_audio)
        
        # Step 3: Run F5TTS to generate speech with cloned voice saying the fluent text
        cmd = [
            "f5-tts_infer-cli",
            "--model", "F5TTS_v1_Base",
            "--ref_audio", reference_audio,
            "--ref_text", ref_text,
            "--gen_text", fluent_text,
            "--output", output_path
        ]
        
        subprocess.run(cmd, check=True)
        
        return output_path, fluent_text
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None, None
    
    # def clone_voice_and_generate_speech(reference_audio, text_to_speak, output_path="output.wav"):
#     """
#     Clone a voice from a reference audio file and generate speech.
    
#     Args:
#         reference_audio: Path to the reference audio file for voice cloning
#         text_to_speak: Text to convert to speech
#         output_path: Path to save the generated audio
    
#     Returns:
#         Path to the generated audio file
#     """
#     # Use F5TTS CLI tool to generate speech
#     try:
#         # Transcribe the reference audio to get the text
#         whisper_model = setup_whisper("small")
#         ref_text = transcribe_audio(whisper_model, reference_audio)
        
#         # Run F5TTS to generate speech with cloned voice
#         cmd = [
#             "f5-tts_infer-cli",
#             "--model", "F5TTS_v1_Base",
#             "--ref_audio", reference_audio,
#             "--ref_text", ref_text,
#             "--gen_text", text_to_speak,
#             "--output", output_path
#         ]
        
#         subprocess.run(cmd, check=True)
        
#         return output_path
#     except Exception as e:
#         print(f"Error generating speech: {e}")
#         return None