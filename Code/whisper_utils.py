# # Install Whisper and dependencies
# pip install openai-whisper
# pip install torch
# pip install ffmpeg-python

# # For handling audio files
# pip install pydub

import whisper
import numpy as np
import torch

def setup_whisper(model_size="base"):
    """
    Initialize the Whisper model.
    
    Args:
        model_size: Size of the model to use ("tiny", "base", "small", "medium", "large")
    
    Returns:
        The loaded Whisper model
    """
    # Load the Whisper model
    model = whisper.load_model(model_size)
    return model

def transcribe_audio(model, audio_file):
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