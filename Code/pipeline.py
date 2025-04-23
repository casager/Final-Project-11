# def process_stuttered_speech(input_audio, output_audio="fluent_speech.wav"):
#     """
#     Process stuttered speech to fluent speech.
    
#     Args:
#         input_audio: Path to the stuttered speech audio file
#         output_audio: Path to save the generated fluent speech
    
#     Returns:
#         Path to the generated audio file with fluent speech in the user's voice
#     """
#     # Step 1: Load the fine-tuned Whisper model
#     whisper_model = whisper.load_model("./whisper-fine-tuned-stuttering-final")
    
#     # Step 2: Transcribe the stuttered speech to fluent text
#     fluent_text = transcribe_audio(whisper_model, input_audio)
    
#     # Step 3: Use F5TTS to generate fluent speech with the user's voice
#     output_path = clone_voice_and_generate_speech(
#         reference_audio=input_audio,
#         text_to_speak=fluent_text,
#         output_path=output_audio
#     )
    
#     return output_path, fluent_text

def process_stuttered_speech_one_step(input_audio, fine_tuned_model, output_audio="fluent_speech.wav"):
    """
    Process stuttered speech to fluent speech in one step.
    
    Args:
        input_audio: Path to the stuttered speech audio file
        fine_tuned_model: The fine-tuned Whisper model
        output_audio: Path to save the generated fluent speech
    
    Returns:
        Path to the generated audio file with fluent speech in the user's voice
    """
    # Use the optimized function that handles both transcription and voice cloning
    output_path, fluent_text = clone_voice_and_generate_speech_optimized(
        reference_audio=input_audio,
        fine_tuned_whisper_model=fine_tuned_model,
        output_path=output_audio
    )
    
    return output_path, fluent_text