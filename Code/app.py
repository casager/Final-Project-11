import sounddevice as sd
import soundfile as sf
import tkinter as tk
from tkinter import ttk
import threading
import tempfile
import wave
import time

class StutteringAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stuttering Assistant")
        self.root.geometry("600x400")
        
        # Setup Whisper model
        self.whisper_model = setup_whisper("small")
        
        # Setup F5TTS
        setup_f5tts()
        
        # Initialize variables
        self.recording = False
        self.sample_rate = 16000
        self.frames = []
        
        # Create UI elements
        self.create_widgets()
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = ttk.Label(main_frame, text="Stuttering Assistant", font=("Arial", 18))
        title.pack(pady=10)
        
        # Record button
        self.record_button = ttk.Button(
            main_frame, 
            text="Start Recording", 
            command=self.toggle_recording
        )
        self.record_button.pack(pady=20)
        
        # Text display
        text_frame = ttk.LabelFrame(main_frame, text="Transcription")
        text_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.text_display = tk.Text(text_frame, height=10, wrap=tk.WORD)
        self.text_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.pack(pady=10)
    
    def toggle_recording(self):
        if not self.recording:
            # Start recording
            self.recording = True
            self.record_button.config(text="Stop Recording")
            self.status_label.config(text="Recording...")
            self.frames = []
            
            # Start recording in a separate thread
            threading.Thread(target=self.record_audio).start()
        else:
            # Stop recording
            self.recording = False
            self.record_button.config(text="Start Recording")
            self.status_label.config(text="Processing...")
            
            # Process the recorded audio
            threading.Thread(target=self.process_recording).start()
    
    def record_audio(self):
        """Record audio from the microphone."""
        def callback(indata, frames, time, status):
            if status:
                print(status)
            self.frames.append(indata.copy())
        
        with sd.InputStream(callback=callback, channels=1, samplerate=self.sample_rate):
            while self.recording:
                sd.sleep(100)
    
    def process_recording(self):
        """Process the recorded audio."""
        if not self.frames:
            self.status_label.config(text="No audio recorded.")
            return
        
        # Save the recorded audio to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.close()
        
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.frames))
        
        # Process the stuttered speech
        output_path, fluent_text = process_stuttered_speech(temp_file.name)
        
        # Update the UI
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(tk.END, fluent_text)
        
        # Play the processed audio
        if output_path:
            self.status_label.config(text="Playing processed speech...")
            data, fs = sf.read(output_path)
            sd.play(data, fs)
            sd.wait()
            self.status_label.config(text="Ready")
        else:
            self.status_label.config(text="Error processing speech.")

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = StutteringAssistantApp(root)
    root.mainloop()