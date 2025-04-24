import os
import pandas as pd

# Clone the SEP-28k repository 
# Comment this out after data available
os.system("git clone https://github.com/apple/ml-stuttering-events-dataset.git")
os.chdir("ml-stuttering-events-dataset")

# Download the audio data
os.system("python3 download_audio.py --episodes SEP-28k_episodes.csv --wavs data/wavs")
os.system("python3 extract_clips.py --labels SEP-28k_labels.csv --wavs data/wavs --clips data/clips")

# Also get FluencyBank data
os.system("python3 download_audio.py --episodes fluencybank_episodes.csv --wavs data/wavs")
os.system("python3 extract_clips.py --labels fluencybank_labels.csv --wavs data/wavs --clips data/clips_fluencybank")

def prepare_fine_tuning_data():
    """
    Prepare the dataset for fine-tuning Whisper.
    
    Returns:
        A list of dictionaries with audio paths and transcriptions
    """
    # Load the labels
    sep_labels = pd.read_csv("SEP-28k_labels.csv")
    fluency_labels = pd.read_csv("fluencybank_labels.csv")
    
    # Combine datasets
    all_data = []
    
    # Process SEP-28k data
    for _, row in sep_labels.iterrows():
        # Only include clips with clear transcriptions
        if not pd.isna(row.get('transcription')):
            all_data.append({
                "audio": f"data/clips/{row['keycode']}_{row['start']}_{row['end']}.wav",
                "text": row['transcription']
            })
    
    # Process FluencyBank data
    for _, row in fluency_labels.iterrows():
        if not pd.isna(row.get('transcription')):
            all_data.append({
                "audio": f"data/clips_fluencybank/{row['keycode']}_{row['start']}_{row['end']}.wav",
                "text": row['transcription']
            })
    
    return all_data