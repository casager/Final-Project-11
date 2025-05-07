import os
import pandas as pd
from pydub import AudioSegment
from collections import defaultdict
import re

# Paths
csv_dir = os.path.expanduser('~/TimeStamped/csvs')
wav_dir = os.path.expanduser('~/TimeStamped-wav')
out_csv_dir = os.path.expanduser('~/TimeStamped/combined_csvs')
out_wav_dir = os.path.expanduser('~/TimeStamped-wav/combined_wavs')

# Create output directories
os.makedirs(out_csv_dir, exist_ok=True)
os.makedirs(out_wav_dir, exist_ok=True)

# Regex to extract prefix and index
pattern = re.compile(r"^([a-zA-Z0-9]+)_(\d+)\.csv$")

# Group files by prefix
groups = defaultdict(list)

for fname in os.listdir(csv_dir):
    match = pattern.match(fname)
    if match:
        prefix, idx = match.groups()
        groups[prefix].append((int(idx), fname))

# Process each group
for prefix, files in groups.items():
    files.sort()  # Sort by index
    for i in range(0, len(files), 3):
        group = files[i:i+3]
        if len(group) < 3:
            continue  # Skip incomplete groups

        combined_csv = pd.DataFrame()
        combined_audio = AudioSegment.empty()
        total_offset = 0.0
        out_base = f"{prefix}_{str(i).zfill(3)}_combined"

        for idx, fname in group:
            csv_path = os.path.join(csv_dir, fname)
            wav_name = fname.replace('.csv', '.wav')
            wav_path = os.path.join(wav_dir, wav_name)

            # Load and adjust CSV timestamps
            df = pd.read_csv(csv_path)
            df['wordstart'] += total_offset
            df['wordend'] += total_offset
            combined_csv = pd.concat([combined_csv, df], ignore_index=True)

            # Load audio
            audio = AudioSegment.from_wav(wav_path)
            combined_audio += audio
            total_offset += len(audio) / 1000.0  # in seconds

        # Save combined CSV and WAV
        combined_csv_path = os.path.join(out_csv_dir, out_base + ".csv")
        combined_wav_path = os.path.join(out_wav_dir, out_base + ".wav")

        combined_csv.to_csv(combined_csv_path, index=False)
        combined_audio.export(combined_wav_path, format="wav")

        print(f"Saved: {combined_csv_path}, {combined_wav_path}")
