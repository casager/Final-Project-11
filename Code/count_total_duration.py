import os
import pandas as pd

# Path to your CSVs (you can change this to your cleaned_csvs folder too)
csv_dir = os.path.expanduser('~/TimeStamped/cleaned_csvs')

total_seconds = 0.0
file_count = 0

for fname in os.listdir(csv_dir):
    if not fname.endswith('.csv'):
        continue

    path = os.path.join(csv_dir, fname)
    try:
        df = pd.read_csv(path)
        if not df.empty:
            last_wordend = df['wordend'].max()
            total_seconds += last_wordend
            file_count += 1
    except Exception as e:
        print(f"Error reading {fname}: {e}")

total_minutes = total_seconds / 60

print(f"\n📊 Processed {file_count} CSV files")
print(f"⏱️  Total audio duration: {total_seconds:.2f} seconds ({total_minutes:.2f} minutes)")
