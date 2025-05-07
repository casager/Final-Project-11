import os
import pandas as pd

# Paths
input_dir = os.path.expanduser('~/TimeStamped/combined_csvs')
output_dir = os.path.expanduser('~/TimeStamped/cleaned_csvs')
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.endswith('.csv'):
        continue

    df = pd.read_csv(os.path.join(input_dir, fname))

    # Cleaned rows go here
    cleaned_rows = []
    skip_rows = []

    pending_start = None

    for idx, row in df.iterrows():
        if row['rp'] > 0 or row['pw'] > 0:
            # Stutter row – remember its start time and skip it
            if pending_start is None:
                pending_start = row['wordstart']
            skip_rows.append(idx)
            continue

        if pending_start is not None:
            # Adjust start time of fluent word
            row['wordstart'] = pending_start
            pending_start = None

        cleaned_rows.append(row)

    cleaned_df = pd.DataFrame(cleaned_rows)
    out_path = os.path.join(output_dir, fname)
    cleaned_df.to_csv(out_path, index=False)

    print(f"Saved cleaned file: {out_path}")
