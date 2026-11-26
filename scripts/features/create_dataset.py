from pathlib import Path
import numpy as np
import pandas as pd
import librosa

DATASET_DIR = Path("processed_audio")
OUTPUT_CSV = "whispryn_dataset.csv"

CLASSES = ["clap", "whistle", "hey", "background"]
SAMPLE_RATE = 16000
N_MFCC = 13

rows = []

for label in CLASSES:
    folder = DATASET_DIR / label
    wav_files = sorted(folder.glob("*.wav"))

    print(f"\nProcessing {label}: {len(wav_files)} files")

    for file_path in wav_files:
        audio, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC
        )

        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        features = {}

        for i in range(N_MFCC):
            features[f"mfcc_mean_{i+1}"] = float(mfcc_mean[i])
            features[f"mfcc_std_{i+1}"] = float(mfcc_std[i])

        features["label"] = label
        features["file"] = str(file_path)

        rows.append(features)

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print("\nCSV dataset created successfully!")
print(f"Total recordings: {len(df)}")
print(f"Features per recording: {N_MFCC * 2}")
print("\nClass distribution:")
print(df["label"].value_counts())
print(f"\nSaved as: {OUTPUT_CSV}")