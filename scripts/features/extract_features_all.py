import librosa
import numpy as np
import pandas as pd
from pathlib import Path

DATASET_ROOT = Path("dataset_mono_test")
PEOPLE = ["farhana_faruk", "ahnaf_ahmed"]
CLASSES = ["clap", "whistle", "hey", "background"]
OUTPUT_FILE = "whispryn_features_all.csv"

feature_rows = []

for person in PEOPLE:
    print(f"\n===== Processing {person} =====")

    for class_name in CLASSES:
        folder = DATASET_ROOT / person / class_name
        audio_files = sorted(folder.glob("*.wav"))

        print(f"\n{class_name}: {len(audio_files)} files")

        for audio_file in audio_files:
            audio, sample_rate = librosa.load(
                audio_file,
                sr=16000,
                mono=True
            )

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=13
            )

            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            features = np.concatenate([mfcc_mean, mfcc_std])

            row = {
                "filename": audio_file.name,
                "person": person,
                "label": class_name
            }

            for i, value in enumerate(features):
                row[f"feature_{i + 1}"] = value

            feature_rows.append(row)
            print(f"Processed: {audio_file.name}")

df = pd.DataFrame(feature_rows)
df.to_csv(OUTPUT_FILE, index=False)

print("\n===== Feature Extraction Completed =====")
print("Total recordings:", len(df))
print("Total columns:", len(df.columns))
print("\nRecordings per person and class:")
print(df.groupby(["person", "label"]).size())
print("\nSaved to:", OUTPUT_FILE)