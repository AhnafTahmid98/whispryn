import librosa
import numpy as np
import pandas as pd
from pathlib import Path

DATASET_ROOT = Path("dataset_mono_test")

PEOPLE = ["farhana_faruk", "ahnaf_ahmed"]
CLASSES = ["clap", "whistle", "hey", "background"]

OUTPUT_FILE = "whispryn_features_windows.csv"

SAMPLE_RATE = 16000
WINDOW_SECONDS = 1
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS

feature_rows = []

for person in PEOPLE:
    print(f"\n===== Processing {person} =====")

    for class_name in CLASSES:
        folder = DATASET_ROOT / person / class_name
        audio_files = sorted(folder.glob("*.wav"))

        for audio_file in audio_files:
            audio, sr = librosa.load(
                audio_file,
                sr=SAMPLE_RATE,
                mono=True
            )

            for start in range(0, len(audio), WINDOW_SIZE):
                segment = audio[start:start + WINDOW_SIZE]

                if len(segment) < WINDOW_SIZE:
                    continue

                if np.max(np.abs(segment)) < 0.005:
                    continue

                mfcc = librosa.feature.mfcc(
                    y=segment,
                    sr=sr,
                    n_mfcc=13
                )

                features = np.concatenate([
                    np.mean(mfcc, axis=1),
                    np.std(mfcc, axis=1)
                ])

                row = {
                    "filename": audio_file.name,
                    "person": person,
                    "label": class_name,
                    "window_start": start / sr
                }

                for i, value in enumerate(features):
                    row[f"feature_{i + 1}"] = value

                feature_rows.append(row)

df = pd.DataFrame(feature_rows)
df.to_csv(OUTPUT_FILE, index=False)

print("\n===== Window Feature Extraction Completed =====")
print("Total windows:", len(df))
print("Total columns:", len(df.columns))
print("\nWindows per person and class:")
print(df.groupby(["person", "label"]).size())
print("\nSaved to:", OUTPUT_FILE)