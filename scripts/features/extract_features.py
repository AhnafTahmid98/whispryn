
import librosa
import numpy as np
import pandas as pd
from pathlib import Path

DATASET = Path("dataset_mono_test/farhana_faruk")
OUTPUT_FILE = "whispryn_features_test.csv"

CLASSES = ["clap", "whistle", "hey", "background"]

feature_rows = []

for class_name in CLASSES:
    folder = DATASET / class_name
    audio_files = sorted(folder.glob("*.wav"))

    print(f"\nProcessing {class_name}: {len(audio_files)} files")

    for audio_file in audio_files:
        audio, sample_rate = librosa.load(
            audio_file,
            sr=16000,
            mono=True
        )

        # Extract 13 MFCC coefficients
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=13
        )

        # Calculate mean and standard deviation
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        features = np.concatenate([mfcc_mean, mfcc_std])

        row = {
            "filename": audio_file.name,
            "label": class_name
        }

        for i, value in enumerate(features):
            row[f"feature_{i + 1}"] = value

        feature_rows.append(row)

        print(f"Processed: {audio_file.name}")

# Save all extracted features
df = pd.DataFrame(feature_rows)
df.to_csv(OUTPUT_FILE, index=False)

print("\nFeature extraction completed!")
print("Total recordings:", len(df))
print("Total columns:", len(df.columns))
print("Saved to:", OUTPUT_FILE)
