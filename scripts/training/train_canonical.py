
from pathlib import Path
from hashlib import sha256
import csv

import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Where our recordings are stored
project = Path(__file__).resolve().parents[2]
dataset = project / "dataset"
output = project / "runtime" / "training"
output.mkdir(parents=True, exist_ok=True)

people = ["ahnaf_tahmid", "farhana_faruk", "sayed_bhuiyan"]
sounds = ["background", "clap", "hey", "whistle"]


def audio_features(file):
    audio, sample_rate = librosa.load(file, sr=16000, mono=False)

    # Some original ReSpeaker recordings have six channels.
    if audio.ndim == 2:
        if audio.shape[0] != 6:
            raise ValueError(f"Unexpected channel count: {file}")
        audio = audio[2]  # Third microphone channel

    if len(audio) < 16000 or not np.isfinite(audio).all():
        raise ValueError(f"Audio too short or invalid: {file}")

    # Same audio should not be counted twice.
    audio_bytes = np.round(np.clip(audio, -1, 1) * 32767).astype("<i2")
    audio_hash = sha256(audio_bytes.tobytes()).hexdigest()

    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)

    features = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1)
    ])

    return features, audio_hash


# Read all recordings
recordings = []
seen_audio = {}

for person in people:
    for sound in sounds:
        folder = dataset / person / sound

        for file in sorted(folder.glob("*.wav")):
            features, audio_hash = audio_features(file)

            if audio_hash in seen_audio:
                old_person, old_sound, old_file = seen_audio[audio_hash]

                if (old_person, old_sound) != (person, sound):
                    raise RuntimeError(
                        f"Conflicting labels: {old_file} and {file}"
                    )

                print(f"Skipping duplicate: {file.name}")
                continue

            seen_audio[audio_hash] = (person, sound, file)

            recordings.append({
                "file": str(file.relative_to(project)),
                "person": person,
                "sound": sound,
                "features": features
            })


print("\nRecordings used:", len(recordings))

for person in people:
    count = sum(r["person"] == person for r in recordings)
    print(person, ":", count)

active_people = [
    p for p in people
    if any(r["person"] == p for r in recordings)
]

if len(active_people) < 2:
    raise SystemExit("We need recordings from at least two people.")


def new_model():
    return RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )


# Train with one person's data, test with another person's data.
results = []

for test_person in active_people:
    train = [r for r in recordings if r["person"] != test_person]
    test = [r for r in recordings if r["person"] == test_person]

    if set(r["sound"] for r in train) != set(sounds):
        raise SystemExit("Training data is missing a sound class.")

    model = new_model()

    model.fit(
        np.array([r["features"] for r in train]),
        [r["sound"] for r in train]
    )

    predictions = model.predict(
        np.array([r["features"] for r in test])
    )

    actual = [r["sound"] for r in test]
    accuracy = accuracy_score(actual, predictions)

    print(f"\nTesting on {test_person}: {accuracy:.1%} accuracy")

    for recording, prediction in zip(test, predictions):
        results.append({
            "file": recording["file"],
            "person": recording["person"],
            "actual": recording["sound"],
            "predicted": prediction
        })


# Show final evaluation results
actual = [r["actual"] for r in results]
predicted = [r["predicted"] for r in results]

print("\nOverall accuracy:")
print(f"{accuracy_score(actual, predicted):.1%}")

print("\nPrecision, recall and F1 score:")
print(classification_report(
    actual,
    predicted,
    labels=sounds,
    zero_division=0
))

print("\nConfusion matrix:")
print("Rows = actual, columns = predicted")
print("Class order:", sounds)
print(confusion_matrix(actual, predicted, labels=sounds))


# Save test predictions
with (output / "rf_predictions.csv").open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["file", "person", "actual", "predicted"]
    )
    writer.writeheader()
    writer.writerows(results)


# Train one final model using all available recordings
final_model = new_model()

final_model.fit(
    np.array([r["features"] for r in recordings]),
    [r["sound"] for r in recordings]
)

joblib.dump({
    "model": final_model,
    "feature_columns": [f"feature_{i}" for i in range(1, 27)],
    "classes": final_model.classes_.tolist(),
    "sample_rate": 16000,
    "channels_for_six_channel_audio": 2
}, output / "whispryn_rf_v2.joblib")

print("\nNew model saved!")
print("File: runtime/training/whispryn_rf_v2.joblib")
print("Old model was not changed.")
