import numpy as np
import librosa
import joblib

# Load trained model
saved_data = joblib.load("whispryn_model.joblib")

model = saved_data["model"]
feature_columns = saved_data["feature_columns"]

# Select audio file
AUDIO_FILE = "processed_audio/background/background_01.wav"
# Load audio
audio, sr = librosa.load(
    AUDIO_FILE,
    sr=16000,
    mono=True
)

# Extract 13 MFCC coefficients
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=13
)

# Calculate mean and standard deviation
mfcc_mean = np.mean(mfcc, axis=1)
mfcc_std = np.std(mfcc, axis=1)

# Prepare features
features = {}

for i in range(13):
    features[f"mfcc_mean_{i+1}"] = float(mfcc_mean[i])
    features[f"mfcc_std_{i+1}"] = float(mfcc_std[i])

# Match training feature order
X = np.array([
    [features[name] for name in feature_columns]
])

# Predict sound
prediction = model.predict(X)[0]
probabilities = model.predict_proba(X)[0]

print("\nPredicted sound:", prediction)

print("\nPrediction probabilities:")

for label, probability in zip(model.classes_, probabilities):
    print(f"{label}: {probability * 100:.2f}%")
