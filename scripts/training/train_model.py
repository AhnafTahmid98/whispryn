import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load CSV dataset
df = pd.read_csv("whispryn_dataset.csv")

# Separate features and labels
X = df.drop(columns=["label", "file"])
y = df["label"]

# Define classifier
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

# Test using Leave-One-Out cross-validation
loo = LeaveOneOut()

print("Testing model with Leave-One-Out cross-validation...")

predictions = cross_val_predict(model, X, y, cv=loo)

# Evaluation
accuracy = accuracy_score(y, predictions)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y,
    predictions,
    labels=["clap", "whistle", "hey", "background"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y,
    predictions,
    labels=["clap", "whistle", "hey", "background"]
))

# Train final model using all recordings
model.fit(X, y)

# Save model and feature information
model_data = {
    "model": model,
    "feature_columns": X.columns.tolist(),
    "classes": model.classes_.tolist()
}

joblib.dump(model_data, "whispryn_model.joblib")

print("\nTraining completed!")
print("Model saved as whispryn_model.joblib")
