
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("whispryn_features_test.csv")

X = df.drop(columns=["filename", "label"])
y = df["label"]

cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

labels = ["clap", "whistle", "hey", "background"]

all_true = []
all_pred = []
all_filenames = []

print("WHISPRYN - Initial Model Experiment")
print("Total recordings:", len(df))

for fold, (train_index, test_index) in enumerate(cv.split(X, y), 1):

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    all_true.extend(y_test.tolist())
    all_pred.extend(predictions.tolist())
    all_filenames.extend(df.iloc[test_index]["filename"].tolist())

    print(f"\nFold {fold} accuracy: {accuracy:.2%}")

print("\nOverall results:")
print(classification_report(
    all_true, all_pred, labels=labels, zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(all_true, all_pred, labels=labels))

print("\nIncorrect predictions:")

for filename, actual, predicted in zip(
    all_filenames, all_true, all_pred
):
    if actual != predicted:
        print(f"File: {filename} | Actual: {actual} | Predicted: {predicted}")
