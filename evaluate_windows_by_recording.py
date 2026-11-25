import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("whispryn_features_windows.csv")

feature_columns = [f"feature_{i}" for i in range(1, 27)]

labels_order = ["clap", "whistle", "hey", "background"]

experiments = [
    ("farhana_faruk", "ahnaf_ahmed"),
    ("ahnaf_ahmed", "farhana_faruk")
]

for train_person, test_person in experiments:

    train_df = df[df["person"] == train_person]
    test_df = df[df["person"] == test_person].copy()

    X_train = train_df[feature_columns]
    y_train = train_df["label"]

    X_test = test_df[feature_columns]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    test_df["window_prediction"] = model.predict(X_test)

    recording_results = (
        test_df.groupby(["person", "filename", "label"])["window_prediction"]
        .agg(lambda s: s.mode().iloc[0])
        .reset_index(name="predicted")
    )

    y_true = recording_results["label"]
    y_pred = recording_results["predicted"]

    print("\n" + "=" * 60)
    print(f"Training on: {train_person}")
    print(f"Testing on:  {test_person}")
    print("=" * 60)

    print("Testing recordings:", len(recording_results))

    print(
        f"Recording-level accuracy: "
        f"{accuracy_score(y_true, y_pred):.2%}"
    )

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=labels_order,
        zero_division=0
    ))

    print("Confusion matrix:")
    print(confusion_matrix(
        y_true,
        y_pred,
        labels=labels_order
    ))

    print("\nIndividual recording predictions:")
    print(
        recording_results[
            ["filename", "label", "predicted"]
        ].to_string(index=False)
    )