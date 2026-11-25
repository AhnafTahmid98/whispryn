import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("whispryn_features_windows.csv")

feature_columns = [f"feature_{i}" for i in range(1, 27)]

X = df[feature_columns]
y = df["label"]
people = df["person"]

experiments = [
    ("farhana_faruk", "ahnaf_ahmed"),
    ("ahnaf_ahmed", "farhana_faruk")
]

for train_person, test_person in experiments:

    print("\n" + "=" * 60)
    print(f"Training on: {train_person}")
    print(f"Testing on:  {test_person}")
    print("=" * 60)

    train_mask = people == train_person
    test_mask = people == test_person

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nTraining windows:", len(X_train))
    print("Testing windows:", len(X_test))

    print(f"\nAccuracy: {accuracy_score(y_test, predictions):.2%}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    print("Confusion matrix:")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=["clap", "whistle", "hey", "background"]
    ))