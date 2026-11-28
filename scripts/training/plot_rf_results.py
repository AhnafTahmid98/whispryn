from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
# WHISPRYN Random Forest v2 results
# -----------------------------
overall_accuracy = 93.8

people = ["ahnaf_tahmid", "farhana_faruk"]
people_accuracy = [100.0, 89.6]

classes = ["background", "clap", "hey", "whistle"]

precision = [1.00, 0.79, 1.00, 1.00]
recall = [1.00, 1.00, 0.89, 0.84]
f1_score = [1.00, 0.88, 0.94, 0.91]

confusion_matrix = np.array([
    [23, 0, 0, 0],
    [0, 19, 0, 0],
    [0, 2, 17, 0],
    [0, 3, 0, 16],
])

output_dir = Path("reports/graphs")
output_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------
# 1. Person-wise accuracy graph
# -----------------------------
plt.figure(figsize=(8, 5))
bars = plt.bar(people, people_accuracy)

for bar, value in zip(bars, people_accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.1f}%",
        ha="center"
    )

plt.ylim(0, 110)
plt.ylabel("Accuracy (%)")
plt.title(f"WHISPRYN Random Forest v2\nCross-person Accuracy (Overall: {overall_accuracy:.1f}%)")
plt.tight_layout()
plt.savefig(output_dir / "rf_v2_person_accuracy.png", dpi=200)
plt.close()


# -----------------------------
# 2. Class metrics graph
# -----------------------------
x = np.arange(len(classes))
width = 0.25

plt.figure(figsize=(10, 5))
plt.bar(x - width, precision, width, label="Precision")
plt.bar(x, recall, width, label="Recall")
plt.bar(x + width, f1_score, width, label="F1-score")

plt.xticks(x, classes)
plt.ylim(0, 1.1)
plt.ylabel("Score")
plt.title("WHISPRYN Random Forest v2\nPrecision, Recall and F1-score")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "rf_v2_class_metrics.png", dpi=200)
plt.close()


# -----------------------------
# 3. Confusion matrix graph
# -----------------------------
plt.figure(figsize=(7, 6))
plt.imshow(confusion_matrix, interpolation="nearest")
plt.title("WHISPRYN Random Forest v2\nConfusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        plt.text(j, i, str(confusion_matrix[i, j]), ha="center", va="center")

plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(output_dir / "rf_v2_confusion_matrix.png", dpi=200)
plt.close()

print("Graphs saved in:", output_dir)
print("- rf_v2_person_accuracy.png")
print("- rf_v2_class_metrics.png")
print("- rf_v2_confusion_matrix.png")
