import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =====================================================
# CREATE RESULTS FOLDER
# =====================================================

os.makedirs(
    "results",
    exist_ok=True
)

# =====================================================
# LOAD DATA
# =====================================================

print("Loading dataset...")

df = pd.read_csv(
    "cleaned_taxi_data.csv"
)

print("Dataset Shape:", df.shape)

# =====================================================
# ENCODE TARGET
# =====================================================

encoder = LabelEncoder()

df["congestion_label"] = encoder.fit_transform(
    df["congestion_level"]
)

print("\nClass Mapping:")

for i, cls in enumerate(encoder.classes_):
    print(i, "=", cls)

# =====================================================
# FEATURES
# =====================================================
# Dùng các feature có trước khi chuyến đi kết thúc
# để tránh Data Leakage

features = [
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend",
    "is_rush_hour",
    "passenger_count",
    "PULocationID",
    "DOLocationID"
]

X = df[features]

y = df["congestion_label"]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape :", X_test.shape)

# =====================================================
# MODEL
# =====================================================

print("\nTraining HistGradientBoosting...")

model = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("Training Completed!")

# =====================================================
# PREDICTION
# =====================================================

y_pred = model.predict(
    X_test
)

# =====================================================
# ACCURACY
# =====================================================

acc = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(round(acc * 100, 2), "%")

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_
)

print("\nClassification Report:\n")
print(report)

# Save report

with open(
    "results/hist_gradient_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("Accuracy:\n")
    f.write(str(acc))
    f.write("\n\n")
    f.write(report)

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

# Save CSV

cm_df = pd.DataFrame(cm)

cm_df.to_csv(
    "results/hist_gradient_confusion_matrix.csv",
    index=False
)

# =====================================================
# CONFUSION MATRIX IMAGE
# =====================================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title(
    "HistGradientBoosting Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.savefig(
    "results/hist_gradient_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# SAVE MODEL
# =====================================================

import joblib

joblib.dump(
    model,
    "results/hist_gradient_model.pkl"
)

# =====================================================
# FINISH
# =====================================================

print("\nFiles saved successfully:")

print(
    "results/hist_gradient_report.txt"
)

print(
    "results/hist_gradient_confusion_matrix.csv"
)

print(
    "results/hist_gradient_confusion_matrix.png"
)

print(
    "results/hist_gradient_model.pkl"
)

print("\nPROGRAM FINISHED")