import pandas as pd
import os
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

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

# =====================================================
# FEATURES
# =====================================================

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
# SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# SCALE DATA
# =====================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)

# =====================================================
# MLP MODEL
# =====================================================

print("\nTraining MLP Neural Network...")

model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=200,
    random_state=42,
    early_stopping=True
)

model.fit(
    X_train,
    y_train
)

print("Training Completed!")

# =====================================================
# PREDICT
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
    "results/mlp_report.txt",
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

# =====================================================
# PLOT CONFUSION MATRIX
# =====================================================

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "MLP Neural Network Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.show()

# =====================================================
# LOSS CURVE
# =====================================================

plt.figure(figsize=(10,6))

plt.plot(
    model.loss_curve_
)

plt.title(
    "MLP Training Loss Curve"
)

plt.xlabel(
    "Iterations"
)

plt.ylabel(
    "Loss"
)

plt.grid(True)

plt.show()

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(
    model,
    "results/mlp_model.pkl"
)

joblib.dump(
    scaler,
    "results/mlp_scaler.pkl"
)

print("\nModel Saved!")

print("\nPROGRAM FINISHED")