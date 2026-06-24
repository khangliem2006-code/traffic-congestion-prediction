# ============================================================
# TRAFFIC CONGESTION PREDICTION
# GRADIENT BOOSTING CLASSIFIER
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# LOAD DATA
# ============================================================

print("Loading cleaned dataset...")

df = pd.read_csv("cleaned_taxi_data.csv")

print("Dataset Shape:", df.shape)

# ============================================================
# ENCODE TARGET
# ============================================================

encoder = LabelEncoder()

df['congestion_label'] = encoder.fit_transform(
    df['congestion_level']
)

print("\nClass Mapping:")

for i, c in enumerate(encoder.classes_):
    print(i, "=", c)

# ============================================================
# FEATURES
# ============================================================

features = [
    'passenger_count',
    'trip_distance',
    'trip_duration',
    'pickup_hour',
    'pickup_dayofweek',
    'is_weekend',
    'is_rush_hour',
    'fare_amount',
    'tip_amount',
    'PULocationID',
    'DOLocationID'
]

X = df[features]

y = df['congestion_label']

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# ============================================================
# GRADIENT BOOSTING MODEL
# ============================================================

print("\nTraining Gradient Boosting...")

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

gb.fit(
    X_train,
    y_train
)

print("Training Complete!")

# ============================================================
# PREDICTION
# ============================================================

y_pred = gb.predict(X_test)

# ============================================================
# EVALUATION
# ============================================================

acc = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(round(acc * 100, 2), "%")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)
# ============================================================
# ADDITIONAL METRICS
# ============================================================

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\nPrecision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

# ============================================================
# SAVE METRICS
# ============================================================

import os

os.makedirs(
    "results",
    exist_ok=True
)

results = pd.DataFrame({
    "Model": ["Gradient Boosting"],
    "Accuracy": [acc],
    "Precision": [precision],
    "Recall": [recall],
    "F1": [f1]
})

results.to_csv(
    "results/gradient_boosting_metrics.csv",
    index=False
)

print("\nMetrics saved!")

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens'
)

plt.title(
    "Gradient Boosting Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    'Feature': features,
    'Importance': gb.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance,
    x='Importance',
    y='Feature'
)

plt.title(
    "Gradient Boosting Feature Importance"
)

plt.show()

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    gb,
    "gradient_boosting_congestion.pkl"
)

print("\nModel Saved:")
print("gradient_boosting_congestion.pkl")