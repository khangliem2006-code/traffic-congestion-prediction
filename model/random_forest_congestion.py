# ============================================================
# TRAFFIC CONGESTION LEVEL PREDICTION USING TAXI TRIP DATA
# RANDOM FOREST CLASSIFICATION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(
    "cleaned_taxi_data.csv",
    low_memory=False
)

print("Original Shape:", df.shape)

# ============================================================
# 2. DATA CLEANING
# ============================================================

print("\nCleaning data...")

# Convert datetime

df['tpep_pickup_datetime'] = pd.to_datetime(
    df['tpep_pickup_datetime'],
    errors='coerce'
)

df['tpep_dropoff_datetime'] = pd.to_datetime(
    df['tpep_dropoff_datetime'],
    errors='coerce'
)

# Remove invalid datetime

df = df.dropna(
    subset=[
        'tpep_pickup_datetime',
        'tpep_dropoff_datetime'
    ]
)

# Convert numeric columns

numeric_cols = [
    'passenger_count',
    'trip_distance',
    'fare_amount',
    'tip_amount',
    'total_amount',
    'PULocationID',
    'DOLocationID'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors='coerce'
    )

# Remove missing

df = df.dropna(subset=numeric_cols)

# Remove invalid trip distance

df = df[
    df['trip_distance'] > 0
]

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

print("\nFeature Engineering...")

# Trip Duration

df['trip_duration'] = (
    df['tpep_dropoff_datetime']
    -
    df['tpep_pickup_datetime']
).dt.total_seconds() / 60

# Keep realistic duration

df = df[
    (df['trip_duration'] > 0)
    &
    (df['trip_duration'] < 300)
]

# Pickup Hour

df['pickup_hour'] = (
    df['tpep_pickup_datetime']
    .dt.hour
)

# Day of Week

df['pickup_dayofweek'] = (
    df['tpep_pickup_datetime']
    .dt.dayofweek
)

# Weekend

df['is_weekend'] = (
    df['pickup_dayofweek'] >= 5
).astype(int)

# Rush Hour

df['is_rush_hour'] = (
    (
        (df['pickup_hour'] >= 7)
        &
        (df['pickup_hour'] <= 9)
    )
    |
    (
        (df['pickup_hour'] >= 16)
        &
        (df['pickup_hour'] <= 19)
    )
).astype(int)

# Trip Speed (mph)

df['trip_speed'] = (
    df['trip_distance']
    /
    (df['trip_duration'] / 60)
)

# Remove unrealistic speed

df = df[
    (df['trip_speed'] > 0)
    &
    (df['trip_speed'] < 80)
]

print("Shape after cleaning:", df.shape)

# ============================================================
# 4. CREATE TARGET VARIABLE
# ============================================================

def congestion_level(speed):

    if speed < 10:
        return "High"

    elif speed < 20:
        return "Medium"

    else:
        return "Low"

df['congestion_level'] = (
    df['trip_speed']
    .apply(congestion_level)
)

print("\nCongestion Distribution:")
print(df['congestion_level'].value_counts())

# ============================================================
# 5. LABEL ENCODING
# ============================================================

encoder = LabelEncoder()

df['congestion_label'] = encoder.fit_transform(
    df['congestion_level']
)

print("\nClass Mapping:")
for i, cls in enumerate(encoder.classes_):
    print(i, "=", cls)

# ============================================================
# 6. SELECT FEATURES
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
# 7. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Set:", X_train.shape)
print("Testing Set :", X_test.shape)

# ============================================================
# 8. TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_train,
    y_train
)

print("Training Completed!")

# ============================================================
# 9. PREDICTION
# ============================================================

y_pred = rf.predict(X_test)

# ============================================================
# 10. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

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

import os
os.makedirs("results", exist_ok=True)

results = pd.DataFrame({
    "Model": ["Random Forest"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1": [f1]
})

results.to_csv(
    "results/random_forest_metrics.csv",
    index=False
)

# ============================================================
# 11. CONFUSION MATRIX
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
    cmap='Blues'
)

plt.title(
    'Confusion Matrix'
)

plt.xlabel(
    'Predicted'
)

plt.ylabel(
    'Actual'
)

plt.show()

# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
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
    'Random Forest Feature Importance'
)

plt.show()

# ============================================================
# 13. SAVE MODEL
# ============================================================

joblib.dump(
    rf,
    "random_forest_congestion.pkl"
)

print("\nModel saved:")
print("random_forest_congestion.pkl")

# ============================================================
# 14. SAVE CLEAN DATA
# ============================================================

df.to_csv(
    "cleaned_taxi_data.csv",
    index=False
)

print("\nCleaned dataset saved:")
print("cleaned_taxi_data.csv")

print("\nDONE!")

