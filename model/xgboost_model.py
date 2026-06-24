import pandas as pd
import os
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

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
# TARGET
# =====================================================

encoder = LabelEncoder()

df["congestion_label"] = encoder.fit_transform(
    df["congestion_level"]
)

# =====================================================
# FEATURES
# =====================================================
# Dùng các đặc trưng có ý nghĩa dự báo

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

print("Train Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# =====================================================
# MODEL
# =====================================================

print("\nTraining XGBoost...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss"
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
# EVALUATE
# =====================================================

acc = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(round(acc * 100, 2), "%")

report = classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_
)
print(report)

# =====================================
# ADDITIONAL METRICS
# =====================================

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

print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

# =====================================
# SAVE METRICS
# =====================================

results = pd.DataFrame({
    "Model": ["XGBoost"],
    "Accuracy": [acc],
    "Precision": [precision],
    "Recall": [recall],
    "F1": [f1]
})
results.to_csv(
    "results/xgboost_metrics.csv",
    index=False
)

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title("XGBoost Feature Importance")

plt.show()

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(
    model,
    "results/xgboost_model.pkl"
)

print("\nModel saved:")
print("results/xgboost_model.pkl")

print("\nPROGRAM FINISHED")