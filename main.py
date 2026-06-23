import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =====================================
# 1. LOAD DATA
# =====================================

print("Loading data...")

# Chỉ lấy 100000 dòng để máy chạy nhanh
df = pd.read_csv(
    "yellow_tripdata_2019-01.csv",
    nrows=100000
)

print("Dataset shape:")
print(df.shape)

# =====================================
# 2. CONVERT DATETIME
# =====================================

df['pickup_time'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['dropoff_time'] = pd.to_datetime(df['tpep_dropoff_datetime'])

# =====================================
# 3. CREATE TRAVEL TIME
# =====================================

df['travel_time'] = (
    df['dropoff_time'] - df['pickup_time']
).dt.total_seconds() / 3600

# Xóa dữ liệu lỗi
df = df[df['travel_time'] > 0]

# =====================================
# 4. CALCULATE SPEED
# =====================================

df['avg_speed'] = (
    df['trip_distance'] / df['travel_time']
)

# Xóa tốc độ bất thường
df = df[
    (df['avg_speed'] > 0) &
    (df['avg_speed'] < 80)
]

# =====================================
# 5. CREATE CONGESTION LEVEL
# =====================================

def get_congestion(speed):

    if speed > 25:
        return "Low"

    elif speed > 15:
        return "Medium"

    else:
        return "High"


df['Congestion_Level'] = (
    df['avg_speed']
    .apply(get_congestion)
)

print("\nCongestion distribution:")
print(df['Congestion_Level'].value_counts())

# =====================================
# 6. FEATURE ENGINEERING
# =====================================

df['hour'] = df['pickup_time'].dt.hour
df['dayofweek'] = df['pickup_time'].dt.dayofweek
df['month'] = df['pickup_time'].dt.month

# =====================================
# 7. EDA
# =====================================

plt.figure(figsize=(10,5))
sns.countplot(x='hour', data=df)
plt.title("Trips by Hour")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(
    x='Congestion_Level',
    data=df
)
plt.title("Congestion Level")
plt.show()

# =====================================
# 8. SELECT FEATURES
# =====================================

features = [
    'hour',
    'dayofweek',
    'month',
    'passenger_count',
    'trip_distance',
    'PULocationID',
    'DOLocationID'
]

X = df[features]

# =====================================
# 9. ENCODE LABEL
# =====================================

encoder = LabelEncoder()

y = encoder.fit_transform(
    df['Congestion_Level']
)

# =====================================
# 10. SPLIT DATA
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# 11. TRAIN MODEL
# =====================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# =====================================
# 12. PREDICT
# =====================================

y_pred = model.predict(X_test)

# =====================================
# 13. EVALUATION
# =====================================

acc = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(acc)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

# =====================================
# 14. FEATURE IMPORTANCE
# =====================================

importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(8,5))

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance
)

plt.title("Feature Importance")
plt.show()

print("\nFinished!")