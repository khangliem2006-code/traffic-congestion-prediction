# =====================================================
# TRAFFIC CONGESTION LEVEL PREDICTION USING TAXI DATA
# FIXED VERSION FOR yellow_tripdata_2019-01.csv
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# LOAD DATA
# -----------------------------

file_path = "yellow_tripdata_2019-01.csv"

print("Loading dataset...")

df = pd.read_csv(
    file_path,
    low_memory=False
)

print("Original Shape:", df.shape)

# -----------------------------
# DATETIME CONVERSION
# -----------------------------

df['tpep_pickup_datetime'] = pd.to_datetime(
    df['tpep_pickup_datetime'],
    errors='coerce'
)

df['tpep_dropoff_datetime'] = pd.to_datetime(
    df['tpep_dropoff_datetime'],
    errors='coerce'
)

# Xóa các dòng lỗi thời gian

df = df.dropna(
    subset=[
        'tpep_pickup_datetime',
        'tpep_dropoff_datetime'
    ]
)

print("After datetime cleaning:", df.shape)

# -----------------------------
# CREATE FEATURES
# -----------------------------

df['pickup_hour'] = (
    df['tpep_pickup_datetime']
    .dt.hour
)

df['pickup_day'] = (
    df['tpep_pickup_datetime']
    .dt.day_name()
)

# thời lượng (phút)

df['trip_duration'] = (
    df['tpep_dropoff_datetime']
    -
    df['tpep_pickup_datetime']
).dt.total_seconds() / 60

# chỉ giữ chuyến đi hợp lệ

df = df[
    (df['trip_duration'] > 0)
]

# khoảng cách hợp lệ

df = df[
    (df['trip_distance'] > 0)
]

# tốc độ trung bình (mph)

df['trip_speed'] = (
    df['trip_distance']
    /
    (df['trip_duration'] / 60)
)

# loại tốc độ bất thường

df = df[
    (df['trip_speed'] > 0)
    &
    (df['trip_speed'] < 80)
]

print("After feature engineering:", df.shape)

# -----------------------------
# CREATE TARGET VARIABLE
# CONGESTION LEVEL
# -----------------------------

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

print("\nCongestion Level Distribution:")
print(df['congestion_level'].value_counts())

# -----------------------------
# VISUALIZATION 1
# CONGESTION LEVEL DISTRIBUTION
# -----------------------------

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x='congestion_level'
)

plt.title("Congestion Level Distribution")

plt.show()

# -----------------------------
# VISUALIZATION 2
# CONGESTION BY HOUR
# -----------------------------

hourly = (
    df.groupby('pickup_hour')
    ['trip_speed']
    .mean()
)

plt.figure(figsize=(12,6))

sns.lineplot(
    x=hourly.index,
    y=hourly.values,
    marker='o'
)

plt.title(
    "Average Speed by Hour"
)

plt.xlabel("Hour")
plt.ylabel("Average Speed (mph)")

plt.grid(True)

plt.show()

# -----------------------------
# VISUALIZATION 3
# CONGESTION BY DAY
# -----------------------------

order = [
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday'
]

daily = (
    df.groupby('pickup_day')
    ['trip_speed']
    .mean()
    .reindex(order)
)

plt.figure(figsize=(10,6))

sns.barplot(
    x=daily.index,
    y=daily.values
)

plt.title(
    "Average Speed by Day"
)

plt.xticks(rotation=45)

plt.show()

# -----------------------------
# VISUALIZATION 4
# HEATMAP
# -----------------------------

features = [
    'passenger_count',
    'trip_distance',
    'trip_duration',
    'trip_speed',
    'fare_amount',
    'tip_amount',
    'total_amount'
]

corr = df[features].corr()

plt.figure(figsize=(10,8))

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title("Correlation Heatmap")

plt.show()

# -----------------------------
# VISUALIZATION 5
# TOP PICKUP LOCATIONS
# -----------------------------

top_pickup = (
    df.groupby('PULocationID')
    ['trip_speed']
    .mean()
    .sort_values()
    .head(10)
)

plt.figure(figsize=(12,6))

sns.barplot(
    x=top_pickup.index.astype(str),
    y=top_pickup.values
)

plt.title(
    "Top 10 Most Congested Pickup Locations"
)

plt.ylabel("Average Speed")

plt.show()

print("\nFinal Dataset Shape:", df.shape)
print("\nAnalysis Completed Successfully!")