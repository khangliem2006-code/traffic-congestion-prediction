
# ============================================================
# TRAFFIC CONGESTION LEVEL PREDICTION USING TAXI TRIP DATA
# COMPLETE DATA CLEANING + FEATURE ENGINEERING + SQL ANALYSIS
# ============================================================

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. LOAD DATA
# ============================================================

FILE_PATH = "yellow_tripdata_2019-01.csv"

print("Loading dataset...")

df = pd.read_csv(
    FILE_PATH,
    low_memory=False
)

print("\nOriginal Shape:")
print(df.shape)

# ============================================================
# 2. DATA CLEANING
# ============================================================

print("\nCleaning data...")

# Datetime conversion

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

# Numeric conversion

numeric_cols = [
    'passenger_count',
    'trip_distance',
    'fare_amount',
    'tip_amount',
    'total_amount'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors='coerce'
    )

# Remove missing

df = df.dropna(
    subset=numeric_cols
)

# Remove invalid distance

df = df[
    df['trip_distance'] > 0
]

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

print("\nFeature Engineering...")

# Trip duration

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

# Pickup hour

df['pickup_hour'] = (
    df['tpep_pickup_datetime']
    .dt.hour
)

# Day of week

df['pickup_dayofweek'] = (
    df['tpep_pickup_datetime']
    .dt.dayofweek
)

# Weekend

df['is_weekend'] = (
    df['pickup_dayofweek'] >= 5
).astype(int)

# Rush hour

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

# Average speed

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

# ============================================================
# CREATE TARGET VARIABLE
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

print("\nFinal Shape:")
print(df.shape)

print("\nCongestion Level Distribution:")
print(df['congestion_level'].value_counts())

# ============================================================
# SAVE CLEAN DATA
# ============================================================

df.to_csv(
    "cleaned_taxi_data.csv",
    index=False
)

print("\nSaved: cleaned_taxi_data.csv")

# ============================================================
# SQLITE DATABASE
# ============================================================

print("\nCreating SQLite database...")

conn = sqlite3.connect(
    "traffic_congestion.db"
)

df.to_sql(
    "taxi_trips",
    conn,
    if_exists="replace",
    index=False
)

print("Database Created!")

# ============================================================
# SQL QUERY 1
# TOP BUSIEST HOURS
# ============================================================

query1 = """
SELECT
pickup_hour,
COUNT(*) AS total_trips
FROM taxi_trips
GROUP BY pickup_hour
ORDER BY total_trips DESC
"""

hourly_trip = pd.read_sql_query(
    query1,
    conn
)

print("\nTop Busy Hours:")
print(hourly_trip.head())

# ============================================================
# SQL QUERY 2
# AVG SPEED BY HOUR
# ============================================================

query2 = """
SELECT
pickup_hour,
ROUND(AVG(trip_speed),2) avg_speed
FROM taxi_trips
GROUP BY pickup_hour
ORDER BY pickup_hour
"""

hourly_speed = pd.read_sql_query(
    query2,
    conn
)

# ============================================================
# SQL QUERY 3
# MOST CONGESTED PICKUP LOCATIONS
# ============================================================

query3 = """
SELECT
PULocationID,
ROUND(AVG(trip_speed),2) avg_speed
FROM taxi_trips
GROUP BY PULocationID
ORDER BY avg_speed ASC
LIMIT 10
"""

congested_pickup = pd.read_sql_query(
    query3,
    conn
)

print("\nMost Congested Pickup Areas:")
print(congested_pickup)

# ============================================================
# SQL QUERY 4
# CONGESTION DISTRIBUTION
# ============================================================

query4 = """
SELECT
congestion_level,
COUNT(*) total
FROM taxi_trips
GROUP BY congestion_level
"""

congestion_dist = pd.read_sql_query(
    query4,
    conn
)

print("\nCongestion Distribution:")
print(congestion_dist)

# ============================================================
# VISUALIZATION 1
# CONGESTION LEVEL DISTRIBUTION
# ============================================================

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x='congestion_level'
)

plt.title(
    "Congestion Level Distribution"
)

plt.show()

# ============================================================
# VISUALIZATION 2
# AVG SPEED BY HOUR
# ============================================================

plt.figure(figsize=(12,6))

plt.plot(
    hourly_speed['pickup_hour'],
    hourly_speed['avg_speed'],
    marker='o'
)

plt.title(
    "Average Speed by Hour"
)

plt.xlabel("Hour")

plt.ylabel("Average Speed (mph)")

plt.grid(True)

plt.show()

# ============================================================
# VISUALIZATION 3
# CORRELATION HEATMAP
# ============================================================

heatmap_features = [
    'passenger_count',
    'trip_distance',
    'trip_duration',
    'trip_speed',
    'fare_amount',
    'tip_amount',
    'total_amount'
]

corr = df[
    heatmap_features
].corr()

plt.figure(figsize=(10,8))

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title(
    "Correlation Heatmap"
)

plt.show()

# ============================================================
# VISUALIZATION 4
# TOP 10 CONGESTED PICKUP LOCATIONS
# ============================================================

plt.figure(figsize=(12,6))

sns.barplot(
    data=congested_pickup,
    x='PULocationID',
    y='avg_speed'
)

plt.title(
    "Top 10 Most Congested Pickup Locations"
)

plt.ylabel(
    "Average Speed"
)

plt.show()

# ============================================================
# CLOSE DATABASE
# ============================================================

conn.close()

print("\nAnalysis Completed Successfully!")

