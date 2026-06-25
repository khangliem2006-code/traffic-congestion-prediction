import pandas as pd
import matplotlib.pyplot as plt
import os

# =====================================================
# LOAD METRICS FILES
# =====================================================

rf = pd.read_csv(
    "results/random_forest_metrics.csv"
)

gb = pd.read_csv(
    "results/gradient_boosting_metrics.csv"
)

hgb = pd.read_csv(
    "results/hist_gradient_metrics.csv"
)

xgb = pd.read_csv(
    "results/xgboost_metrics.csv"
)

mlp = pd.read_csv(
    "results/mlp_metrics.csv"
)

# =====================================================
# COMBINE RESULTS
# =====================================================

comparison = pd.concat([
    rf,
    gb,
    hgb,
    xgb,
    mlp
])

# =====================================================
# SORT BY ACCURACY
# =====================================================

comparison = comparison.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n========== MODEL COMPARISON ==========\n")
print(comparison)

# =====================================================
# SAVE COMPARISON TABLE
# =====================================================

comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nComparison file saved!")

# =====================================================
# ACCURACY COMPARISON
# =====================================================

plt.figure(figsize=(10,6))

plt.bar(
    comparison["Model"],
    comparison["Accuracy"]
)

plt.title(
    "Accuracy Comparison of Models"
)

plt.ylabel(
    "Accuracy"
)

plt.grid(True)

plt.show()

# =====================================================
# MULTI-METRIC COMPARISON
# =====================================================

comparison.set_index(
    "Model"
)[
    ["Accuracy", "Precision", "Recall", "F1"]
].plot(
    kind="bar",
    figsize=(12,6)
)

plt.title(
    "Performance Comparison"
)

plt.ylabel(
    "Score"
)

plt.xticks(rotation=0)

plt.grid(True)

plt.show()

print("\nPROGRAM FINISHED")