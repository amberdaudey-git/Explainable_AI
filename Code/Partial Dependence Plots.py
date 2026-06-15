import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay

# ----------------------------
# Paths
# ----------------------------

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(project_root, "data", "breast-cancer.csv")

fig_dir = os.path.join(project_root, "figures")
os.makedirs(fig_dir, exist_ok=True)

# ----------------------------
# Load data
# ----------------------------

dataset = pd.read_csv(data_path)

target = (dataset["diagnosis"] == "M").astype(int)
features = dataset.drop(columns=["id", "diagnosis"], errors="ignore")

# ----------------------------
# Train / test split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42,
    stratify=target
)

# ----------------------------
# Scaling
# ----------------------------

scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

X_train_df = pd.DataFrame(X_train_s, columns=features.columns)
X_test_df = pd.DataFrame(X_test_s, columns=features.columns)

# ----------------------------
# Model
# ----------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_df, y_train)

print(f"Accuracy: {model.score(X_test_df, y_test):.4f}")

# ----------------------------
# Selected features for PDP
# ----------------------------

top_features = [
    "concave points_worst",
    "texture_worst",
    "area_worst",
    "concavity_worst"
]

top_features = [f for f in top_features if f in features.columns]

# ----------------------------
# 1D PDP
# ----------------------------

print("Generating 1D PDP...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

PartialDependenceDisplay.from_estimator(
    model,
    X_train_df,
    features=top_features,
    target=1,
    ax=axes,
    grid_resolution=20
)

for ax, feat in zip(axes, top_features):
    ax.set_title(feat)
    ax.set_ylabel("P(Malignant)")

plt.tight_layout()

plt.savefig(os.path.join(fig_dir, "pdp_top_features.png"), dpi=300)
plt.close("all")

print("Saved: pdp_top_features.png")

# ----------------------------
# 2D PDP
# ----------------------------

print("Generating 2D PDP...")

if len(top_features) >= 2:

    fig, ax = plt.subplots(figsize=(6, 5))

    PartialDependenceDisplay.from_estimator(
        model,
        X_train_df,
        features=[(top_features[0], top_features[1])],
        target=1,
        ax=ax,
        grid_resolution=20
    )

    ax.set_title(f"{top_features[0]} vs {top_features[1]}")

    plt.tight_layout()

    plt.savefig(os.path.join(fig_dir, "pdp_2d.png"), dpi=300)
    plt.close("all")

    print("Saved: pdp_2d.png")

# ----------------------------
# ICE + PDP
# ----------------------------

print("Generating ICE + PDP...")

fig, ax = plt.subplots(figsize=(7, 5))

PartialDependenceDisplay.from_estimator(
    model,
    X_train_df,
    features=[top_features[0]],
    target=1,
    kind="both",
    subsample=50,
    grid_resolution=20,
    ax=ax
)

ax.set_title(f"ICE + PDP: {top_features[0]}")
ax.set_ylabel("P(Malignant)")

plt.tight_layout()

plt.savefig(os.path.join(fig_dir, "pdp_ice.png"), dpi=300)
plt.close("all")

print("Saved: pdp_ice.png")
