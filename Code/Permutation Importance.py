import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

# ----------------------------
# Paths 
# ----------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(project_root, "data", "breast-cancer.csv")

fig_dir = os.path.join(project_root, "figures")
results_dir = os.path.join(project_root, "results")

os.makedirs(fig_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)


# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(data_path)

y = (df["diagnosis"] == "M").astype(int)
X = df.drop(columns=["id", "diagnosis"])

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Scaling
# ----------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ----------------------------
# Model
# ----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_s, y_train)

acc = accuracy_score(y_test, model.predict(X_test_s))
print(f"Baseline accuracy: {acc:.4f}")

# ----------------------------
# Permutation importance
# ----------------------------
perm = permutation_importance(
    model,
    X_test_s,
    y_test,
    n_repeats=30,
    random_state=42,
    n_jobs=-1,
    scoring="accuracy"
)

pfi_df = pd.DataFrame({
    "feature": X.columns,
    "importance": perm.importances_mean,
    "std": perm.importances_std
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("\nTop 10 features:")
print(pfi_df.head(10).to_string(index=False))

# save table
pfi_df.to_csv(
    os.path.join(results_dir, "permutation_importance.csv"),
    index=False
)

# ----------------------------
# Full importance plot
# ----------------------------
fig, ax = plt.subplots(figsize=(10, 7))

ax.barh(
    pfi_df["feature"][::-1],
    pfi_df["importance"][::-1],
    xerr=pfi_df["std"][::-1],
    color=["#d62728" if v > 0 else "#aec7e8" for v in pfi_df["importance"][::-1]],
    capsize=3
)

ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Decrease in accuracy")
ax.set_title("Permutation Feature Importance")

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "permutation_importance.png"), dpi=150)
plt.close()

print("Saved full plot")

# ----------------------------
# Top-10 plot
# ----------------------------
top10 = pfi_df.head(10)

fig, ax = plt.subplots(figsize=(8, 5))

ax.barh(
    top10["feature"][::-1],
    top10["importance"][::-1],
    xerr=top10["std"][::-1],
    color="#d62728",
    capsize=3
)

ax.set_xlabel("Decrease in accuracy")
ax.set_title("Top-10 Permutation Importance")

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "permutation_importance_top10.png"), dpi=150)
plt.close()

print("Saved top 10 plot")

