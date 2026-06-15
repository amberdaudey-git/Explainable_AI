import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================================================
# PATH SETUP
# =========================================================

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(project_root, "data", "breast-cancer.csv")

results_dir = os.path.join(project_root, "results")
fig_dir = os.path.join(project_root, "figures")

os.makedirs(results_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

wachter_path = os.path.join(results_dir, "wachter_counterfactual.csv")
dice_path = os.path.join(results_dir, "dice_counterfactuals.csv")

summary_path = os.path.join(results_dir, "evaluation_summary.csv")
plot_path = os.path.join(fig_dir, "evaluation_metrics.png")

# =========================================================
# LOAD DATA
# =========================================================

dataset = pd.read_csv(data_path)

target = (dataset["diagnosis"] == "M").astype(int)
features = dataset.drop(columns=["id", "diagnosis"], errors="ignore")

# =========================================================
# SPLIT + SCALE
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42,
    stratify=target
)

scaler = StandardScaler()
scaler.fit(X_train)

# Original instance
x_orig = scaler.transform(X_test.iloc[0:1])[0]

# =========================================================
# LOAD COUNTERFACTUALS
# =========================================================

wachter_df = pd.read_csv(wachter_path)
dice_df = pd.read_csv(dice_path)

# Keep only shared feature columns (safe alignment)
common_cols = list(features.columns)

wachter_df = wachter_df.reindex(columns=common_cols, fill_value=0)
dice_df = dice_df.reindex(columns=common_cols, fill_value=0)

# Scale CFs
wachter_s = scaler.transform(wachter_df.values)
dice_s = scaler.transform(dice_df.values)

# =========================================================
# METRICS
# =========================================================

def l1(cf, orig):
    return np.mean(np.abs(cf - orig), axis=1)

def l2(cf, orig):
    return np.linalg.norm(cf - orig, axis=1)

def sparsity(cf, orig, tol=1e-3):
    changed = np.abs(cf - orig) > tol
    n_changed = changed.sum(axis=1)
    return 1 - (n_changed / cf.shape[1]), n_changed

def diversity(cf):
    n = len(cf)
    if n < 2:
        return np.nan
    d = []
    for i in range(n):
        for j in range(i + 1, n):
            d.append(np.linalg.norm(cf[i] - cf[j]))
    return np.mean(d)

# =========================================================
# COMPUTE METRICS
# =========================================================

w_l1 = l1(wachter_s, x_orig)
w_l2 = l2(wachter_s, x_orig)
w_spar, w_nchg = sparsity(wachter_s, x_orig)
w_div = diversity(wachter_s)

d_l1 = l1(dice_s, x_orig)
d_l2 = l2(dice_s, x_orig)
d_spar, d_nchg = sparsity(dice_s, x_orig)
d_div = diversity(dice_s)

# =========================================================
# SUMMARY TABLE
# =========================================================

summary = pd.DataFrame({
    "Method": ["Wachter", "DiCE"],
    "Avg L1": [w_l1.mean(), d_l1.mean()],
    "Avg L2": [w_l2.mean(), d_l2.mean()],
    "Avg Sparsity": [w_spar.mean(), d_spar.mean()],
    "Avg #Changed": [w_nchg.mean(), d_nchg.mean()],
    "Diversity": [0.0 if np.isnan(w_div) else w_div, d_div]
})

print("\nSUMMARY")
print(summary)

summary.to_csv(summary_path, index=False)

# =========================================================
# PRINT PER-INSTANCE RESULTS
# =========================================================

print("\nWACHTER")
for i in range(len(wachter_s)):
    print(f"CF {i+1}: L1={w_l1[i]:.3f}, L2={w_l2[i]:.3f}, "
          f"Sparsity={w_spar[i]:.3f}, Changed={w_nchg[i]}")

print("\nDICE")
for i in range(len(dice_s)):
    print(f"CF {i+1}: L1={d_l1[i]:.3f}, L2={d_l2[i]:.3f}, "
          f"Sparsity={d_spar[i]:.3f}, Changed={d_nchg[i]}")

# =========================================================
# PLOT
# =========================================================

metrics = ["Avg L1", "Avg L2", "Avg Sparsity", "Diversity"]

w_vals = [w_l1.mean(), w_l2.mean(), w_spar.mean(), 0.0 if np.isnan(w_div) else w_div]
d_vals = [d_l1.mean(), d_l2.mean(), d_spar.mean(), d_div]

x = np.arange(len(metrics))

plt.figure(figsize=(9, 5))
plt.bar(x - 0.2, w_vals, 0.4, label="Wachter")
plt.bar(x + 0.2, d_vals, 0.4, label="DiCE")

plt.xticks(x, metrics, rotation=20)
plt.ylabel("Score")
plt.title("Counterfactual Evaluation Metrics")
plt.legend()
plt.tight_layout()

plt.savefig(plot_path, dpi=300)
plt.close("all")
