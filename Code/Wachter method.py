import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ----------------------------
# Paths
# ----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

data_path = os.path.join(project_root, "data", "breast-cancer.csv")
output_csv = os.path.join(project_root, "results", "wachter_counterfactual.csv")

os.makedirs(os.path.join(project_root, "results"), exist_ok=True)

# ----------------------------
# Data
# ----------------------------
dataset = pd.read_csv(data_path)
y = (dataset["diagnosis"] == "M").astype(int)
X = dataset.drop(columns=["id", "diagnosis"], errors="ignore")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# Scaling
# ----------------------------
scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

feat_min = X_train_s.min(axis=0)
feat_max = X_train_s.max(axis=0)

# ----------------------------
# Model
# ----------------------------
model = LogisticRegression(max_iter=2000)
model.fit(X_train_s, y_train)

# ----------------------------
# Query instance
# ----------------------------
idx = 0
x_orig = X_test_s[idx].copy().reshape(1, -1)

orig_pred = model.predict(x_orig)[0]
target_class = 1 - orig_pred

print(f"Original prediction: {orig_pred}")
print(f"Target class: {target_class}")

# ----------------------------
# Optimization
# ----------------------------
max_iter = 80
lr = 0.05
lambda_param = 0.2

x_cf = x_orig.copy()
history = []

for i in range(max_iter):

    probs = model.predict_proba(x_cf)[0]
    target_prob = probs[target_class]
    history.append(target_prob)

    if i % 5 == 0:
        print(f"Iteration {i} | prob = {target_prob:.4f}")

    if target_prob > 0.90:
        break

    w = model.coef_[0]
    b = model.intercept_[0]

    z = np.dot(x_cf, w) + b
    p = 1 / (1 + np.exp(-z))

    grad = (p - target_class) * w

    x_cf = x_cf - lr * (grad + lambda_param * (x_cf - x_orig))
    x_cf = np.clip(x_cf, feat_min, feat_max)

# ----------------------------
# Result
# ----------------------------
final_pred = model.predict(x_cf)[0]
print("\nFinal prediction:", final_pred)

cf_original = scaler.inverse_transform(x_cf)[0]
orig_original = scaler.inverse_transform(x_orig)[0]

