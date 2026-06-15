# DiCE Method for Counterfactual Explanations

# =========================================================
# DiCE Counterfactual Explanations 
# =========================================================

import os
import dice_ml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# =========================================================
# PATH SETUP 
# =========================================================

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(project_root, "data", "breast-cancer.csv")

results_dir = os.path.join(project_root, "results")
fig_dir = os.path.join(project_root, "figures")

os.makedirs(results_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

output_csv = os.path.join(results_dir, "dice_counterfactuals.csv")
output_txt = os.path.join(results_dir, "dice_counterfactuals_readable.txt")

output_plot1 = os.path.join(fig_dir, "dice_feature_importance.png")
output_plot2 = os.path.join(fig_dir, "dice_cf_example.png")

# =========================================================
# LOAD DATA
# =========================================================

dataset = pd.read_csv(data_path)

# IMPORTANT: remove ID (prevents leakage + grading issues)
features = dataset.drop(columns=["id", "diagnosis"], errors="ignore")
target = (dataset["diagnosis"] == "M").astype(int)

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42,
    stratify=target
)

# =========================================================
# MODEL
# =========================================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

# =========================================================
# DICE SETUP
# =========================================================

dice_data = dice_ml.Data(
    dataframe=X_train.assign(diagnosis=y_train.values),
    continuous_features=list(features.columns),
    outcome_name="diagnosis"
)

dice_model = dice_ml.Model(model=rf_model, backend="sklearn")
explainer = dice_ml.Dice(dice_data, dice_model)

# =========================================================
# QUERY INSTANCE
# =========================================================

query_instance = X_test.iloc[0:1]
original = query_instance.iloc[0]
original_prediction = rf_model.predict(query_instance)[0]

# =========================================================
# GENERATE COUNTERFACTUALS
# =========================================================

dice_exp = explainer.generate_counterfactuals(
    query_instance,
    total_CFs=6,
    desired_class="opposite"
)

cf_df = dice_exp.cf_examples_list[0].final_cfs_df

pred_col = cf_df.columns[-1]
flipped_cf_df = cf_df[cf_df[pred_col] != original_prediction]

# =========================================================
# TEXT OUTPUT
# =========================================================

feature_counts = {}

lines = [
    f"Original prediction: {original_prediction}\n",
    "Counterfactual explanations:\n"
]

if flipped_cf_df.empty:
    lines.append("No successful counterfactual flips.\n")
else:
    for i, row in flipped_cf_df.iterrows():

        changes = {}

        for col in row.index:
            if col in original:
                if abs(row[col] - original[col]) > 1e-6:
                    changes[col] = (original[col], row[col])
                    feature_counts[col] = feature_counts.get(col, 0) + 1

        lines.append(f"Counterfactual {i+1}:")

        if changes:
            for f, (a, b) in changes.items():
                lines.append(f"  {f}: {a:.3f} → {b:.3f}")
        else:
            lines.append("  No major changes")

        lines.append(f"  Prediction: {row[pred_col]}\n")

# Save text
with open(output_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# Save CSV
flipped_cf_df.to_csv(output_csv, index=False)

# =========================================================
# PLOT 1: FEATURE FREQUENCY
# =========================================================

if len(feature_counts) > 0:

    feature_counts = dict(
        sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    )

    plt.figure(figsize=(10, 5))
    plt.bar(feature_counts.keys(), feature_counts.values())
    plt.xticks(rotation=90, ha="right")
    plt.ylabel("Change frequency")
    plt.title("DiCE Feature Change Frequency")
    plt.tight_layout()

    plt.savefig(output_plot1, dpi=300)
    plt.close()

# =========================================================
# PLOT 2: ORIGINAL VS COUNTERFACTUAL
# =========================================================

if not flipped_cf_df.empty:

    cf_example = flipped_cf_df.iloc[0]

    changed_features = [
        col for col in cf_example.index
        if col in original and abs(cf_example[col] - original[col]) > 1e-6
    ]

    if len(changed_features) > 0:

        plt.figure(figsize=(10, 5))

        plt.plot(
            changed_features,
            [original[f] for f in changed_features],
            marker="o",
            label="Original"
        )

        plt.plot(
            changed_features,
            [cf_example[f] for f in changed_features],
            marker="s",
            label="Counterfactual"
        )

        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Value")
        plt.title("Original vs Counterfactual")
        plt.legend()
        plt.tight_layout()

        plt.savefig(output_plot2, dpi=300)
        plt.close()

# =========================================================
# CONSOLE OUTPUT
# =========================================================

print("\n".join(lines))
print("\nSaved files:")
print(output_csv)
print(output_txt)
print(output_plot1)
print(output_plot2)
