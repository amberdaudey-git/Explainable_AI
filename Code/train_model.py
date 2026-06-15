import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Paths 
# ----------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE, "data", "breast-cancer.csv")

FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------------
# Load data
# ----------------------------
dataset = pd.read_csv(DATA_PATH)

y = (dataset["diagnosis"] == "M").astype(int)
X = dataset.drop(columns=["id", "diagnosis"])

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
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

# ----------------------------
# Model
# ----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_s, y_train)

# ----------------------------
# Evaluation
# ----------------------------
y_pred = model.predict(X_test_s)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"])
disp.plot(cmap="Blues")

plt.title("Confusion Matrix")
plt.tight_layout()

plt.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

