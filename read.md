read.me file 

Author: Amber Daudey (s1146575)
Course: Explainable AI

Project overview: 

This project explores explainable AI methods for breast cancer classification using a Random Forest model trained on the Breast Cancer Dataset.

The focus is on comparing Local counterfactual explanation methods (Wachter & DiCE) with global explanation techniques (Permutation Feature Importance & PDP's)

Project Structure:

Final Assignment/
│
├── Code/
│   ├── your scripts.py
│
├── data/
│   └── breast-cancer.csv
│
├── figures/
│   └── (all .png here)
│
└── results/
    └── (all .csv here)


Install dependencies with:
pip install scikit-learn dice-ml matplotlib pandas numpy

Run scripts

python scripts/train_model.py
python scripts/Partial Dependence Plots.py
python scripts/Permutation Importance.py
python scripts/DiCE Method.py
python scripts/Wachter method.py
python scripts/Evaluate Counterfactuals.py
