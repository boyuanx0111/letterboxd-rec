import os
import sys
import faulthandler

# Must be set BEFORE numpy is imported. Works around the MKL/OpenMP crash.
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

faulthandler.enable()

print("1. Starting test", flush=True)
print("Python:", sys.executable, flush=True)

import numpy as np
print("2. NumPy", np.__version__, flush=True)

# --- Layer 1: raw NumPy BLAS ---
a = np.eye(51)
print("3. matmul:", (a @ a).shape, flush=True)

# --- Layer 2: linear solve ---
x = np.linalg.solve(a + np.eye(51), np.ones(51))
print("4. solve:", x.shape, flush=True)

# --- Layer 3: realistic ridge-style system (tall X, small A) ---
rng = np.random.default_rng(0)
X = rng.normal(size=(3, 51))
y = rng.normal(size=3)
A = X.T @ X + 10.0 * np.eye(51)
b = X.T @ y
theta = np.linalg.solve(A, b)
print("5. ridge-style solve:", theta.shape, flush=True)

# --- Layer 4: SciPy ---
from scipy.linalg import solve
theta2 = solve(A, b, assume_a="pos")
print("6. scipy solve matches:", np.allclose(theta, theta2), flush=True)

import pandas as pd
import joblib
print("7. pandas/joblib imported", flush=True)

# --- Layer 5: Surprise model ---
cf_model = joblib.load("models/collaborative_filtering.pkl")
print("8. CF model loaded:", type(cf_model).__name__, flush=True)

# --- Layer 6: full fold-in function ---
from cf import cf_predict_new_user
print("9. cf_predict_new_user imported", flush=True)

trainset = cf_model.trainset
raw_ids = [trainset.to_raw_iid(i) for i in range(min(10, trainset.n_items))]

# ADJUST HERE if your function expects a different user_ratings format
user_ratings = pd.DataFrame({
    "tmdbId": raw_ids[:3],
    "rating": [4.0, 3.5, 5.0],
})
candidate_ids = raw_ids[3:10]

preds = cf_predict_new_user(
    cf_model,
    user_ratings,
    candidate_ids,
    reg=10.0,
)
print("10. Predictions:", flush=True)
print(preds, flush=True)

print("\nALL CHECKS PASSED", flush=True)