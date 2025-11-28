# CT4101 Assignment 2 - Algorithm 2: Random Forest Regression

# importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

RANDOM_STATE = 42  # for repeatability

# function to calculate metrics
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, r2, rmse

# loading dataset
df = pd.read_csv("steel.csv")

# separating features and target
X = df.drop(columns=["tensile_strength"])
y = df["tensile_strength"]

# setting up 10-fold cross validation
kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

# ---------------------------------------------------------
# Default Random Forest Model
# ---------------------------------------------------------

default_rf = RandomForestRegressor(
    random_state=RANDOM_STATE,
    n_jobs=-1
)

default_test_mae = []
default_test_r2 = []
default_test_rmse = []

default_y_test_all = []
default_y_pred_all = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    default_rf.fit(X_train, y_train)
    y_pred = default_rf.predict(X_test)

    mae, r2, rmse = compute_metrics(y_test, y_pred)

    default_test_mae.append(mae)
    default_test_r2.append(r2)
    default_test_rmse.append(rmse)

    default_y_test_all.append(y_test.values)
    default_y_pred_all.append(y_pred)

default_y_test_all = np.concatenate(default_y_test_all)
default_y_pred_all = np.concatenate(default_y_pred_all)

print("\nDefault RF (10-fold average):")
print(f"MAE:  {np.mean(default_test_mae):.2f}")
print(f"R2:   {np.mean(default_test_r2):.3f}")
print(f"RMSE: {np.mean(default_test_rmse):.2f}")

# ---------------------------------------------------------
# Tuned Random Forest Model (GridSearchCV)
# ---------------------------------------------------------

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10]
}

rf_grid = GridSearchCV(
    estimator=RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=5,
    n_jobs=-1
)

rf_grid.fit(X, y)

best_rf = rf_grid.best_estimator_

tuned_test_mae = []
tuned_test_r2 = []
tuned_test_rmse = []

tuned_y_test_all = []
tuned_y_pred_all = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    best_rf.fit(X_train, y_train)
    y_pred = best_rf.predict(X_test)

    mae, r2, rmse = compute_metrics(y_test, y_pred)

    tuned_test_mae.append(mae)
    tuned_test_r2.append(r2)
    tuned_test_rmse.append(rmse)

    tuned_y_test_all.append(y_test.values)
    tuned_y_pred_all.append(y_pred)

tuned_y_test_all = np.concatenate(tuned_y_test_all)
tuned_y_pred_all = np.concatenate(tuned_y_pred_all)

print("\nTuned RF (10-fold average):")
print(f"MAE:  {np.mean(tuned_test_mae):.2f}")
print(f"R2:   {np.mean(tuned_test_r2):.3f}")
print(f"RMSE: {np.mean(tuned_test_rmse):.2f}")

# ---------------------------------------------------------
# Plots for RF
# ---------------------------------------------------------

# default RF plot
plt.figure(figsize=(7, 6))
plt.scatter(default_y_test_all, default_y_pred_all, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Default Random Forest - Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.tight_layout()
plt.show()

# tuned RF plot
plt.figure(figsize=(7, 6))
plt.scatter(tuned_y_test_all, tuned_y_pred_all, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Tuned Random Forest - Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.tight_layout()
plt.show()
