# CT4101 Assignment 2 - Algorithm 2: Random Forest Regressor

# importing required libraries
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

RANDOM_STATE = 42  # for consistent results


# function to calculate metrics
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, r2, rmse


# loading the dataset
df = pd.read_csv("steel.csv")

# separating features and target
X = df.drop(columns=["tensile_strength"])
y = df["tensile_strength"]


# setting up 10-fold cross validation
kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)


# -----------------------------------------
# Default Random Forest Model
# -----------------------------------------

# creating the default model
default_rf = RandomForestRegressor(random_state=RANDOM_STATE)

default_test_mae = []
default_test_r2 = []
default_test_rmse = []

# running 10-fold CV
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # training the model
    default_rf.fit(X_train, y_train)

    # making predictions
    y_test_pred = default_rf.predict(X_test)

    # calculating metrics
    mae, r2, rmse = compute_metrics(y_test, y_test_pred)

    default_test_mae.append(mae)
    default_test_r2.append(r2)
    default_test_rmse.append(rmse)

# printing average results
print("\nDefault Random Forest (10-fold average):")
print(f"MAE:  {np.mean(default_test_mae):.2f}")
print(f"R2:   {np.mean(default_test_r2):.3f}")
print(f"RMSE: {np.mean(default_test_rmse):.2f}")


# -----------------------------------------
# Tuned Random Forest Model (GridSearchCV)
# -----------------------------------------

# parameter grid for tuning n_estimators and max_depth
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10, 20]
}

# running grid search
rf_grid = GridSearchCV(
    estimator=RandomForestRegressor(random_state=RANDOM_STATE),
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=5,
    n_jobs=-1
)

rf_grid.fit(X, y)

# best tuned model
best_rf = rf_grid.best_estimator_

tuned_test_mae = []
tuned_test_r2 = []
tuned_test_rmse = []

# evaluating tuned model with 10-fold CV
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    best_rf.fit(X_train, y_train)
    y_test_pred = best_rf.predict(X_test)

    mae, r2, rmse = compute_metrics(y_test, y_test_pred)

    tuned_test_mae.append(mae)
    tuned_test_r2.append(r2)
    tuned_test_rmse.append(rmse)

# printing average results
print("\nTuned Random Forest (10-fold average):")
print(f"MAE:  {np.mean(tuned_test_mae):.2f}")
print(f"R2:   {np.mean(tuned_test_r2):.3f}")
print(f"RMSE: {np.mean(tuned_test_rmse):.2f}")
