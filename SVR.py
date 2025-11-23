# CT4101 Assignment 2 - Algorithm 1: Support Vector Regression (SVR)

# importing required libraries
import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

RANDOM_STATE = 42  # for repeatability


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


# Default SVR Model

# creating a pipeline for scaling + SVR
default_svr = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf"))
])

default_test_mae = []
default_test_r2 = []
default_test_rmse = []

# running 10-fold CV
for train_idx, test_idx in kf.split(X):
    # splitting data for this fold
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # training the model
    default_svr.fit(X_train, y_train)

    # making predictions
    y_test_pred = default_svr.predict(X_test)

    # calculating metrics
    mae, r2, rmse = compute_metrics(y_test, y_test_pred)

    default_test_mae.append(mae)
    default_test_r2.append(r2)
    default_test_rmse.append(rmse)

# printing average results
print("\nDefault SVR (10-fold average):")
print(f"MAE:  {np.mean(default_test_mae):.2f}")
print(f"R2:   {np.mean(default_test_r2):.3f}")
print(f"RMSE: {np.mean(default_test_rmse):.2f}")



# Tuned SVR Model (GridSearchCV)


# pipeline for tuning
svr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf"))
])

# parameter grid for tuning C and gamma
param_grid = {
    "svr__C": [1, 10, 100],
    "svr__gamma": ["scale", 0.01, 0.1]
}

# running grid search
svr_grid = GridSearchCV(
    estimator=svr_pipeline,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=5,
    n_jobs=-1
)

svr_grid.fit(X, y)

# best tuned model
best_svr = svr_grid.best_estimator_

tuned_test_mae = []
tuned_test_r2 = []
tuned_test_rmse = []

# evaluating tuned model with 10-fold CV
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    best_svr.fit(X_train, y_train)
    y_test_pred = best_svr.predict(X_test)

    mae, r2, rmse = compute_metrics(y_test, y_test_pred)

    tuned_test_mae.append(mae)
    tuned_test_r2.append(r2)
    tuned_test_rmse.append(rmse)

# printing average results
print("\nTuned SVR (10-fold average):")
print(f"MAE:  {np.mean(tuned_test_mae):.2f}")
print(f"R2:   {np.mean(tuned_test_r2):.3f}")
print(f"RMSE: {np.mean(tuned_test_rmse):.2f}")
