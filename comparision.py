# CT4101 Assignment 2 - Comparative Analysis of Algorithm Performances

# importing required libraries
import matplotlib.pyplot as plt
import numpy as np

# model names
models = ["Default SVR", "Tuned SVR", "Default RF", "Tuned RF"]

# averaged metrics from 10-fold CV
mae_values  = [62.58, 31.73, 21.55, 22.02]
r2_values   = [0.234, 0.790, 0.896, 0.893]
rmse_values = [78.62, 40.14, 28.28, 28.66]

x = np.arange(len(models))  # positions on x-axis
width = 0.35                # bar width

# Figure 1: MAE comparison
plt.figure(figsize=(8, 5))
plt.bar(x, mae_values)
plt.xticks(x, models, rotation=15)
plt.ylabel("MAE")
plt.title("Comparison of MAE for SVR and Random Forest")
plt.tight_layout()
plt.show()

# Figure 2: R² comparison
plt.figure(figsize=(8, 5))
plt.bar(x, r2_values)
plt.xticks(x, models, rotation=15)
plt.ylabel("R²")
plt.ylim(0, 1.0)
plt.title("Comparison of R² for SVR and Random Forest")
plt.tight_layout()
plt.show()

# Figure 3: RMSE comparison (optional, similar to MAE)
plt.figure(figsize=(8, 5))
plt.bar(x, rmse_values)
plt.xticks(x, models, rotation=15)
plt.ylabel("RMSE")
plt.title("Comparison of RMSE for SVR and Random Forest")
plt.tight_layout()
plt.show()
