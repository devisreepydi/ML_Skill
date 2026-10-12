
# CALIFORNIA HOUSING - LINEAR REGRESSION AND REGULARISATION

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("Dataset Loaded Successfully!")
print(df.head())
print("Dataset Shape:", df.shape)

# 2. Basic EDA
print("\nDataset Information:")
print(df.describe())

# 3. Visualize house values
plt.figure(figsize=(7, 4))
sns.histplot(df["MedHouseVal"], bins=30, kde=True)
plt.title("Distribution of California House Values")
plt.show()

# 4. Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), cmap="coolwarm", annot=True, fmt=".1f")
plt.title("Feature Correlation Heatmap")
plt.show()

# 5. Separate features and target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Create models
models = {
    "Linear Regression": make_pipeline(
        StandardScaler(), LinearRegression()
    ),
    "Ridge Regression": make_pipeline(
        StandardScaler(), Ridge(alpha=1.0)
    ),
    "Lasso Regression": make_pipeline(
        StandardScaler(), Lasso(alpha=0.01, max_iter=10000)
    )
}

# 8. Train and evaluate
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    results.append([name, rmse, r2])
    print(f"\n{name}")
    print("RMSE:", round(rmse, 3))
    print("R2 Score:", round(r2, 3))

# 9. Compare models
results_df = pd.DataFrame(
    results, columns=["Model", "RMSE", "R2 Score"]
)

print("\nModel Comparison:")
print(results_df)

# 10. Plot actual vs predicted values
best_model = models["Ridge Regression"]
pred = best_model.predict(X_test)

plt.figure(figsize=(6, 5))
plt.scatter(y_test, pred, alpha=0.3)
plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Actual vs Predicted House Values")
plt.show()

# 11. Predict an example
sample = X_test.iloc[[0]]
prediction = best_model.predict(sample)[0]

print("\nExample Predicted House Value:",
      round(prediction, 3), "hundreds of thousands of dollars")

# 12. Save results
results_df.to_csv("housing_model_comparison.csv", index=False)
print("\nProject Completed Successfully!")
