
# HEART DISEASE CLASSIFICATION
# XGBoost, LightGBM and Feature Importance

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 2. Create sample dataset
data = {
    "age": [25, 30, 45, 50, 60, 65, 35, 55, 40, 70,
            28, 48, 62, 33, 58, 42, 67, 38, 52, 72],

    "cholesterol": [180, 190, 220, 240, 250, 260, 200, 230, 210, 270,
                    175, 225, 255, 185, 245, 205, 265, 195, 235, 280],

    "max_heart_rate": [190, 185, 170, 160, 150, 140, 180, 155, 175, 135,
                       195, 165, 145, 188, 152, 172, 138, 182, 158, 130],

    "target": [0, 0, 0, 1, 1, 1, 0, 1, 0, 1,
               0, 0, 1, 0, 1, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df.head())

# 3. Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. Train XGBoost
xgb = XGBClassifier(
    n_estimators=20,
    max_depth=2,
    eval_metric="logloss",
    random_state=42
)

xgb.fit(X_train, y_train)
pred1 = xgb.predict(X_test)

print("\nXGBoost Accuracy:",
      round(accuracy_score(y_test, pred1) * 100, 2), "%")

print("\nXGBoost Classification Report:")
print(classification_report(y_test, pred1, zero_division=0))

# 6. Train LightGBM
lgb = LGBMClassifier(
    n_estimators=20,
    max_depth=2,
    verbosity=-1,
    random_state=42
)

lgb.fit(X_train, y_train)
pred2 = lgb.predict(X_test)

print("\nLightGBM Accuracy:",
      round(accuracy_score(y_test, pred2) * 100, 2), "%")

print("\nLightGBM Classification Report:")
print(classification_report(y_test, pred2, zero_division=0))

# 7. Compare models
results = pd.DataFrame({
    "Model": ["XGBoost", "LightGBM"],
    "Accuracy": [
        accuracy_score(y_test, pred1),
        accuracy_score(y_test, pred2)
    ]
})

print("\nModel Comparison:")
print(results)

# 8. XGBoost confusion matrix
ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, pred1)
).plot()

plt.title("XGBoost Confusion Matrix")
plt.show()

# 9. LightGBM confusion matrix
ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, pred2)
).plot()

plt.title("LightGBM Confusion Matrix")
plt.show()

# 10. Compare feature importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "XGBoost": xgb.feature_importances_,
    "LightGBM": lgb.feature_importances_
})

print("\nFeature Importance:")
print(importance)

importance.set_index("Feature").plot(kind="bar")

plt.title("Feature Importance Comparison")
plt.ylabel("Importance")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 11. Predict a new example
sample = pd.DataFrame({
    "age": [45],
    "cholesterol": [220],
    "max_heart_rate": [170]
})

print("\nXGBoost predicted class:", xgb.predict(sample)[0])
print("LightGBM predicted class:", lgb.predict(sample)[0])
