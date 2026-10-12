
# WINE QUALITY CLASSIFICATION
# Decision Tree vs Random Forest

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Create sample dataset
data = {
    "alcohol": [8.5, 9.0, 9.2, 9.5, 10.0, 10.2, 10.5, 11.0,
                11.2, 11.5, 12.0, 12.2, 12.5, 13.0, 13.2,
                8.8, 9.8, 10.8, 11.8, 12.8],
    "volatile_acidity": [0.9, 0.8, 0.85, 0.7, 0.65, 0.6, 0.55, 0.5,
                         0.48, 0.45, 0.4, 0.38, 0.35, 0.3, 0.28,
                         0.88, 0.68, 0.52, 0.42, 0.32],
    "citric_acid": [0.1, 0.15, 0.12, 0.2, 0.22, 0.25, 0.28, 0.3,
                    0.32, 0.35, 0.38, 0.4, 0.42, 0.45, 0.48,
                    0.13, 0.23, 0.29, 0.37, 0.44],
    "residual_sugar": [2.0, 2.5, 2.2, 2.8, 3.0, 3.2, 3.5, 3.0,
                       3.8, 4.0, 3.5, 4.2, 4.0, 3.8, 4.5,
                       2.3, 2.9, 3.4, 3.7, 4.3],
    "quality": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                1, 1, 1, 1, 1, 0, 0, 0, 1, 1]
}

df = pd.DataFrame(data)

# 2. Prepare features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. Train Decision Tree
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# 5. Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# 6. Compare accuracy
print("Decision Tree Accuracy:",
      round(accuracy_score(y_test, dt_pred) * 100, 2), "%")

print("Random Forest Accuracy:",
      round(accuracy_score(y_test, rf_pred) * 100, 2), "%")

# 7. Classification reports
print("\nDecision Tree Report:")
print(classification_report(y_test, dt_pred, zero_division=0))

print("\nRandom Forest Report:")
print(classification_report(y_test, rf_pred, zero_division=0))

# 8. Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ConfusionMatrixDisplay(
    confusion_matrix(y_test, dt_pred),
    display_labels=["Lower", "Higher"]
).plot(ax=axes[0], colorbar=False)

axes[0].set_title("Decision Tree")

ConfusionMatrixDisplay(
    confusion_matrix(y_test, rf_pred),
    display_labels=["Lower", "Higher"]
).plot(ax=axes[1], colorbar=False)

axes[1].set_title("Random Forest")
plt.tight_layout()
plt.show()

# 9. Decision Tree visualization
plt.figure(figsize=(14, 7))

plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=["Lower", "Higher"],
    filled=True,
    rounded=True
)

plt.title("Wine Quality Decision Tree")
plt.show()

# 10. Random Forest feature importance
pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values().plot(kind="barh")

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.show()
