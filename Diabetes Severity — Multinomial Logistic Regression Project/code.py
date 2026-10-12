
# DIABETES SEVERITY - MULTINOMIAL LOGISTIC REGRESSION PROJECT

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

print("Libraries imported successfully!")

# 2. Load diabetes dataset
diabetes = load_diabetes(as_frame=True)
df = diabetes.frame.copy()

print("\nDataset loaded successfully!")
print(df.head())
print("\nDataset shape:", df.shape)

# 3. Explore dataset
print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe())

# 4. Visualise the original target
plt.figure(figsize=(8, 5))
sns.histplot(df["target"], bins=30, kde=True)
plt.title("Distribution of Diabetes Progression Target")
plt.xlabel("Target value")
plt.ylabel("Frequency")
plt.show()

# 5. Create three target categories
# These categories are based on the target's tertiles,
# not on medical severity guidelines.
first_cut = df["target"].quantile(1/3)
second_cut = df["target"].quantile(2/3)

df["severity"] = pd.cut(
    df["target"],
    bins=[-np.inf, first_cut, second_cut, np.inf],
    labels=["Category 1", "Category 2", "Category 3"],
    include_lowest=True
)

print("\nCategory counts:")
print(df["severity"].value_counts())

# 6. Visualise categories
sns.countplot(data=df, x="severity")
plt.title("Distribution of Target Categories")
plt.xlabel("Target category")
plt.ylabel("Number of records")
plt.show()

# 7. Select input features and target
X = df.drop(columns=["target", "severity"])
y = df["severity"]

print("\nInput features:")
print(X.head())

print("\nTarget categories:")
print(y.head())

# 8. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# 9. Create multinomial logistic regression pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        solver="lbfgs",
        max_iter=5000
    ))
])

print("\nModel pipeline created successfully!")

# 10. Train model
model.fit(X_train, y_train)

print("Model trained successfully!")

# 11. Make predictions
y_pred = model.predict(X_test)

print("\nActual categories:")
print(y_test.head(10).to_list())

print("\nPredicted categories:")
print(y_pred[:10])

# 12. Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# 13. Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 14. Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.named_steps["classifier"].classes_
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.named_steps["classifier"].classes_
)

disp.plot(cmap="Blues")
plt.title("Diabetes Category Confusion Matrix")
plt.show()

# 15. Predict category for a sample record
sample = X_test.iloc[[0]]
predicted_category = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]

print("\nSample Record:")
print(sample)

print("\nPredicted target category:", predicted_category)
print("\nPredicted category probabilities:")

for category, probability in zip(
    model.named_steps["classifier"].classes_,
    probabilities
):
    print(category, ":", round(probability * 100, 2), "%")

# 16. Save predictions
results = X_test.copy()
results["Actual_Category"] = y_test
results["Predicted_Category"] = y_pred

results.to_csv("diabetes_multinomial_results.csv", index=False)

# 17. Save trained model
import joblib
joblib.dump(model, "diabetes_multinomial_model.pkl")

print("\nResults saved to diabetes_multinomial_results.csv")
print("Model saved to diabetes_multinomial_model.pkl")

# 18. Project completed
print("\n======================================")
print("DIABETES MULTINOMIAL PROJECT COMPLETED!")
print("======================================")
