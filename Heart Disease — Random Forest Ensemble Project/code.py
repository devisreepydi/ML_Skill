
# HEART DISEASE - RANDOM FOREST ENSEMBLE PROJECT

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Create dataset
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

# 2. Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. Create and train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Evaluate model
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 7. Confusion matrix
ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, y_pred)
).plot()
plt.show()

# 8. Feature importance
pd.Series(
    model.feature_importances_, index=X.columns
).sort_values().plot(kind="barh")

plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()

# 9. New sample prediction
sample = pd.DataFrame({
    "age": [45],
    "cholesterol": [220],
    "max_heart_rate": [170]
})

print("Sample predicted class:", model.predict(sample)[0])
