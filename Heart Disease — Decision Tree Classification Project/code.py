
# HEART DISEASE CLASSIFICATION USING DECISION TREE

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 2. Create a sample dataset
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

# 4. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. Create and train the model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

print("\nActual values:", y_test.values)
print("Predicted values:", y_pred)

# 7. Evaluate the model
print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 8. Display confusion matrix
cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Confusion Matrix")
plt.show()

# 9. Visualize the Decision Tree
plt.figure(figsize=(14, 7))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Class 0", "Class 1"],
    filled=True,
    rounded=True
)

plt.title("Decision Tree Classification")
plt.show()

# 10. Try a sample prediction
sample = pd.DataFrame({
    "age": [45],
    "cholesterol": [220],
    "max_heart_rate": [170]
})

prediction = model.predict(sample)

print("\nSample predicted class:", prediction[0])
