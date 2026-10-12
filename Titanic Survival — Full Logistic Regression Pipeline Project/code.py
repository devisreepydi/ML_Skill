
# TITANIC SURVIVAL - FULL LOGISTIC REGRESSION PIPELINE PROJECT

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

print("Libraries imported successfully!")

# 2. Load dataset
df = sns.load_dataset("titanic")

print("\nDataset loaded successfully!")
print(df.head())
print("\nDataset shape:", df.shape)

# 3. Explore dataset
print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nSurvival counts:")
print(df["survived"].value_counts())

# 4. Visualise survival
sns.countplot(data=df, x="survived")
plt.title("Titanic Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.show()

sns.countplot(data=df, x="sex", hue="survived")
plt.title("Survival by Sex")
plt.show()

sns.countplot(data=df, x="pclass", hue="survived")
plt.title("Survival by Passenger Class")
plt.show()

# 5. Select features and target
features = [
    "pclass", "sex", "age",
    "sibsp", "parch", "fare", "embarked"
]

X = df[features]
y = df["survived"]

print("\nSelected features:")
print(X.head())

# 6. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# 7. Define numerical and categorical columns
numeric_features = [
    "pclass", "age", "sibsp", "parch", "fare"
]

categorical_features = ["sex", "embarked"]

# 8. Preprocess numerical features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 9. Preprocess categorical features
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 10. Combine preprocessing steps
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 11. Create complete machine learning pipeline
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

print("\nFull pipeline created successfully!")

# 12. Train model
model.fit(X_train, y_train)

print("Model trained successfully!")

# 13. Make predictions
y_pred = model.predict(X_test)

print("\nFirst 10 actual values:")
print(y_test.head(10).to_list())

print("\nFirst 10 predicted values:")
print(y_pred[:10])

# 14. Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# 15. Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 16. Confusion matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Did Not Survive", "Survived"]
)

disp.plot(cmap="Blues")
plt.title("Titanic Confusion Matrix")
plt.show()

# 17. Predict survival for a sample passenger
new_passenger = pd.DataFrame([{
    "pclass": 1,
    "sex": "female",
    "age": 25,
    "sibsp": 0,
    "parch": 0,
    "fare": 80.0,
    "embarked": "S"
}])

prediction = model.predict(new_passenger)[0]
probabilities = model.predict_proba(new_passenger)[0]

print("\nSample Passenger Prediction:")
print(
    "Predicted result:",
    "Survived" if prediction == 1 else "Did not survive"
)
print("Probability of not surviving:",
      round(probabilities[0] * 100, 2), "%")
print("Probability of surviving:",
      round(probabilities[1] * 100, 2), "%")

# 18. Save trained model
joblib.dump(model, "titanic_logistic_pipeline.pkl")
print("\nTrained model saved as titanic_logistic_pipeline.pkl")

# 19. Save dataset
cleaned_df = df[features + ["survived"]].copy()
cleaned_df.to_csv("titanic_cleaned_dataset.csv", index=False)

print("Dataset saved as titanic_cleaned_dataset.csv")

# 20. Project completed
print("\n====================================")
print("TITANIC ML PROJECT COMPLETED!")
print("====================================")
