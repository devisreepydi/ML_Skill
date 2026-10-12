
# TITANIC SURVIVAL - EDA AND ML LIFECYCLE MAPPING

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Load dataset
df = sns.load_dataset("titanic")
print("Dataset Loaded Successfully!")
print(df.head())
print("\nDataset Shape:", df.shape)

# 3. Exploratory Data Analysis (EDA)
print("\nMissing Values:")
print(df.isnull().sum())

print("\nSurvival Counts:")
print(df["survived"].value_counts())

# 4. Visualizations
sns.countplot(data=df, x="survived")
plt.title("Titanic Survival Count")
plt.show()

sns.barplot(data=df, x="sex", y="survived", errorbar=None)
plt.title("Survival Rate by Sex")
plt.show()

sns.barplot(data=df, x="pclass", y="survived", errorbar=None)
plt.title("Survival Rate by Passenger Class")
plt.show()

# 5. Data Cleaning
data = df[["survived", "pclass", "sex", "age", "sibsp",
           "parch", "fare", "embarked"]].copy()

data["age"] = data["age"].fillna(data["age"].median())
data["fare"] = data["fare"].fillna(data["fare"].median())
data["embarked"] = data["embarked"].fillna(data["embarked"].mode()[0])
data = data.drop_duplicates()

# 6. Feature Engineering
data = pd.get_dummies(data, columns=["sex", "embarked"], drop_first=True)

X = data.drop("survived", axis=1)
y = data["survived"]

# 7. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Train the Machine Learning model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 9. Model Evaluation
predictions = model.predict(X_test)

print("\nModel Accuracy:",
      round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# 10. Predict a passenger's survival
passenger = pd.DataFrame([{
    "pclass": 3,
    "age": 22,
    "sibsp": 1,
    "parch": 0,
    "fare": 7.25,
    "sex_male": True,
    "embarked_Q": False,
    "embarked_S": True
}])

passenger = passenger.reindex(columns=X.columns, fill_value=False)
result = model.predict(passenger)[0]

print("\nPassenger Prediction:",
      "Survived" if result == 1 else "Did Not Survive")

print("\nTitanic ML Project Completed Successfully!")
