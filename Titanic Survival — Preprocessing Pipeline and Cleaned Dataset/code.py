
# TITANIC DATA PREPROCESSING PIPELINE

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
df = sns.load_dataset("titanic")
print("Original Dataset:", df.shape)

# 2. Select useful columns
df = df[["survived", "pclass", "sex", "age",
         "sibsp", "parch", "fare", "embarked"]].copy()

# 3. Handle missing values
df["age"] = df["age"].fillna(df["age"].median())
df["fare"] = df["fare"].fillna(df["fare"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# 4. Remove duplicate rows
df = df.drop_duplicates()

# 5. Convert categorical data into numbers
df = pd.get_dummies(df, columns=["sex", "embarked"], dtype=int)

# 6. Separate features and target
X = df.drop("survived", axis=1)
y = df["survived"]

# 7. Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Scale numerical features
numeric_cols = ["age", "fare", "sibsp", "parch"]
scaler = StandardScaler()

X_train.loc[:, numeric_cols] = scaler.fit_transform(
    X_train[numeric_cols]
)
X_test.loc[:, numeric_cols] = scaler.transform(
    X_test[numeric_cols]
)

# 9. Save cleaned dataset
df.to_csv("titanic_cleaned.csv", index=False)

# 10. Display results
print("\nCleaned Dataset:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nCleaned Data Preview:")
print(df.head())
print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)
print("\nCleaned dataset saved as titanic_cleaned.csv")
