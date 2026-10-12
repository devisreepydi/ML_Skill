
# ADULT INCOME - EDA AND FEATURE ENGINEERING PROJECT

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

cols = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week",
    "native_country", "income"
]

df = pd.read_csv(url, names=cols, skipinitialspace=True)

print("Dataset Loaded Successfully!")
print("Original Shape:", df.shape)
print(df.head())

# 2. Explore dataset
print("\nDataset Information:")
df.info()

print("\nIncome Counts:")
print(df["income"].value_counts())

# 3. Clean missing values
df = df.replace("?", pd.NA).dropna()
print("\nMissing Values:", df.isnull().sum().sum())
print("Cleaned Shape:", df.shape)

# 4. EDA Graph 1: Income distribution
sns.countplot(data=df, x="income")
plt.title("Adult Income Distribution")
plt.show()

# 5. EDA Graph 2: Income by education
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="education", hue="income")
plt.xticks(rotation=45)
plt.title("Income by Education")
plt.tight_layout()
plt.show()

# 6. EDA Graph 3: Weekly working hours
sns.boxplot(data=df, x="income", y="hours_per_week")
plt.title("Weekly Working Hours by Income")
plt.show()

# 7. Feature Engineering: Age groups
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 25, 40, 60, 100],
    labels=["Young", "Adult", "Middle-aged", "Senior"]
)

# 8. Feature Engineering: Work hours category
df["hours_category"] = pd.cut(
    df["hours_per_week"],
    bins=[0, 40, 60, 100],
    labels=["Part-time", "Full-time", "Overtime"],
    include_lowest=True
)

# 9. Feature Engineering: Net capital
df["capital_change"] = df["capital_gain"] - df["capital_loss"]

print("\nNew Features:")
print(df[["age_group", "hours_category", "capital_change"]].head())

# 10. Encode categorical columns
df_encoded = pd.get_dummies(df, drop_first=True, dtype=int)

print("\nEncoded Dataset Shape:", df_encoded.shape)
print(df_encoded.head())

# 11. Save datasets
df.to_csv("adult_income_feature_engineered.csv", index=False)
df_encoded.to_csv("adult_income_encoded.csv", index=False)

print("\nProject Completed Successfully!")
print("Saved: adult_income_feature_engineered.csv")
print("Saved: adult_income_encoded.csv")
