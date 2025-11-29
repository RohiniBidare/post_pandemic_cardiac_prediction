# train_multi_target.py
import pandas as pd
import numpy as np
import pickle
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    ConfusionMatrixDisplay
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ LOAD AND CLEAN DATA ------------------

DATA_PATH = "HealthData_covid.csv"   # or "HealthData.csv" if you want

df = pd.read_csv(DATA_PATH)
print("Loaded dataset (raw):", df.shape)

# Drop exact duplicates
before = df.shape[0]
df = df.drop_duplicates().reset_index(drop=True)
after = df.shape[0]
print(f"Dropped duplicates: {before - after}, remaining: {after}")

# Drop known leakage / unused columns
df = df.drop(columns=["num", "covid1", "covid2"], errors="ignore")

TARGET_CLASS = "classification"
TARGET_REG = "mortality_rate"

# Drop rows missing targets
df = df.dropna(subset=[TARGET_CLASS, TARGET_REG])

# Targets
y_class = df[TARGET_CLASS].astype(int)
y_reg = df[TARGET_REG].astype(float)

# Features
X = df.drop(columns=[TARGET_CLASS, TARGET_REG], errors="ignore")

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

print("Numeric features:", numeric_cols)
print("Categorical features:", categorical_cols)

# ------------------ DISEASE MAP ------------------

disease_map = {
    0: "No Heart Disease",
    1: "coronary artery disease",
    2: "Heart Failure",
    3: "Arrhythmias",
    4: "Valvular Heart Disease",
    5: "Cardiomyopathy",
    6: "congenital heart disease",
    7: "Pericarditis",
    8: "Myocarditis",
    9: "Hypertensive Heart Disease",
    10: "Rheumatic Heart Disease"
}

# ------------------ PREPROCESSING ------------------

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# ------------------ TRAIN / TEST SPLIT ------------------

X_train, X_test, yclass_train, yclass_test, yreg_train, yreg_test = train_test_split(
    X,
    y_class,
    y_reg,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)

# ------------------ MODELS (FOCUS ON MINORITY CLASSES) ------------------

clf_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=6,          # limit depth to reduce overfitting
        min_samples_leaf=3,   # avoid tiny leaves
        class_weight="balanced",  # pay attention to rare classes
        random_state=42
    ))
])

reg_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42
    ))
])

# ------------------ TRAINING ------------------

print("\nTraining classification model...")
clf_pipe.fit(X_train, yclass_train)
print("Done.")

print("\nTraining regression model...")
reg_pipe.fit(X_train, yreg_train)
print("Done.")

# ------------------ CLASSIFICATION EVALUATION ------------------

y_pred_class_train = clf_pipe.predict(X_train)
y_pred_class = clf_pipe.predict(X_test)

acc_train = accuracy_score(yclass_train, y_pred_class_train)
acc_test = accuracy_score(yclass_test, y_pred_class)

print("\nClassification Report:")
print("Train Accuracy:", acc_train)
print("Test Accuracy:", acc_test)
print(classification_report(yclass_test, y_pred_class, zero_division=0))

# Confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(
    yclass_test, y_pred_class, xticks_rotation=45
)
plt.title("Confusion Matrix - 11-Class Heart Disease")
plt.savefig("confusion_matrix_classification.png")
plt.close()

# ------------------ REGRESSION EVALUATION ------------------

y_pred_reg = reg_pipe.predict(X_test)
mae = mean_absolute_error(yreg_test, y_pred_reg)
rmse = math.sqrt(mean_squared_error(yreg_test, y_pred_reg))
r2 = r2_score(yreg_test, y_pred_reg)

#print("\nRegression Performance:")
#print(f"MAE: {mae:.2f}")
#print(f"RMSE: {rmse:.2f}")
#print(f"R²: {r2:.2f}")

# ------------------ SAVE MODELS ------------------

with open("clf_pipeline.pkl", "wb") as f:
    pickle.dump(clf_pipe, f)

with open("reg_pipeline.pkl", "wb") as f:
    pickle.dump(reg_pipe, f)

with open("disease_map.pkl", "wb") as f:
    pickle.dump(disease_map, f)

print("\nModels saved as:")
print("  clf_pipeline.pkl")
print("  reg_pipeline.pkl")
print("  disease_map.pkl")

# ------------------ EDA GRAPHS ------------------

print("\nSaving EDA Graphs as PNG files...")

plt.figure(figsize=(7, 5))
sns.histplot(df["age"], kde=True)
plt.title("Age Distribution")
plt.savefig("age_distribution.png")
plt.close()

plt.figure(figsize=(7, 5))
sns.histplot(df["chol"], kde=True, color="orange")
plt.title("Cholesterol Distribution")
plt.savefig("cholesterol_distribution.png")
plt.close()

plt.figure(figsize=(7, 5))
sns.histplot(df["trestbps"], kde=True, color="purple")
plt.title("Resting BP Distribution")
plt.savefig("bp_distribution.png")
plt.close()

plt.figure(figsize=(7, 5))
sns.histplot(df["thalach"], kde=True, color="green")
plt.title("Max Heart Rate Distribution")
plt.savefig("thalach_distribution.png")
plt.close()

plt.figure(figsize=(7, 5))
sns.countplot(x=df["cp"])
plt.title("Chest Pain Type Frequency")
plt.savefig("cp_frequency.png")
plt.close()

plt.figure(figsize=(7, 5))
sns.countplot(x=df["fbs"])
plt.title("Fasting Sugar Frequency")
plt.savefig("fbs_frequency.png")
plt.close()

plt.figure(figsize=(7, 5))
sns.countplot(x=df["slope"])
plt.title("Slope Distribution")
plt.savefig("slope_distribution.png")
plt.close()

plt.figure(figsize=(7, 5))
sns.histplot(df["oldpeak"], kde=True, color="red")
plt.title("Oldpeak Distribution")
plt.savefig("oldpeak_distribution.png")
plt.close()

plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

sns.pairplot(df[["age", "chol", "trestbps", "thalach"]])
plt.savefig("pairplot.png")
plt.close()

print("\nEDA Graphs saved successfully.")
print("Training complete.")
