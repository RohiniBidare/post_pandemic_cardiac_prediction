import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
import math

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("HealthData_covid.csv")
print("Dataset loaded:", df.shape)

# -------------------------------
# Remove leakage / target columns
# -------------------------------
df = df.drop(columns=["num"], errors="ignore")

TARGET_CLASS = "classification"
TARGET_REG = "mortality_rate"

y_class = df[TARGET_CLASS]
y_reg = df[TARGET_REG]

# --------- FEATURES ---------
X = df.drop(columns=[TARGET_CLASS, TARGET_REG])

print("Final input features:", X.columns.tolist())
print("X shape:", X.shape)

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# -------------------------------
# Preprocessing
# -------------------------------
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

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, yclass_train, yclass_test, yreg_train, yreg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
)

# -------------------------------
# Build Models
# -------------------------------
clf_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])

reg_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])

# -------------------------------
# Train Models
# -------------------------------
clf_pipe.fit(X_train, yclass_train)
reg_pipe.fit(X_train, yreg_train)

# -------------------------------
# Evaluate Classification
# -------------------------------
y_pred_class = clf_pipe.predict(X_test)
print("\nClassification Accuracy:", accuracy_score(yclass_test, y_pred_class))
print(classification_report(yclass_test, y_pred_class))

# -------------------------------
# Evaluate Regression
# -------------------------------
y_pred_reg = reg_pipe.predict(X_test)
print("\nRegression MAE:", mean_absolute_error(yreg_test, y_pred_reg))
print("Regression RMSE:", math.sqrt(mean_squared_error(yreg_test, y_pred_reg)))
print("Regression R²:", r2_score(yreg_test, y_pred_reg))

# -------------------------------
# Save Models
# -------------------------------
pickle.dump(clf_pipe, open("clf_pipeline.pkl", "wb"))
pickle.dump(reg_pipe, open("reg_pipeline.pkl", "wb"))

print("\nModels saved successfully!")
