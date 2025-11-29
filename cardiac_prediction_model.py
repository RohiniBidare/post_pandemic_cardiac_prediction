# card_predict_pipeline.py
# Requirements: pandas, numpy, scikit-learn, matplotlib, xgboost (optional), joblib or pickle
# Run: python card_predict_pipeline.py  (or run in a notebook cell)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix, roc_curve)

# --------- Config ---------
#DATA_PATH = "/mnt/data/HealthData.csv"  # update path if different
DATA_PATH = "HealthData_covid.csv"

# If you know the true target, set it here. If None, script will auto-detect.

# If you know the true target, set it here. If None, script will auto-detect.
FORCED_TARGET = "num"  # target column for cardiac prediction

RANDOM_STATE = 42
TEST_SIZE = 0.20

MODEL_OUTPUT = "best_cardiac_model.pkl"

# --------------------------

df = pd.read_csv(DATA_PATH)
# ======== MACHINE LEARNING PIPELINE ========

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1️ Set your target column manually
FORCED_TARGET = "num"   # cardiac prediction target
target = FORCED_TARGET

# 2️ Separate features and target
y = df[target]
X = df.drop(columns=[target])

# 3️ Identify numeric & categorical features
numeric_feats = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_feats = X.select_dtypes(exclude=[np.number]).columns.tolist()

# 4️ Preprocessing pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))

   
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_feats),
    ("cat", categorical_pipeline, categorical_feats)
])

# 5️ Encode target if categorical
if y.dtype == object or y.dtype == bool:
    le = LabelEncoder()
    y = le.fit_transform(y.astype(str))

# 6️ Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 7️ Build models
log_pipe = Pipeline([("pre", preprocessor), ("clf", LogisticRegression(max_iter=2000))])
rf_pipe = Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(random_state=42))])

# 8️ Train and evaluate
def eval_model(pipe, X_t, y_t):
    y_pred = pipe.predict(X_t)
    y_proba = pipe.predict_proba(X_t)[:,1] if hasattr(pipe.named_steps['clf'], "predict_proba") else None
    metrics = {
        "accuracy": accuracy_score(y_t, y_pred),
        "precision": precision_score(y_t, y_pred, zero_division=0),
        "recall": recall_score(y_t, y_pred, zero_division=0),
        "f1": f1_score(y_t, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_t, y_proba) if y_proba is not None else None
    }
    return metrics, y_pred, y_proba

print("\nTraining Logistic Regression...")
log_pipe.fit(X_train, y_train)
metrics_log, ypred_log, yprob_log = eval_model(log_pipe, X_test, y_test)
print("Logistic Regression:", metrics_log)

print("\nTraining Random Forest...")
rf_pipe.fit(X_train, y_train)
metrics_rf, ypred_rf, yprob_rf = eval_model(rf_pipe, X_test, y_test)
print("Random Forest:", metrics_rf)

# 9️ Classification report & confusion matrix
print("\nClassification Report (Random Forest):\n", classification_report(y_test, ypred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, ypred_rf))

# ROC curve
if yprob_rf is not None:
    fpr, tpr, _ = roc_curve(y_test, yprob_rf)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title("ROC Curve - Random Forest")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.show()

# 11️ Feature importances
try:
    pre = rf_pipe.named_steps['pre']
    num_cols = numeric_feats
    cat_cols = []
    if categorical_feats:
        ohe = pre.named_transformers_['cat'].named_steps['onehot']
        ohe_cols = list(ohe.get_feature_names_out(categorical_feats))
        cat_cols = ohe_cols
    feat_names = num_cols + cat_cols
    importances = rf_pipe.named_steps['clf'].feature_importances_
    fi = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(20)
    print("\nTop 20 Feature Importances:\n", fi)
    fi.plot(kind='barh', figsize=(8,6))
    plt.title("Top 20 Important Features")
    plt.show()
except Exception as e:
    print("Could not show feature importances:", e)

# 12️ Save model
with open("best_cardiac_model.pkl", "wb") as f:
    pickle.dump(rf_pipe, f)
print("\n Saved trained model to: best_cardiac_model.pkl")

import seaborn as sns
import matplotlib.pyplot as plt

# Quick overview
print("Data shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nBasic statistics:")
print(df.describe(include='all'))

# Plot missing values
plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()

# Correlation heatmap (numerical only)
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

print("Loaded:", df.shape)
print(df.columns.tolist())
print(df.head())

# ----- Auto-detect target if not provided -----
if FORCED_TARGET:
    target = FORCED_TARGET
else:
    # heuristic: columns that are binary-like or contain typical keywords
    candidate = []
    for col in df.columns:
        vals = df[col].dropna().unique()
        if len(vals) <= 3:
            candidate.append(col)
    # prefer ones with keywords
    keywords = ["heart", "card", "cardiac", "target", "disease", "outcome", "event"]
    target = None
    for kw in keywords:
        for c in candidate:
            if kw in c.lower():
                target = c
                break
        if target:
            break
    if target is None and candidate:
        target = candidate[-1]  # fallback
if target is None:
    raise RuntimeError("No target column found; set FORCED_TARGET to the correct column name.")

print("Using target column:", target)

# Drop rows with missing target
df = df[df[target].notna()].copy()

y = df[target]
X = df.drop(columns=[target])

# Basic checks
print("Missing per column:\n", df.isna().sum().sort_values(ascending=False).head(20))
print("Target value counts:\n", y.value_counts())

# Identify feature types
numeric_feats = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_feats = X.select_dtypes(exclude=[np.number]).columns.tolist()
print("Numeric:", numeric_feats)
print("Categorical:", categorical_feats)

# Preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

   

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_feats),
    ("cat", categorical_pipeline, categorical_feats)
])

# Encode target if needed
if y.dtype == object or y.dtype == bool:
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    print("LabelEncoder classes:", le.classes_)
else:
    # ensure binary 0/1
    uniques = sorted(y.unique())
    if set(uniques) <= {0,1}:
        y_enc = y.astype(int)
    else:
        # Simple mapping: most frequent -> 0, others -> 1 (adjust as needed)
        order = y.value_counts().index.tolist()
        mapping = {order[0]: 0}
        for v in order[1:]:
            mapping[v] = 1
        y_enc = y.map(mapping)
        print("Applied fallback mapping to target:", mapping)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc if len(np.unique(y_enc))>1 else None)
print("Train/test:", X_train.shape, X_test.shape)

# Model pipelines
log_pipe = Pipeline([("pre", preprocessor), ("clf", LogisticRegression(max_iter=2000))])
rf_pipe = Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(random_state=RANDOM_STATE))])

# Fit baselines
print("Training Logistic Regression...")
log_pipe.fit(X_train, y_train)
print("Training Random Forest...")
rf_pipe.fit(X_train, y_train)
print("\nTraining Logistic Regression...")
log_pipe.fit(X_train, y_train)

print("Train Accuracy (Logistic):", log_pipe.score(X_train, y_train))
print("Test Accuracy  (Logistic):", log_pipe.score(X_test, y_test))


print("\nTraining Random Forest...")
rf_pipe.fit(X_train, y_train)

print("Train Accuracy (RF):", rf_pipe.score(X_train, y_train))
print("Test Accuracy  (RF):", rf_pipe.score(X_test, y_test))


# Predict & evaluate
def eval_model(pipe, X_t, y_t):
    y_pred = pipe.predict(X_t)
    y_proba = pipe.predict_proba(X_t)[:,1] if hasattr(pipe.named_steps['clf'], "predict_proba") else None
    metrics = {
        "accuracy": accuracy_score(y_t, y_pred),
        "precision": precision_score(y_t, y_pred, zero_division=0),
        "recall": recall_score(y_t, y_pred, zero_division=0),
        "f1": f1_score(y_t, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_t, y_proba) if y_proba is not None and len(np.unique(y_t))>1 else None
    }
    return metrics, y_pred, y_proba

metrics_log, ypred_log, yprob_log = eval_model(log_pipe, X_test, y_test)
metrics_rf, ypred_rf, yprob_rf = eval_model(rf_pipe, X_test, y_test)

print("Logistic:", metrics_log)
print("RF:", metrics_rf)
print("Classification report (RF):\n", classification_report(y_test, ypred_rf, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_test, ypred_rf))

# Small GridSearch for RandomForest
param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [None, 10, 20],
    "clf__class_weight": [None, "balanced"]
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(rf_pipe, param_grid, cv=cv, scoring="roc_auc" if len(np.unique(y_train))>1 else "accuracy", n_jobs=-1, verbose=1)
print("Starting GridSearch (this may take a few minutes)...")
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
best_model = grid.best_estimator_

# Evaluate best model
best_metrics, ypred_best, yprob_best = eval_model(best_model, X_test, y_test)
print("Best model metrics:", best_metrics)
print("Classification report (best):\n", classification_report(y_test, ypred_best, zero_division=0))

# ROC curve (if available)
if yprob_best is not None and len(np.unique(y_test))>1:
    fpr, tpr, _ = roc_curve(y_test, yprob_best)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title("ROC Curve - Best Model")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.grid(True)
    plt.show()

# Save model
with open(MODEL_OUTPUT, "wb") as f:
    pickle.dump(best_model, f)
print("Saved model to:", MODEL_OUTPUT)

# Optionally show top feature importances (for RF)
try:
    pre = best_model.named_steps['pre']
    num_cols = numeric_feats
    cat_cols = []
    if categorical_feats:
        ohe = pre.named_transformers_['cat'].named_steps['onehot']
        ohe_cols = list(ohe.get_feature_names_out(categorical_feats))
        cat_cols = ohe_cols
    feat_names = num_cols + cat_cols
    importances = best_model.named_steps['clf'].feature_importances_
    import pandas as pd
    fi = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(30)
    print(fi)
except Exception as e:
    print("Feature importance extraction error:", e)
