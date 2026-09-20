import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_auc_score, RocCurveDisplay
)
from xgboost import XGBClassifier

RANDOM_STATE = 42

df = pd.read_csv("/mnt/user-data/uploads/train_u6lujuX_CVtuZ9i__1_.csv")
print("Raw shape:", df.shape)
print(df.isnull().sum())

df = df.drop(columns=["Loan_ID"])

cat_cols = ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History"]
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

num_cols = ["LoanAmount", "Loan_Amount_Term"]
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

df["Dependents"] = df["Dependents"].replace("3+", "3").astype(int)

df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["LoanAmount_log"] = np.log1p(df["LoanAmount"])
df["TotalIncome_log"] = np.log1p(df["TotalIncome"])

label_encoders = {}
for col in ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("\nCleaned shape:", df.shape)
print(df.isnull().sum().sum(), "missing values remain")

feature_cols = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "Property_Area", "TotalIncome_log", "LoanAmount_log"
]

X = df[feature_cols]
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=300, max_depth=6, min_samples_leaf=3,
    random_state=RANDOM_STATE, class_weight="balanced"
)
rf.fit(X_train, y_train)

xgb = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=RANDOM_STATE
)
xgb.fit(X_train, y_train)

models = {"Random Forest": rf, "XGBoost": xgb}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
results = {}

for ax, (name, model) in zip(axes, models.items()):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    results[name] = {"accuracy": acc, "auc": auc}

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}   ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, preds, target_names=["Rejected", "Approved"]))

    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Rejected", "Approved"],
                yticklabels=["Rejected", "Approved"])
    ax.set_title(f"{name}\nAcc: {acc:.2%}  AUC: {auc:.2f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/confusion_matrices.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
importances.plot(kind="barh", color="#4C72B0")
plt.title("Random Forest — Feature Importance")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/feature_importance.png", dpi=150)
plt.close()

best_name = max(results, key=lambda k: results[k]["auc"])
best_model = models[best_name]
print(f"\nBest model: {best_name} -> {results[best_name]}")

joblib.dump(
    {"model": best_model, "encoders": label_encoders, "feature_cols": feature_cols},
    "/mnt/user-data/outputs/loan_model.pkl"
)

def predict_loan(applicant: dict):
    """Predict whether a new loan application will be approved."""
    row = pd.DataFrame([applicant])
    row["Dependents"] = str(row["Dependents"].iloc[0]).replace("3+", "3")
    row["Dependents"] = row["Dependents"].astype(int)
    row["TotalIncome_log"] = np.log1p(row["ApplicantIncome"] + row["CoapplicantIncome"])
    row["LoanAmount_log"] = np.log1p(row["LoanAmount"])
    for col in ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]:
        row[col] = label_encoders[col].transform(row[col])
    row = row[feature_cols]
    pred = best_model.predict(row)[0]
    prob = best_model.predict_proba(row)[0][1]
    return ("Approved" if pred == 1 else "Rejected"), round(float(prob), 3)

sample = {
    "Gender": "Male", "Married": "Yes", "Dependents": "0", "Education": "Graduate",
    "Self_Employed": "No", "ApplicantIncome": 5000, "CoapplicantIncome": 2000,
    "LoanAmount": 150, "Loan_Amount_Term": 360, "Credit_History": 1.0,
    "Property_Area": "Urban"
}
status, prob = predict_loan(sample)
print(f"\nSample applicant prediction: {status} (approval probability: {prob})")

print("\nSaved: confusion_matrices.png, feature_importance.png, loan_model.pkl")
