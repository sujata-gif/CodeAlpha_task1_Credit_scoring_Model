# Credit Scoring Model using Random Forest

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ==========================
# Load Dataset
# ==========================
data = pd.read_csv("credit_data.csv")

# Features and Target
X = data.drop("Target", axis=1)
y = data["Target"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Train Model
# ==========================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# Evaluate Model
# ==========================
y_pred = model.predict(X_test)

print("=" * 40)
print("MODEL EVALUATION")
print("=" * 40)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC AUC  :", roc_auc_score(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# ==========================
# Save Model
# ==========================
joblib.dump(model, "credit_model.pkl")
print("\nModel saved as credit_model.pkl")

# ==========================
# Predict New Customer
# ==========================
new_customer = pd.DataFrame({
    "Income": [65000],
    "Debt": [12000],
    "PaymentHistory": [1],
    "CreditScore": [740],
    "LoanAmount": [18000]
})

prediction = model.predict(new_customer)

print("\n" + "=" * 40)
print("NEW CUSTOMER PREDICTION")
print("=" * 40)

if prediction[0] == 1:
    print("Credit Status : GOOD CREDIT")
else:
    print("Credit Status : BAD CREDIT")
