# Credit Scoring Model using Random Forest

## 📌 Overview
This project implements a **credit scoring model** using the Random Forest algorithm to classify customers into **Good Credit** or **Bad Credit** categories based on financial attributes such as income, debt, payment history, credit score, and loan amount.

## 🚀 Features
- Loads and preprocesses credit data from CSV.
- Splits dataset into training and testing sets.
- Trains a Random Forest Classifier.
- Evaluates model performance with metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC AUC
  - Confusion Matrix
- Saves the trained model as `credit_model.pkl`.
- Predicts credit status for new customers.

## 🛠️ Tech Stack
- **Language**: Python  
- **Libraries**: pandas, scikit-learn, joblib  

## 📂 Dataset
The dataset `credit_data.csv` should contain customer financial information with a target column:
- **Features**: Income, Debt, PaymentHistory, CreditScore, LoanAmount  
- **Target**: Binary label (1 = Good Credit, 0 = Bad Credit)

## 📈 Model Evaluation
The script prints evaluation metrics after training, helping assess model performance.

## 🎯 Example Prediction
For a new customer:
```python
new_customer = pd.DataFrame({
    "Income": [65000],
    "Debt": [12000],
    "PaymentHistory": [1],
    "CreditScore": [740],
    "LoanAmount": [18000]
})
prediction = model.predict(new_customer)
