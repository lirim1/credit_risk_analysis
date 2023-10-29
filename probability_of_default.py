import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# Read in loan data from a CSV file
df = pd.read_csv('loan_data.csv')

features = ['credit_lines_outstanding', 'debt_to_income', 'payment_to_income', 'years_employed', 'fico_score']

df['payment_to_income'] = df['loan_amt_outstanding'] / df['income']


df['debt_to_income'] = df['total_debt_outstanding'] / df['income']

# Create and train the Logistic Regression model
clf = LogisticRegression(random_state=0, solver='liblinear', tol=1e-5, max_iter=10000).fit(df[features], df['default'])

# Display the model's coefficients and intercept
print("Coefficients:", clf.coef_)
print("Intercept:", clf.intercept_)

# Use the trained model to make predictions
y_pred = clf.predict(df[features])

# Calculate the accuracy of the model's predictions
accuracy = (1.0 * (abs(df['default'] - y_pred)).sum()) / len(df)
print("Accuracy:", accuracy)

# Calculate the AUC (Area Under the ROC Curve)
fpr, tpr, thresholds = metrics.roc_curve(df['default'], y_pred)
auc = metrics.auc(fpr, tpr)
print("AUC:", auc) 
