import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Load the dataset
df = pd.read_csv('C:/Users/User/Downloads/insurance (1).csv')

# Preprocess the dataset
df['sex'] = df['sex'].str.lower().map({'male': 1, 'female': 0})
df['smoker'] = df['smoker'].str.lower().map({'yes': 1, 'no': 0})
df = pd.get_dummies(df, columns=['region'])

df['sex'].fillna(df['sex'].mode()[0], inplace=True)
df['bmi'].fillna(df['bmi'].mean(), inplace=True)
df['children'].fillna(df['children'].mean(), inplace=True)
df['smoker'].fillna(df['smoker'].mode()[0], inplace=True)
df['charges'].fillna(df['charges'].mean(), inplace=True)

df['high_risk_flag'] = np.where((df['bmi'] > 30) & (df['smoker'] == 1), 1, 0)
df.dropna(inplace=True)

features = ['age', 'sex', 'bmi', 'children', 'smoker'] + [col for col in df.columns if 'region_' in col]
high_risk_target = 'high_risk_flag'

X_train_high_risk, X_test_high_risk, y_train_high_risk, y_test_high_risk = train_test_split(df[features], df[high_risk_target], test_size=0.2, random_state=42)

high_risk_model = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gbr', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('svc', SVC(kernel='linear', probability=True)),
        ('xgb', xgb.XGBClassifier(random_state=42))
    ],
    final_estimator=RandomForestClassifier(n_estimators=100, random_state=42)
)

high_risk_model.fit(X_train_high_risk, y_train_high_risk)
high_risk_predictions_proba = high_risk_model.predict_proba(X_test_high_risk)[:, 1]  # Predict probabilities for class 1 (high risk)

# Define thresholds for categorization
low_threshold = 0.3
high_threshold = 0.7

# Categorize predictions based on thresholds
high_risk_predictions = np.where(high_risk_predictions_proba < low_threshold, 'Low', 
                        np.where(high_risk_predictions_proba < high_threshold, 'Medium', 'High'))

# Save the high-risk model
with open('high_risk_model.pkl', 'wb') as f:
    pickle.dump(high_risk_model, f)

# Predict for sample data
sample_data = pd.DataFrame([[25, 1, 22.5, 0, 0, 0, 1, 0, 0]], columns=features)
predicted_high_risk_proba = high_risk_model.predict_proba(sample_data)[:, 1]
predicted_high_risk_category = np.where(predicted_high_risk_proba < low_threshold, 'Low', 
                                        np.where(predicted_high_risk_proba < high_threshold, 'Medium', 'High'))

print(f"Predicted High-Risk Probability: {predicted_high_risk_proba[0]:.2f}")
print(f"Predicted High-Risk Category: {predicted_high_risk_category[0]}")

def print_classification_metrics(y_true, y_pred, model_name):
    # Map categorical predictions to numerical labels
    label_map = {"Low": 0, "Medium": 1, "High": 2}
    y_pred_numeric = np.array([label_map[label] for label in y_pred])

    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred_numeric):.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred_numeric))
    print("Classification Report:")
    print(classification_report(y_true, y_pred_numeric))
    print("\n")

# Call the function with categorical predictions
print_classification_metrics(y_test_high_risk, high_risk_predictions, "High-Risk Customer Identification")

