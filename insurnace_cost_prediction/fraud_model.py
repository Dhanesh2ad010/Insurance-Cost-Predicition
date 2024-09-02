import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

# Check for unique regions and create dummy variables
print("Unique regions before creating dummies:", df['region'].unique())
df = pd.get_dummies(df, columns=['region'])
print("Columns after creating dummies:", df.columns)

df['sex'].fillna(df['sex'].mode()[0], inplace=True)
df['bmi'].fillna(df['bmi'].mean(), inplace=True)
df['children'].fillna(df['children'].mean(), inplace=True)
df['smoker'].fillna(df['smoker'].mode()[0], inplace=True)
df['charges'].fillna(df['charges'].mean(), inplace=True)

df['high_risk_flag'] = np.where((df['bmi'] > 30) & (df['smoker'] == 1), 1, 0)
df['fraud_flag'] = np.random.randint(0, 2, df.shape[0])

df.dropna(inplace=True)

# Define features and target for fraud detection
features = ['age', 'sex', 'bmi', 'children', 'smoker'] + [col for col in df.columns if 'region_' in col]
fraud_target = 'fraud_flag'

# Split the data for fraud detection
X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(df[features], df[fraud_target], test_size=0.2, random_state=42)

# Define and train the stacking classifier for fraud detection
fraud_model = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gbr', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('svc', SVC(kernel='linear', probability=True)),
        ('xgb', xgb.XGBClassifier(random_state=42))
    ],
    final_estimator=RandomForestClassifier(n_estimators=100, random_state=42)
)

fraud_model.fit(X_train_fraud, y_train_fraud)
fraud_predictions = fraud_model.predict(X_test_fraud)

# Function to print classification metrics
def print_classification_metrics(y_true, y_pred, title):
    print(f"=== {title} ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

# Print classification metrics for fraud detection
print_classification_metrics(y_test_fraud, fraud_predictions, "Fraud Detection")

# Save the fraud detection model
with open('fraud_model.pkl', 'wb') as f:
    pickle.dump(fraud_model, f)

# Make predictions for a sample data
# Ensure the sample_data has the same columns as features
sample_data = pd.DataFrame([[25, 1, 22.5, 0, 0, 0, 1, 0, 0]], columns=features)
predicted_fraud = fraud_model.predict(sample_data)
print(f"Predicted Fraud Flag: {predicted_fraud[0]}")

# Print completion message
print("Fraud model training, saving, and prediction complete.")
