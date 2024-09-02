import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
import xgboost as xgb
import pickle

df = pd.read_csv('C:/Users/User/Downloads/insurance (1).csv')

df['sex'] = df['sex'].str.lower().map({'male': 1, 'female': 0})
df['smoker'] = df['smoker'].str.lower().map({'yes': 1, 'no': 0})
df = pd.get_dummies(df, columns=['region'])

df['sex'].fillna(df['sex'].mode()[0], inplace=True)
df['bmi'].fillna(df['bmi'].mean(), inplace=True)
df['children'].fillna(df['children'].mean(), inplace=True)
df['smoker'].fillna(df['smoker'].mode()[0], inplace=True)
df['charges'].fillna(df['charges'].mean(), inplace=True)


df.dropna(inplace=True)

features = ['age', 'sex', 'bmi', 'children', 'smoker'] + [col for col in df.columns if 'region_' in col]
premium_target = 'charges'

X_train_premium, X_test_premium, y_train_premium, y_test_premium = train_test_split(df[features], df[premium_target], test_size=0.2, random_state=42)

premium_model = StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gbr', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ('svr', SVR(kernel='linear')),
        ('xgb', xgb.XGBRegressor(random_state=42))
    ],
    final_estimator=RandomForestRegressor(n_estimators=100, random_state=42)
)

premium_model.fit(X_train_premium, y_train_premium)


with open('premium_model.pkl', 'wb') as f:
    pickle.dump(premium_model, f)

sample_data = pd.DataFrame([[25, 1, 22.5, 0, 0, 0, 1, 0, 0]], columns=features)
predicted_premium = premium_model.predict(sample_data)
print(f"Predicted Premium: {predicted_premium[0]:.2f}")


print("Premium model training, saving, and prediction complete.")
