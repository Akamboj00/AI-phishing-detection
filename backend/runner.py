# unified_runner.py
import pandas as pd
from RandomForest import load_and_preprocess_data, train_and_evaluate_model
from sklearn.ensemble import RandomForestClassifier
from NaiveBayes import train_and_evaluate_model as nb_train_and_evaluate_model
from sklearn.naive_bayes import GaussianNB
from XGBoost import train_and_evaluate_model as xgb_train_and_evaluate_model
import xgboost as xgb

# Define the file path
filepath = '../output/combined_email_features.csv'

# Load and preprocess data (shared across all models)
X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)

# Run RandomForestClassifier
print("Running RandomForestClassifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
train_and_evaluate_model(rf_model, X_train, X_test, y_train, y_test)

# Run GaussianNB
print("\nRunning NaiveBayes...")
nb_model = GaussianNB()
nb_train_and_evaluate_model(nb_model, X_train, X_test, y_train, y_test)

# Run XGBoost
print("\nRunning XGBoost...")
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_train_and_evaluate_model(xgb_model, X_train, X_test, y_train, y_test)