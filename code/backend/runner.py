# unified_runner.py
import pandas as pd
from RandomForest import load_and_preprocess_data, train_and_evaluate_model as rf_train_and_evaluate_model
from sklearn.ensemble import RandomForestClassifier
from NaiveBayes import train_and_evaluate_model as nb_train_and_evaluate_model
from sklearn.naive_bayes import GaussianNB
from XGBoost import train_and_evaluate_model as xgb_train_and_evaluate_model
import xgboost as xgb

# Define the file path for the dataset
filepath = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\output\AI_and_legit.csv'

# Load and preprocess data (shared across all models)
print("Loading and preprocessing data...")
X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)

# RandomForestClassifier
print("\nRunning RandomForestClassifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_train_and_evaluate_model(rf_model, X_train, X_test, y_train, y_test, model_path='trained_models/random_forest_model.pkl')

# NaiveBayes
print("\nRunning NaiveBayes...")
nb_model = GaussianNB()
nb_train_and_evaluate_model(nb_model, X_train, X_test, y_train, y_test, model_path='trained_models/naive_bayes_model.pkl')

# XGBoost
print("\nRunning XGBoost...")
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_train_and_evaluate_model(xgb_model, X_train, X_test, y_train, y_test, model_path='trained_models/xgboost_model.pkl')

print("\nTraining completed for all models!")
