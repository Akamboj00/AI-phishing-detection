import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
import joblib


def load_and_preprocess_data(filepath):
    """
    Loads and preprocesses the data without using the vectorizer since the CSV already contains TF-IDF features.
    """
    df = pd.read_csv(filepath)
    df = df.drop_duplicates()

    # Use the existing TF-IDF feature columns directly (without transforming again)
    X = df.drop('label', axis=1).values  # Use the TF-IDF features already in the CSV
    y = df['label'].values  # Labels remain the same
    return train_test_split(X, y, test_size=0.3, random_state=42)


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_path):
    """
    Trains the model and evaluates its performance on the test set.
    Saves the trained model to the specified path and prints the classification report.
    """
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Set Accuracy: {accuracy:.4f}")

    # Generate and print the classification report
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)

    # Save the model to a file
    joblib.dump(model, model_path)
    print(f"Random Forest model saved to {model_path}")


def plot_feature_importance(importances, feature_names, top_n=20):
    """
    Plots the top N most important features from the model.
    """
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:top_n]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[top_indices], y=[feature_names[i] for i in top_indices])
    plt.title(f"Top {top_n} Important Features")
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.show()


if __name__ == "__main__":
    # Filepaths
    filepath = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\output\AI_and_legit.csv'
    model_path = 'trained_models/random_forest_model.pkl'

    # Load data (no need for vectorizer at this stage)
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)

    # Initialize RandomForest and train
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    train_and_evaluate_model(rf_model, X_train, X_test, y_train, y_test, model_path)

    # --- Manual Feature Selection (Top N Features) ---
    N = 50  # Select the top N most important features
    sorted_indices = np.argsort(rf_model.feature_importances_)[::-1]
    top_n_indices = sorted_indices[:N]

    # Reduce the training and test sets to the top N features
    X_train_reduced = X_train[:, top_n_indices]
    X_test_reduced = X_test[:, top_n_indices]

    # Retrain with reduced features (manual selection)
    print("\nRetraining with the top 50 features (manual selection)...")
    rf_model_reduced = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_reduced.fit(X_train_reduced, y_train)
    y_pred_reduced = rf_model_reduced.predict(X_test_reduced)
    accuracy_reduced = accuracy_score(y_test, y_pred_reduced)
    print(f"Test Set Accuracy with {N} features: {accuracy_reduced:.4f}")
    print("\nClassification Report (Manual Selection):")
    print(classification_report(y_test, y_pred_reduced))

    # Retrain with reduced features (automatic selection)
    print("\nApplying automatic feature selection (SelectFromModel)...")
    selector = SelectFromModel(rf_model, threshold='mean', prefit=True)
    X_train_reduced_auto = selector.transform(X_train)
    X_test_reduced_auto = selector.transform(X_test)

    rf_model_reduced_auto = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_reduced_auto.fit(X_train_reduced_auto, y_train)
    y_pred_reduced_auto = rf_model_reduced_auto.predict(X_test_reduced_auto)
    accuracy_reduced_auto = accuracy_score(y_test, y_pred_reduced_auto)
    print(f"Test Set Accuracy after automatic feature selection: {accuracy_reduced_auto:.4f}")
    print("\nClassification Report (Automatic Selection):")
    print(classification_report(y_test, y_pred_reduced_auto))

    # Plot feature importance (optional)
    plot_feature_importance(rf_model.feature_importances_, [f'term_{i}' for i in range(X_train.shape[1])], top_n=20)
