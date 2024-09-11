import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
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
    Trains the XGBoost model and evaluates its performance.
    Saves the trained model and prints the classification report.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Set Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model to a file
    joblib.dump(model, model_path)
    print(f"XGBoost model saved to {model_path}")


def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plots the top N most important features from the XGBoost model.
    """
    importance = model.feature_importances_
    indices = importance.argsort()[::-1]
    top_indices = indices[:top_n]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance[top_indices], y=[feature_names[i] for i in top_indices])
    plt.title(f"Top {top_n} Important Features")
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.show()


if __name__ == "__main__":
    filepath = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\output\AI_and_legit.csv'
    model_path = 'trained_models/xgboost_model.pkl'

    # Load data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)

    # Initialize XGBoost and train
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    train_and_evaluate_model(xgb_model, X_train, X_test, y_train, y_test, model_path)

    # Plot feature importance
    plot_feature_importance(xgb_model, [f'term_{i}' for i in range(X_train.shape[1])], top_n=20)
