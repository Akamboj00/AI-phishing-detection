import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
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
    Trains the Naive Bayes model and evaluates its performance.
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
    print(f"Naive Bayes model saved to {model_path}")

if __name__ == "__main__":
    filepath = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\output\AI_and_legit.csv'
    model_path = 'trained_models/naive_bayes_model.pkl'

    # Load data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)

    # Initialize Naive Bayes and train
    nb_model = GaussianNB()
    train_and_evaluate_model(nb_model, X_train, X_test, y_train, y_test, model_path)
