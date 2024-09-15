import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import joblib


# Load and preprocess the data
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop_duplicates()

    # Use the existing TF-IDF feature columns directly
    X = df.drop('label', axis=1).values
    y = df['label'].values
    return train_test_split(X, y, test_size=0.3, random_state=42)


if __name__ == "__main__":
    # Filepath to your dataset
    filepath = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\data\final_testing\AI_legit_phish_train\training.csv'
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)

    # Initialize Naive Bayes model
    nb_model = GaussianNB()

    # Cross-validation to check the consistency of model performance
    scores = cross_val_score(nb_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy scores: {scores}")
    print(f"Mean cross-validation accuracy: {scores.mean()}")

    # Train the model
    nb_model.fit(X_train, y_train)

    # Make predictions
    y_pred = nb_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(nb_model, 'trained_models/naive_bayes_model.pkl')
