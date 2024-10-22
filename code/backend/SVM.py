import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
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

    # Initialize SVM model
    svm_model = SVC(probability=True, random_state=42)

    # Perform grid search to find best hyperparameters
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    }
    grid_search = GridSearchCV(svm_model, param_grid, cv=5, verbose=1)
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters
    print(f"Best Hyperparameters: {grid_search.best_params_}")

    # Use the best model to make predictions
    best_svm_model = grid_search.best_estimator_
    y_pred = best_svm_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(best_svm_model, 'trained_models/svm_model.pkl')
