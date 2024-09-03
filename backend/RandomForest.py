import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_and_preprocess_data(filepath):
    # Load the dataset (already contains TF-IDF features)
    df = pd.read_csv(filepath)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Separate features and labels
    X = df.drop('label', axis=1).values  # Assumes all columns except 'label' are features
    y = df['label']

    return train_test_split(X, y, test_size=0.3, random_state=42)

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_path):
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f}")

    # Train the model on the entire training set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Save the trained model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Evaluate the model on the test set
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Set Accuracy: {accuracy:.4f}")

    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    print("Classification Report:\n", df_report)

    # Plotting the metrics
    metrics = ['precision', 'recall', 'f1-score']

    plt.figure(figsize=(12, 6))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        sns.barplot(x=df_report.index[:-3], y=metric, data=df_report[:-3])
        plt.title(f'{metric.capitalize()}')
        plt.ylim(0, 1)
        plt.xlabel('Class')
        plt.ylabel(metric.capitalize())
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    filepath = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\output\combined_email_features.csv'  # Provide the correct path to your dataset
    model_path = 'trained_models/random_forest_model.pkl'

    # Load, preprocess data, and split into training and test sets
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)

    # Initialize and train the RandomForest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    train_and_evaluate_model(rf_model, X_train, X_test, y_train, y_test, model_path)
