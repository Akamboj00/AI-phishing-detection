import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# Function to load all models once and reuse them
def load_models():
    rf_model = joblib.load(r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\code\backend\trained_models\random_forest_model.pkl')
    xgb_model = joblib.load(r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\code\backend\trained_models\xgboost_model.pkl')
    nb_model = joblib.load(r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\code\backend\trained_models\naive_bayes_model.pkl')
    svm_model = joblib.load(r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\code\backend\trained_models\svm_model.pkl')
    return rf_model, xgb_model, nb_model, svm_model


# Load and preprocess the data
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop_duplicates()

    # Use the existing TF-IDF feature columns directly
    X = df.drop('label', axis=1).values
    y = df['label'].values
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Helper function to process data for predictions (used for both single and batch)
def prepare_features(single_or_batch_data, expected_features):
    # If it's already a DataFrame or 2D numpy array, don't try to wrap it again
    if isinstance(single_or_batch_data, pd.DataFrame):
        df = single_or_batch_data
    else:
        # If it's a 2D array, assume it's the batch data and don't wrap in a list
        if np.ndim(single_or_batch_data) == 2:
            df = pd.DataFrame(single_or_batch_data, columns=expected_features)
        else:
            # For a single row of data, wrap it
            df = pd.DataFrame([single_or_batch_data], columns=expected_features)

    # Fill missing features with 0
    missing_features = list(set(expected_features) - set(df.columns))
    if missing_features:
        for feature in missing_features:
            df[feature] = 0

    # Ensure the features are in the correct order
    df = df[expected_features]
    return df.values


# Function to handle ensemble prediction (same logic for both single and batch)
def ensemble_predict(features, models):
    rf_model, xgb_model, nb_model, svm_model = models

    # Get probability predictions from each model
    rf_pred_prob = rf_model.predict_proba(features)[:, 1]
    xgb_pred_prob = xgb_model.predict_proba(features)[:, 1]
    nb_pred_prob = nb_model.predict_proba(features)[:, 1]
    svm_pred_prob = svm_model.decision_function(features)


    # Weights for the models
    weights = {
        'rf': 0.25,
        'xgb': 0.5,
        'nb': 0.1,
        'svm': 0.5,
    }

    # Adjust threshold to 0.6
    threshold = 0.6

    # Combine the predictions using weighted average (soft voting)
    combined_prob = (weights['rf'] * rf_pred_prob +
                     weights['xgb'] * xgb_pred_prob +
                     weights['nb'] * nb_pred_prob +
                     weights['svm'] * svm_pred_prob)

    # Convert probabilities to binary predictions using the adjusted threshold
    predictions = (combined_prob > threshold).astype(int)
    return predictions


def batch_prediction(filepath, expected_features, models):
    # Use the shared function to load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)

    # Prepare the features for prediction
    X_pred = prepare_features(X_test, expected_features)

    # Get predictions from the ensemble model
    y_pred = ensemble_predict(X_pred, models)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)



# Single email prediction function for frontend (now from CSV)
def single_email_prediction_from_csv(csv_filepath, expected_features, models):
    # Load the features for a single email from a CSV file
    df = pd.read_csv(csv_filepath)

    # Prepare the single email features
    email_features_prepared = prepare_features(df.iloc[0], expected_features)

    # Get prediction from the ensemble model
    prediction = ensemble_predict(email_features_prepared, models)

    # Return phishing (1) or legitimate (0) result
    return prediction[0]  # Since it's a single email, return the first (and only) prediction


if __name__ == "__main__":
    # Expected features (ensure this matches your training data)
    expected_features = [f'term_{i}' for i in range(500)]

    # Load models only once
    models = load_models()

    # Example: Batch prediction for testing
    batch_filepath = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\data\final_testing\AI_legit_phish_pred\predictions.csv'
    # Uncomment the line below for batch prediction
    batch_prediction(batch_filepath, expected_features, models)

    # Example: Single email prediction from CSV (for frontend use case)
    single_email_filepath = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\data\single_tests\single3.csv'
    #prediction = single_email_prediction_from_csv(single_email_filepath, expected_features, models)
    #print(f"Single email prediction: {'Phishing' if prediction == 1 else 'Legitimate'}")
