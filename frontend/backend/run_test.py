import pandas as pd
import joblib
import numpy as np

def load_model_and_vectorizer(model_path):
    # Load the trained RandomForest model
    model = joblib.load(model_path)
    return model

def prepare_features_for_prediction(single_email_df, expected_features):
    # Get the actual features in the single email's TF-IDF dataset
    actual_features = list(single_email_df.columns)

    # Identify missing features
    missing_features = list(set(expected_features) - set(actual_features))

    # Create a DataFrame with missing features filled with zeros
    if missing_features:
        missing_df = pd.DataFrame(0, index=single_email_df.index, columns=missing_features)
        single_email_df = pd.concat([single_email_df, missing_df], axis=1)

    # Ensure columns are in the correct order as expected by the model
    single_email_df = single_email_df[expected_features]

    return single_email_df.values

def predict_single_email(model, single_email_features):
    # Make prediction
    prediction = model.predict(single_email_features)
    return prediction

if __name__ == "__main__":
    model_path = 'trained_models/random_forest_model.pkl'  # Path to your trained model
    single_email_csv_path = r'/Code/AI-phishing-detection/frontend/backend/temp_email_features.csv'  # Path to the single email's TF-IDF CSV

    # Load the trained model
    model = load_model_and_vectorizer(model_path)

    # Load the single email's TF-IDF features
    single_email_df = pd.read_csv(single_email_csv_path)

    # Expected features as per the trained model
    expected_features = [f'term_{i}' for i in range(500)]  # The model expects exactly 500 features

    # Prepare the features for prediction
    single_email_features = prepare_features_for_prediction(single_email_df, expected_features)

    # Predict
    prediction = predict_single_email(model, single_email_features)

    # Output the prediction
    if prediction[0] == 1:
        print("The email is predicted to be phishing.")
    else:
        print("The email is predicted to be legitimate.")
