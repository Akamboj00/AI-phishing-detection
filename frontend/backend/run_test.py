import pandas as pd
import joblib
import numpy as np


def load_model_and_vectorizer(model_path):
    # Load the trained model
    model = joblib.load(model_path)
    return model


def prepare_features_for_prediction(single_email_df, expected_features):
    actual_features = list(single_email_df.columns)
    missing_features = list(set(expected_features) - set(actual_features))
    if missing_features:
        missing_df = pd.DataFrame(0, index=single_email_df.index, columns=missing_features)
        single_email_df = pd.concat([single_email_df, missing_df], axis=1)
    single_email_df = single_email_df[expected_features]
    return single_email_df.values


def predict_single_email(model, single_email_features):
    prediction = model.predict(single_email_features)
    return prediction


if __name__ == "__main__":
    #model_choice = 'random_forest'  # Can be 'random_forest', 'naive_bayes', or 'xgboost'
    #model_choice = 'naive_bayes'
    model_choice = 'xgboost'

    if model_choice == 'random_forest':
        model_path = 'trained_models/random_forest_model.pkl'
    elif model_choice == 'naive_bayes':
        model_path = 'trained_models/naive_bayes_model.pkl'
    elif model_choice == 'xgboost':
        model_path = 'trained_models/xgboost_model.pkl'

    single_email_csv_path = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\data\single_tests\single5.csv'

    model = load_model_and_vectorizer(model_path)
    single_email_df = pd.read_csv(single_email_csv_path)

    expected_features = [f'term_{i}' for i in range(500)]
    single_email_features = prepare_features_for_prediction(single_email_df, expected_features)

    prediction = predict_single_email(model, single_email_features)

    if prediction[0] == 1:
        print("The email is predicted to be phishing.")
    else:
        print("The email is predicted to be legitimate.")
