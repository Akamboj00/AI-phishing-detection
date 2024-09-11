import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report


def load_model(model_path):
    return joblib.load(model_path)


def prepare_features_for_prediction(email_df, expected_features):
    actual_features = list(email_df.columns)
    missing_features = list(set(expected_features) - set(actual_features))
    if missing_features:
        missing_df = pd.DataFrame(0, index=email_df.index, columns=missing_features)
        email_df = pd.concat([email_df, missing_df], axis=1)
    email_df = email_df[expected_features]
    return email_df.values


def predict_emails(model, email_features):
    return model.predict(email_features)


def predict_single_email(model, single_email_features):
    prediction = model.predict(single_email_features)
    return prediction


if __name__ == "__main__":
    models = {
        'random_forest': 'trained_models/random_forest_model.pkl',
        'naive_bayes': 'trained_models/naive_bayes_model.pkl',
        'xgboost': 'trained_models/xgboost_model.pkl'
    }

    email_csv_path = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\data\final_testing\AI_legit_phish_pred\predictions.csv'

    email_df = pd.read_csv(email_csv_path)

    expected_features = [f'term_{i}' for i in range(500)]
    X = prepare_features_for_prediction(email_df.drop('label', axis=1), expected_features)
    y_true = email_df['label'].values

    # Print columns and sample data
    print("Expected feature columns:")
    print(expected_features)
    print("Actual feature columns in email_df:")
    print(email_df.columns)

    print("Sample feature values from X:")
    print(X[:5])  # Print the first 5 rows

    for model_name, model_path in models.items():
        model = load_model(model_path)
        print(f"Model {model_name} loaded successfully.")

        try:
            y_pred = predict_emails(model, X)
            print(f"Predictions made successfully for {model_name}.")
            print(f"Sample predictions: {y_pred[:5]}")  # Print the first 5 predictions

            accuracy = accuracy_score(y_true, y_pred)
            print(f"\nPerformance Evaluation for {model_name.capitalize()}:")
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred))
        except Exception as e:
            print(f"Error making predictions with {model_name}: {e}")


# if __name__ == "__main__":
#     model_choice = 'random_forest'  # Can be 'random_forest', 'naive_bayes', or 'xgboost'
#     #model_choice = 'naive_bayes'
#     #model_choice = 'xgboost'
#
#     if model_choice == 'random_forest':
#         model_path = 'trained_models/random_forest_model.pkl'
#     elif model_choice == 'naive_bayes':
#         model_path = 'trained_models/naive_bayes_model.pkl'
#     elif model_choice == 'xgboost':
#         model_path = 'trained_models/xgboost_model.pkl'
#
#     single_email_csv_path = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\data\single_tests\single4.csv'
#
#     model = load_model(model_path)
#     single_email_df = pd.read_csv(single_email_csv_path)
#
#     expected_features = [f'term_{i}' for i in range(500)]
#     single_email_features = prepare_features_for_prediction(single_email_df, expected_features)
#
#     prediction = predict_single_email(model, single_email_features)
#
#     if prediction[0] == 1:
#         print("The email is predicted to be phishing.")
#     else:
#         print("The email is predicted to be legitimate.")