import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

# Load and preprocess the data for predictions (no labels)
def load_data_for_predictions(filepath, expected_features):
    df = pd.read_csv(filepath)

    # Ensure the dataset has the same features as the training dataset
    actual_features = list(df.columns)
    missing_features = list(set(expected_features) - set(actual_features))

    # Fill missing features with 0
    if missing_features:
        for feature in missing_features:
            df[feature] = 0

    # Ensure the features are in the correct order
    df = df[expected_features]

    return df.values

if __name__ == "__main__":
    # Filepath to your predictions dataset (with true labels)
    pred_filepath = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\data\final_testing\AI_legit_phish_pred\predictions.csv'

    # Expected feature columns (make sure this matches your training data)
    expected_features = [f'term_{i}' for i in range(500)]

    # Load the prediction dataset
    X_pred = load_data_for_predictions(pred_filepath, expected_features)

    # Load the pre-trained models
    rf_model = joblib.load('trained_models/random_forest_model.pkl')
    xgb_model = joblib.load('trained_models/xgboost_model.pkl')
    nb_model = joblib.load('trained_models/naive_bayes_model.pkl')

    # Get probability predictions from each model (soft voting)
    rf_proba = rf_model.predict_proba(X_pred)
    xgb_proba = xgb_model.predict_proba(X_pred)
    nb_proba = nb_model.predict_proba(X_pred)

    # Combine the predictions by averaging the probabilities (soft voting)
    avg_proba = (rf_proba + xgb_proba + nb_proba) / 3

    # Final prediction is the class with the highest average probability
    y_pred = np.argmax(avg_proba, axis=1)

    # Load the true labels from the prediction dataset
    true_labels = pd.read_csv(pred_filepath)['label'].values

    # Calculate accuracy and display the classification report
    accuracy = accuracy_score(true_labels, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    print("Classification Report:")
    print(classification_report(true_labels, y_pred))



# Chatgpt improvement suggestions:
# The improved accuracy of 69.84% and balanced precision and recall across both classes show that the soft voting ensemble is providing better results than hard voting. Here's a breakdown of what the report suggests:
# Analysis of Results:
#
#     Balanced Precision and Recall:
#         The model is performing almost equally well for both classes (phishing and legitimate emails), with a good balance between precision and recall for both.
#         Class 0 (Legitimate Emails): Precision = 0.65, Recall = 0.75
#         Class 1 (Phishing Emails): Precision = 0.75, Recall = 0.65
#         This balance shows that the ensemble model is now predicting both phishing and legitimate emails with a good balance between false positives and false negatives.
#
#     Overall Accuracy:
#         With an accuracy of nearly 70%, the ensemble model is providing better generalization than any individual model alone.
#
# Further Improvement Suggestions:
#
#     Threshold Tuning:
#         You can experiment with adjusting the decision threshold for phishing emails. By default, the threshold is 0.5, but lowering or raising it might improve recall for phishing without sacrificing too much precision.
#
#     Feature Selection:
#         Consider performing additional feature selection (such as focusing on the top features as indicated by Random Forest's feature importance) to simplify the model and potentially boost performance.
#
#     Fine-Tune Hyperparameters:
#         Further fine-tuning of hyperparameters could slightly improve the performance. You can experiment with a more granular grid search for each model, especially focusing on XGBoost and Random Forest.
#
#     Stacking:
#         If you're looking for further improvement, you could try stacking the models instead of voting. Stacking uses another model (meta-model) to learn from the predictions of your base models, potentially boosting performance.
#
# Would you like to try tuning thresholds or experiment with stacking, or are you satisfied with the current performance? Let me know how you'd like to proceed!