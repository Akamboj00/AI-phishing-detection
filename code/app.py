from backend.ChatGPT import parse_eml, generate_prompt, query_llm, extract_label_from_response
from backend.feature_extraction import EmailFeatureExtractor
from backend.run_test import load_model, predict_single_email, prepare_features_for_prediction
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from flask import Flask, render_template, request, send_file
from backend.RandomForest import load_and_preprocess_data
from backend.NaiveBayes import load_and_preprocess_data
from backend.XGBoost import load_and_preprocess_data
from backend.ensemble import ensemble_predict, load_models  # You need to create this function


app = Flask(__name__)

# Define the path for the uploads directory
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train')
def train():
    return render_template('train.html')


# Define the path for the trained models and vectorizer
MODEL_PATHS = {
    'random_forest': r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\code\backend\trained_models\random_forest_model.pkl',

    'naive_bayes': r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\code\backend\trained_models\naive_bayes_model.pkl',

    'xgboost': r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\code\backend\trained_models\xgboost_model.pkl',

    'svm': r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\code\backend\trained_models\svm_model.pkl',
}
VECTORIZER_PATH = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\code\backend\trained_models\vectorizer.pkl'


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        method = request.form.get('method')
        result = "Unknown"
        analysis = "No analysis available."

        # Check if a file was uploaded
        eml_file = request.files.get('eml_file')
        if eml_file:
            # Save the file in the uploads directory
            eml_file_path = os.path.join(UPLOAD_FOLDER, eml_file.filename)
            eml_file.save(eml_file_path)

            # Parse the email and extract headers/body
            headers, body = parse_eml(eml_file_path)
        else:
            # Use manual entry data
            headers = {
                "Sender": request.form.get('sender'),
                "Subject": request.form.get('subject')
            }
            body = request.form.get('email_body')

        # If ChatGPT model is selected
        if method == 'chatgpt-model':
            prompt = generate_prompt(headers, body)
            response = query_llm(prompt)
            result = extract_label_from_response(response)
            analysis = response

        # If Ensemble model is selected
        elif method == 'ensemble-model':
            # Initialize the EmailFeatureExtractor
            extractor = EmailFeatureExtractor()
            email_features_df = extractor.process_single_email(eml_file_path, VECTORIZER_PATH)
            expected_features = [f'term_{i}' for i in range(500)]
            single_email_features = prepare_features_for_prediction(email_features_df, expected_features)
            models = load_models()
            prediction = ensemble_predict(single_email_features, models)

            if prediction == 1:
                result = "Phishing"
            else:
                result = "Legitimate"

            analysis = f"The email was classified as {result} by the ensemble model."

        # If both models combined is selected
        elif method == 'both':
            # Run ChatGPT model
            prompt = generate_prompt(headers, body)
            chatgpt_response = query_llm(prompt)
            chatgpt_result = extract_label_from_response(chatgpt_response)

            # Run Ensemble model
            extractor = EmailFeatureExtractor()
            email_features_df = extractor.process_single_email(eml_file_path, VECTORIZER_PATH)
            expected_features = [f'term_{i}' for i in range(500)]
            single_email_features = prepare_features_for_prediction(email_features_df, expected_features)
            models = load_models()
            ensemble_prediction = ensemble_predict(single_email_features, models)

            ensemble_result = "Phishing" if ensemble_prediction == 1 else "Legitimate"

            # Combine results from both models
            result = f"ChatGPT: {chatgpt_result}\n, Ensemble: {ensemble_result}\n"
            analysis = f"ChatGPT Analysis: {chatgpt_response}\n\nEnsemble Analysis: The email was classified as {ensemble_result}."

        # Render the results page with the result and detailed analysis
        return render_template('results.html', result=result, analysis=analysis)

    elif request.method == 'GET':
        return render_template('detect.html')


# Path for saving charts
CHARTS_FOLDER = os.path.join(os.getcwd(), 'static')
os.makedirs(CHARTS_FOLDER, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))  # Set smaller size for confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing'])
    plt.title(f'{model_name.capitalize()} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(CHARTS_FOLDER, f'confusion_matrix_{model_name}.png'))
    plt.close()

def plot_roc_curve(y_true, y_proba, model_name):
    # Check if y_proba is 1D (some models might only return probability for the positive class)
    if y_proba.ndim == 1:
        # Assuming this is the probability for the positive class (phishing)
        fpr, tpr, _ = roc_curve(y_true, y_proba)
    else:
        # Assuming the second column represents the probability of the positive class (phishing)
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])

    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))  # Set smaller size for ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name.capitalize()} ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(CHARTS_FOLDER, f'roc_curve_{model_name}.png'))
    plt.close()

def plot_feature_importance(model, feature_names, model_name):
    importance = model.feature_importances_
    indices = importance.argsort()[::-1][:20]  # Get top 20 features
    plt.figure(figsize=(8, 4))  # Set smaller size for feature importance
    sns.barplot(x=importance[indices], y=[feature_names[i] for i in indices])
    plt.title(f'{model_name.capitalize()} Feature Importance')
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.savefig(os.path.join(CHARTS_FOLDER, f'feature_importance_{model_name}.png'))
    plt.close()


@app.route('/metrics')
def metrics():
    # Load the dataset for individual models
    training_filepath = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\data\final_testing\AI_legit_phish_train\training.csv'
    X_train, X_test, y_train, y_test = load_and_preprocess_data(training_filepath)

    # Load the dataset for the ensemble model
    predictions_filepath = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\data\final_testing\AI_legit_phish_pred\predictions.csv'
    X_ens_train, X_ens_test, y_ens_train, y_ens_test = load_and_preprocess_data(predictions_filepath)

    # Load individual model paths
    model_paths = {
        'random_forest': r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\code\backend\trained_models\random_forest_model.pkl',
        'naive_bayes': r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\code\backend\trained_models\naive_bayes_model.pkl',
        'xgboost': r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\code\backend\trained_models\xgboost_model.pkl',
        'svm': r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\code\backend\trained_models\svm_model.pkl',
    }

    # Dictionary to store metrics
    metrics = {}

    # Generate metrics and visualizations for individual models
    for model_name, model_path in model_paths.items():
        model = load_model(model_path)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics[model_name] = report

        # ROC curve
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            plot_roc_curve(y_test, y_proba, model_name)

        # Confusion Matrix
        plot_confusion_matrix(y_test, y_pred, model_name)

        # Feature Importance (for models that support it)
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(model, [f'term_{i}' for i in range(X_train.shape[1])], model_name)

    # Generate metrics for the ensemble model using the predictions_filepath dataset
    models = load_models()  # Load models needed for the ensemble
    y_pred_ensemble = ensemble_predict(X_ens_test, models)
    report_ensemble = classification_report(y_ens_test, y_pred_ensemble, output_dict=True)
    metrics['ensemble'] = report_ensemble

    # Confusion Matrix for the ensemble model
    plot_confusion_matrix(y_ens_test, y_pred_ensemble, 'ensemble')

    # ROC curve for the ensemble model
    plot_roc_curve(y_ens_test, y_pred_ensemble, 'ensemble')

    # Hardcoded ChatGPT model metrics (from results.json)
    chatgpt_metrics = {
        "accuracy": 1.00,
        "precision": 1.00,
        "recall": 1.00,
        "f1-score": 1.00
    }
    metrics['chatgpt'] = {
        'accuracy': chatgpt_metrics['accuracy'],
        '0': {
            'precision': chatgpt_metrics['precision'],
            'recall': chatgpt_metrics['recall'],
            'f1-score': chatgpt_metrics['f1-score'],
            'support': 260
        },
        '1': {
            'precision': chatgpt_metrics['precision'],
            'recall': chatgpt_metrics['recall'],
            'f1-score': chatgpt_metrics['f1-score'],
            'support': 296
        }
    }

    return render_template('metrics.html', metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True)
