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

    'xgboost': r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\code\backend\trained_models\xgboost_model.pkl'
}
VECTORIZER_PATH = 'backend/trained_models/vectorizer.pkl'


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
            headers, body = parse_eml(eml_file_path)  # Assuming parse_eml is available
        else:
            # Use manual entry data
            headers = {
                "Sender": request.form.get('sender'),
                "Subject": request.form.get('subject')
            }
            body = request.form.get('email_body')

        if method == 'chatgpt-model':
            prompt = generate_prompt(headers, body)
            response = query_llm(prompt)
            print(f"Raw Model Response: {response}")  # Log the raw response for debugging

            result = extract_label_from_response(response)
            print(f"Extracted Result: {result}")  # Log the extracted result for verification
            analysis = response if isinstance(response, str) else response.get('analysis', 'No detailed analysis available.')

        # ML Model-based Detection
        elif method in ['random-forest', 'naive-bayes', 'xgboost'] and eml_file:
            # Initialize the EmailFeatureExtractor
            extractor = EmailFeatureExtractor()

            # Process the uploaded email and extract its features using the saved vectorizer
            email_features_df = extractor.process_single_email(eml_file_path, VECTORIZER_PATH)

            # Define the expected features (500 features in total)
            expected_features = [f'term_{i}' for i in range(500)]

            # Ensure the extracted features match the expected 500 features
            single_email_features = prepare_features_for_prediction(email_features_df, expected_features)

            # Load the selected ML model
            model = load_model(MODEL_PATHS[method.replace('-', '_')])

            # Make the prediction
            prediction = predict_single_email(model, single_email_features)

            # Determine if it's phishing or legitimate based on the prediction
            if prediction[0] == 1:
                result = "Phishing"
            else:
                result = "Legitimate"

            analysis = f"The email was classified as {result}."

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
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])  # Assuming class 1 is "phishing"
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
    # Assuming you have a shared load_and_preprocess_data method
    filepath = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\output\AI_and_legit.csv'
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)

    # Load models and generate metrics & visualizations
    model_paths = {
        'random_forest': 'backend/trained_models/random_forest_model.pkl',
        'naive_bayes': 'backend/trained_models/naive_bayes_model.pkl',
        'xgboost': 'backend/trained_models/xgboost_model.pkl'
    }

    metrics = {}
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

    return render_template('metrics.html', metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True)
