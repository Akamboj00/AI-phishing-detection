import json
from flask import Flask, render_template, request, jsonify
import os
import joblib
from backend.ChatGPT import parse_eml, generate_prompt, query_llm, extract_label_from_response
from backend.feature_extraction import EmailFeatureExtractor
from backend.run_test import load_model, predict_single_email, prepare_features_for_prediction


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
    'random_forest': 'backend/trained_models/random_forest_model.pkl',
    'naive_bayes': 'backend/trained_models/naive_bayes_model.pkl',
    'xgboost': 'backend/trained_models/xgboost_model.pkl'
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

if __name__ == '__main__':
    app.run(debug=True)
