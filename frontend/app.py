import json
from flask import Flask, render_template, request, jsonify
import os

# Directly import from backend as it's now within the frontend directory
from backend.ChatGPT import parse_eml, generate_prompt, query_llm, extract_label_from_response

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

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        method = request.form.get('method')

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

            if result not in ["Legitimate", "Phishing"]:
                result = "Unknown"  # Fallback to a default if extraction fails
            analysis = response if isinstance(response, str) else response.get('analysis',
                                                                               'No detailed analysis available.')

        # Render the results page with the result and detailed analysis
        return render_template('results.html', result=result, analysis=analysis)

    elif request.method == 'GET':
        return render_template('detect.html')

if __name__ == '__main__':
    app.run(debug=True)
