import json
import os
from email import policy
from email.parser import BytesParser
from unittest.mock import Mock
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Initialize OpenAI client
client = OpenAI()

# Mocking the chat completions.create method
client.chat.completions.create = Mock(return_value={
    "choices": [
        {
            "message": {
                "content": "This is a simulated response. The email appears to be a phishing attempt because it uses urgent language and requests personal information."
            }
        }
    ]
})

def query_llm(prompt):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an email spam detector."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion["choices"][0]["message"]["content"]

def generate_prompt(headers, body):
    prompt_template = """
    I want you to act as a spam detector to determine whether a given email is a phishing email or a legitimate email. Your analysis should be thorough and evidence-based. Phishing emails often impersonate legitimate brands and use social engineering techniques to deceive users. These techniques include, but are not limited to: fake rewards, fake warnings about account problems, and creating a sense of urgency or interest. Spoofing the sender address and embedding deceptive HTML links are also common tactics. Analyze the email by following these steps:

    1. Identify any impersonation of well-known brands.

    2. Examine the email header for spoofing signs, such as discrepancies in the sender name or email address. Evaluate the subject line for typical phishing characteristics (e.g., urgency, promise of reward). Note that the To address has been replaced with a dummy address.

    3. Analyze the email body for social engineering tactics designed to induce clicks on hyperlinks. Inspect URLs to determine if they are misleading or lead to suspicious websites.

    4. Provide a comprehensive evaluation of the email, highlighting specific elements that support your conclusion. Include a detailed explanation of any phishing or legitimacy indicators found in the email.

    5. Summarize your findings and provide your final verdict on the legitimacy of the email, supported by the evidence you gathered.

    Email:
    '''{headers}

    {body}'''
    """
    return prompt_template.format(headers=json.dumps(headers, indent=2), body=body)

def parse_eml(file_path):
    with open(file_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    headers = dict(msg.items())
    body = msg.get_body(preferencelist=('plain', 'html')).get_content()
    return headers, body

def extract_label_from_response(response):
    # A simple check to determine if the response indicates phishing or legitimate
    if "phishing" in response.lower():
        return "phishing"
    else:
        return "legitimate"

def process_eml_files_in_directory(directory_path, label, results, start_count, true_labels, predicted_labels):
    count = start_count  # Start the counter from the provided start_count value
    print(f"Processing directory: {directory_path}")
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        print(f"Processing file: {file_path}")

        try:
            headers, body = parse_eml(file_path)
            prompt = generate_prompt(headers, body)
            response = query_llm(prompt)
            predicted_label = extract_label_from_response(response)
            true_labels.append(label)
            predicted_labels.append(predicted_label)
            results.append({
                "result_id": count,  # Add the counter as the result_id
                "file": filename,
                "label": label,
                "predicted_label": predicted_label,
                "response": response
            })
            count += 1  # Increment the counter after each file
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            results.append({
                "result_id": count,  # Add the counter even in case of an error
                "file": filename,
                "label": label,
                "predicted_label": "error",
                "error": str(e)
            })
            count += 1  # Increment the counter even if there's an error

    return count  # Return the current count

def calculate_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, pos_label="phishing")
    recall = recall_score(true_labels, predicted_labels, pos_label="phishing")
    f1 = f1_score(true_labels, predicted_labels, pos_label="phishing")
    report = classification_report(true_labels, predicted_labels, labels=["phishing", "legitimate"])
    return accuracy, precision, recall, f1, report

def main():
    phishing_dir = r'C:\Users\abhil\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Data\testing\phishing_pot-main\combined\phishing_emails'
    legitimate_dir = r'C:\Users\abhil\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Data\testing\phishing_pot-main\combined\legitimate_emails'

    # Output file to save results
    output_file = r'C:\Users\abhil\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\Output\testing\results.json'

    # Initialize results list and label lists
    results = []
    true_labels = []
    predicted_labels = []

    # Process phishing emails
    count = process_eml_files_in_directory(phishing_dir, label="phishing", results=results, start_count=1, true_labels=true_labels, predicted_labels=predicted_labels)

    # Process legitimate emails
    process_eml_files_in_directory(legitimate_dir, label="legitimate", results=results, start_count=count, true_labels=true_labels, predicted_labels=predicted_labels)

    # Calculate metrics
    accuracy, precision, recall, f1, report = calculate_metrics(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("\nClassification Report:\n", report)

    # Save results to the output file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()