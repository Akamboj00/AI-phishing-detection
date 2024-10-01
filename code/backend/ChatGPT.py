from openai import OpenAI
import json
import os
from email import policy
from email.parser import BytesParser
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import re

# Initialize OpenAI client for querying the GPT model
client = OpenAI()


# Function to query the OpenAI model with a prompt
def query_llm(prompt):
    # Calls the OpenAI GPT model with a system message indicating it is an email spam detector
    completion = client.chat.completions.create(
        model="gpt-4o",  # Using the GPT-4o model for email spam detection
        messages=[
            {"role": "system", "content": "You are an email spam detector."},
            {"role": "user", "content": prompt}
        ]
    )
    # Return the content of the model's response
    return completion.choices[0].message.content



def generate_prompt(headers, body): # Generates a prompt to be used for the model based on email headers and body
    prompt_template = """
    I want you to act as a spam detector to determine whether a given email is a phishing email or a legitimate email. Your analysis should be thorough and evidence-based. Phishing emails often impersonate legitimate brands and use social engineering techniques to deceive users. These techniques include, but are not limited to: fake rewards, fake warnings about account problems, and creating a sense of urgency or interest. Spoofing the sender address and embedding deceptive HTML links are also common tactics. 
    Here are the details of the email:

    Email Headers:
    {headers}

    Email Body:
    {body}

    Analyze the email by following these steps:

    1. Identify any impersonation of well-known brands.

    2. Examine the email header for spoofing signs, such as discrepancies in the sender name or email address. Evaluate the subject line for typical phishing characteristics (e.g., urgency, promise of reward). Note that the To address has been replaced with a dummy address.

    3. Analyze the email body for social engineering tactics designed to induce clicks on hyperlinks. Inspect URLs to determine if they are misleading or lead to suspicious websites.

    4. Provide a comprehensive evaluation of the email, highlighting specific elements that support your conclusion. Include a detailed explanation of any phishing or legitimacy indicators found in the email.

    5. Summarize your findings and provide your final verdict on the legitimacy of the email, supported by the evidence you gathered. Return the result using the following format:

    {{
        "is_phishing": 1, # Use 1 if the email is phishing, 0 if it is legitimate
        "analysis": "Provide a brief explanation of your decision here."
    }}
    """
    # Format the prompt with the given email headers and body content
    return prompt_template.format(headers=json.dumps(headers, indent=2), body=body)


# Function to parse .eml files and extract the headers and body of the email
def parse_eml(file_path):
    with open(file_path, 'rb') as f:
        # Parse the email file with default policy
        msg = BytesParser(policy=policy.default).parse(f)
    # Convert the headers into a dictionary and get the email body (preferring plain text or HTML)
    headers = dict(msg.items())
    body = msg.get_body(preferencelist=('plain', 'html')).get_content()
    return headers, body


# Function to extract label from the GPT model response
def extract_label_from_response(response):
    # Check if the model response contains '"is_phishing": 1', indicating a phishing email
    if '"is_phishing": 1' in response:
        return "phishing"
    else:
        return "legitimate"


# Function to process all .eml files in a directory and get true and predicted labels
def process_eml_files_in_directory(directory_path, label, results, start_count, true_labels, predicted_labels):
    count = start_count  # Start file counter from the provided start count
    print(f"Processing directory: {directory_path}")
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)  # Construct full file path
        print(f"Processing file: {file_path}")

        try:
            # Parse the email and generate the prompt for the GPT model
            headers, body = parse_eml(file_path)
            prompt = generate_prompt(headers, body)
            response = query_llm(prompt)  # Query the model with the generated prompt
            predicted_label = extract_label_from_response(response)  # Extract label from model response

            # Append true and predicted labels for evaluation
            true_labels.append(label)
            predicted_labels.append(predicted_label)
            results.append({
                "result_id": count,  # Assign result ID based on the file counter
                "file": filename,
                "label": label,  # True label (phishing or legitimate)
                "predicted_label": predicted_label,  # Predicted label from model
                "response": response  # Full model response
            })
            count += 1  # Increment file counter after each processed file

        # Handle JSON decode errors in case of invalid responses
        except json.decoder.JSONDecodeError as json_error:
            print(f"JSON decode error in {file_path}: {json_error}. Skipping....")

        # General exception handling for file processing errors
        except Exception as e:
            print(f"Failed to process {file_path}: {e} Skipping...")
            results.append({
                "result_id": count,  # Even in case of error, increment counter
                "file": filename,
                "label": label,
                "predicted_label": "error",
                "error": str(e)  # Log the error message
            })
            count += 1  # Increment file counter even in case of an error

    return count  # Return the updated file counter


# Function to calculate accuracy, precision, recall, F1-score, and classification report
def calculate_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)  # Accuracy calculation
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)  # Precision
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)  # Recall
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)  # F1-Score
    # Generate detailed classification report
    report = classification_report(true_labels, predicted_labels, labels=["phishing", "legitimate"], zero_division=1)
    return accuracy, precision, recall, f1, report


# Main function to handle the workflow for phishing email detection
def main():
    # Directory containing phishing emails
    phishing_dir = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\data\final_testing\AI_legit_phish_pred\phishing_emails'

    # Directory containing legitimate emails
    legitimate_dir = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\data\final_testing\AI_legit_phish_pred\legitimate_emails'

    # Output file to save the results of the analysis
    output_file = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\output\testing\results.json'

    # Initialize lists to store results, true labels, and predicted labels
    results = []
    true_labels = []
    predicted_labels = []

    # Process phishing emails and keep track of the file counter
    count = process_eml_files_in_directory(phishing_dir, label="phishing", results=results, start_count=1,
                                           true_labels=true_labels, predicted_labels=predicted_labels)

    # Process legitimate emails, starting the count from where phishing email processing stopped
    process_eml_files_in_directory(legitimate_dir, label="legitimate", results=results, start_count=count,
                                   true_labels=true_labels, predicted_labels=predicted_labels)

    # Calculate performance metrics
    accuracy, precision, recall, f1, report = calculate_metrics(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("\nClassification Report:\n", report)

    # Save the detailed results to the output file in JSON format
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


# Entry point of the script
if __name__ == "__main__":
    main()
