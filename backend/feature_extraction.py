from email import policy
from email.parser import BytesParser
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os

class EmailFeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def parse_eml(self, file_path):
        with open(file_path, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)

        headers = dict(msg.items())
        body = ""

        if msg.is_multipart():
            # Iterate through parts and extract content based on type
            for part in msg.iter_parts():
                content_type = part.get_content_type()
                if content_type == 'text/plain':
                    body = part.get_content().strip()  # Plain text preferred
                    break  # Stop at the first plain text part found
                elif content_type == 'text/html':
                    body = part.get_content().strip()  # Fallback to HTML if no plain text
        else:
            # Handle non-multipart emails
            body = msg.get_body(preferencelist=('plain', 'html')).get_content().strip()

        return headers, body

    def extract_tfidf_features(self, corpus):
        """
        Extracts TF-IDF features from a list of email bodies.
        :param corpus: A list of email bodies.
        :return: TF-IDF feature matrix.
        """
        if not corpus:
            raise ValueError("The corpus is empty. Ensure there are valid email bodies to process.")
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        return tfidf_matrix

    def process_directory(self, directory_path, label):
        """
        Processes all files in a directory, extracts TF-IDF features, and adds a label.
        :param directory_path: Path to the directory containing email-like files.
        :param label: The label to assign to each email (1 for phishing, 0 for legitimate).
        :return: DataFrame containing TF-IDF features and labels for all emails in the directory.
        """
        email_bodies = []
        labels = []

        # Iterate over all files in the directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                headers, body = self.parse_eml(file_path)
                if body:  # Only include non-empty bodies
                    email_bodies.append(body)
                    labels.append(label)
                else:
                    print(f"Warning: Empty email body in file {filename}. Skipping...")
            except Exception as e:
                print(f"Error processing file {filename}: {e}. Skipping...")

        if not email_bodies:
            raise ValueError("No valid email bodies found in the directory.")

        # Extract TF-IDF features for all email bodies
        tfidf_features = self.extract_tfidf_features(email_bodies)
        dense_features = tfidf_features.toarray()

        # Convert to DataFrame and add labels
        df = pd.DataFrame(dense_features, columns=[f'term_{i}' for i in range(dense_features.shape[1])])
        df['label'] = labels
        return df

    def save_combined_csv(self, phishing_dir, legitimate_dir, csv_file_path):
        """
        Processes both phishing and legitimate email directories and saves the combined features to a CSV file.
        :param phishing_dir: Path to the directory containing phishing emails.
        :param legitimate_dir: Path to the directory containing legitimate emails.
        :param csv_file_path: Path to the output CSV file.
        """
        phishing_df = self.process_directory(phishing_dir, label=1)
        legitimate_df = self.process_directory(legitimate_dir, label=0)

        # Combine both dataframes
        combined_df = pd.concat([phishing_df, legitimate_df], ignore_index=True)

        # Save to CSV
        combined_df.to_csv(csv_file_path, index=False)
        print(f"Combined TF-IDF features and labels saved to {csv_file_path}")

# Example usage
if __name__ == "__main__":
    phishing_dir = r'C:\Users\abhil\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\data\testing_datasets\combined_spam_ham_eml\phishing_emails'

    legitimate_dir = r'C:\Users\abhil\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\data\testing_datasets\combined_spam_ham_eml\legitimate_emails'

    output_csv = r'C:\Users\abhil\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\output\combined_email_features.csv'

    extractor = EmailFeatureExtractor()
    extractor.save_combined_csv(phishing_dir, legitimate_dir, output_csv)
