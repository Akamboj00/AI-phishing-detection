from email import policy
from email.parser import BytesParser
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os

class EmailFeatureExtractor:
    def __init__(self):
        # Initialize a new TF-IDF vectorizer without loading from a file
        self.vectorizer = TfidfVectorizer(max_features=500)

    def parse_eml(self, file_path):
        with open(file_path, 'rb') as f:
            try:
                msg = BytesParser(policy=policy.default).parse(f)
            except Exception as e:
                print(f"Error parsing email file {file_path}: {e}")
                return {}, ""

        headers = dict(msg.items())
        body = ""

        try:
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    try:
                        if content_type == 'text/plain' or content_type == 'text/html':
                            part_body = part.get_content().strip()
                            if part_body:
                                body = part_body  # preferring the last text part found
                    except Exception as e:
                        print(f"Error processing part of type {content_type} in file {file_path}: {e}")
            else:
                body = msg.get_body(preferencelist=('plain', 'html')).get_content().strip()
        except Exception as e:
            print(f"Error processing email body in file {file_path}: {e}")

        if not body:
            print(f"Warning: No body content found in file {file_path}.")

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

    def extract_tfidf_features_single_email(self, email_body):
        """
        Extract TF-IDF features for a single email body.
        :param email_body: The body of the email.
        :return: DataFrame containing TF-IDF features for the email.
        """
        if not email_body:
            raise ValueError("The email body is empty. Ensure there is valid email content to process.")

        # Transform using the vectorizer (fit and transform for a single instance)
        tfidf_features = self.vectorizer.transform([email_body])
        dense_features = tfidf_features.toarray()

        # Convert to DataFrame
        df = pd.DataFrame(dense_features, columns=[f'term_{i}' for i in range(dense_features.shape[1])])
        return df

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

    def extract_features_without_label(self, directory_path):
        email_bodies = []

        # Iterate over all files in the directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                headers, body = self.parse_eml(file_path)
                if body:  # Only include non-empty bodies
                    email_bodies.append(body)
            except Exception as e:
                print(f"Error processing file {filename}: {e}. Skipping...")

        if not email_bodies:
            raise ValueError("No valid email bodies found in the directory.")

        # Extract TF-IDF features for all email bodies
        tfidf_features = self.extract_tfidf_features(email_bodies)
        dense_features = tfidf_features.toarray()

        # Convert to DataFrame
        df = pd.DataFrame(dense_features, columns=[f'term_{i}' for i in range(dense_features.shape[1])])
        return df

    def process_single_email(self, file_path):
        """
        Process a single email file to extract its TF-IDF features.
        :param file_path: Path to the email file.
        :return: DataFrame containing TF-IDF features.
        """
        headers, body = self.parse_eml(file_path)
        if not body:
            raise ValueError("The email body is empty. Ensure there is valid email content to process.")

        # Fit and transform the vectorizer with the single email body
        tfidf_features = self.vectorizer.fit_transform([body])
        dense_features = tfidf_features.toarray()

        # Convert to DataFrame
        df = pd.DataFrame(dense_features, columns=[f'term_{i}' for i in range(dense_features.shape[1])])
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


if __name__ == "__main__":
    # Paths for directories used in combined CSV creation
    # phishing_dir = r'path\to\phishing\emails'
    # legitimate_dir = r'path\to\legitimate\emails'
    # output_csv = r'path\to\output\combined_email_features.csv'

    # Initialize the feature extractor
    extractor = EmailFeatureExtractor()

    # Create the combined CSV for training data
    # extractor.save_combined_csv(phishing_dir, legitimate_dir, output_csv)

    # Path for the single email to be processed
    single_email_path = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\data\testing_datasets\combined_spam_ham_eml\phishing_emails\sample-10.eml'

    # Path to save the TF-IDF features from the single email
    single_email_output_csv = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\backend\temp_email_features.csv'

    # Process the single email and extract features
    single_email_features_df = extractor.process_single_email(single_email_path)

    # Save the features to the specified CSV file
    single_email_features_df.to_csv(single_email_output_csv, index=False)
    print(f"TF-IDF features for the single email saved to {single_email_output_csv}")
