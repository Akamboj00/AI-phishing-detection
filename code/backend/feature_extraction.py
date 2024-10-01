from email import policy
from email.parser import BytesParser
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import joblib


class EmailFeatureExtractor:
    def __init__(self):
        # Initialize a new TF-IDF vectorizer with a maximum of 500 features
        # This vectorizer will be used to convert email text into a matrix of TF-IDF features
        self.vectorizer = TfidfVectorizer(max_features=500)

    def parse_eml(self, file_path):
        """
        Parses an email file (.eml) and extracts the headers and body content.
        :param file_path: Path to the .eml file.
        :return: Tuple containing email headers as a dictionary and the body content as a string.
        """
        with open(file_path, 'rb') as f:
            try:
                # Parse the email file using BytesParser with the default policy
                msg = BytesParser(policy=policy.default).parse(f)
            except Exception as e:
                # Catch and display any parsing errors
                print(f"Error parsing email file {file_path}: {e}")
                return {}, ""

        # Extract email headers as a dictionary
        headers = dict(msg.items())
        body = ""

        try:
            # Check if the email is multipart (contains multiple parts like text and attachments)
            if msg.is_multipart():
                # Iterate over each part of the email
                for part in msg.walk():
                    content_type = part.get_content_type()
                    try:
                        # If the part is text/plain or text/html, extract the content
                        if content_type == 'text/plain' or content_type == 'text/html':
                            part_body = part.get_content().strip()
                            if part_body:
                                # Update body content (will store the last found text part)
                                body = part_body
                    except Exception as e:
                        # Handle errors in extracting content from specific parts
                        print(f"Error processing part of type {content_type} in file {file_path}: {e}")
            else:
                # If the email is not multipart, directly extract the body with a preference for plain text
                body = msg.get_body(preferencelist=('plain', 'html')).get_content().strip()
        except Exception as e:
            # Handle errors in processing the email body
            print(f"Error processing email body in file {file_path}: {e}")

        # Warn if no body content was found in the email
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
            # Raise an error if the corpus is empty
            raise ValueError("The corpus is empty. Ensure there are valid email bodies to process.")

        # Fit the vectorizer to the corpus and transform it into TF-IDF features
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        return tfidf_matrix

    def extract_tfidf_features_single_email(self, email_body):
        """
        Extract TF-IDF features for a single email body.
        :param email_body: The body of the email.
        :return: DataFrame containing TF-IDF features for the email.
        """
        if not email_body:
            # Raise an error if the email body is empty
            raise ValueError("The email body is empty. Ensure there is valid email content to process.")

        # Transform the single email body using the pre-fitted vectorizer
        tfidf_features = self.vectorizer.transform([email_body])
        dense_features = tfidf_features.toarray()

        # Convert the TF-IDF feature matrix to a DataFrame
        df = pd.DataFrame(dense_features, columns=[f'term_{i}' for i in range(dense_features.shape[1])])
        return df

    def process_directory(self, directory_path, label):
        """
        Processes all email files in a directory, extracts email bodies, and assigns a label.
        :param directory_path: Path to the directory containing email files.
        :param label: The label to assign to each email (1 for phishing, 0 for legitimate).
        :return: DataFrame containing email bodies and labels for all emails in the directory.
        """
        email_bodies = []
        labels = []

        # Iterate over all files in the specified directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                # Parse each email file to extract headers and body
                headers, body = self.parse_eml(file_path)
                if body:  # Only include emails with non-empty bodies
                    email_bodies.append(body)
                    labels.append(label)
                else:
                    # Warn if an email has an empty body
                    print(f"Warning: Empty email body in file {filename}. Skipping...")
            except Exception as e:
                # Handle any errors encountered during file processing
                print(f"Error processing file {filename}: {e}. Skipping...")

        if not email_bodies:
            # Raise an error if no valid email bodies were found
            raise ValueError("No valid email bodies found in the directory.")

        # Create a DataFrame with the email bodies and corresponding labels
        df = pd.DataFrame({
            'body': email_bodies,
            'label': labels
        })

        return df

    def extract_features_without_label(self, directory_path):
        """
        Extracts TF-IDF features from email bodies in a directory without assigning labels.
        :param directory_path: Path to the directory containing email files.
        :return: DataFrame containing the TF-IDF features.
        """
        email_bodies = []

        # Iterate over all files in the directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                # Parse the email file to extract its body
                headers, body = self.parse_eml(file_path)
                if body:  # Only include non-empty bodies
                    email_bodies.append(body)
            except Exception as e:
                # Handle any errors encountered during file processing
                print(f"Error processing file {filename}: {e}. Skipping...")

        if not email_bodies:
            # Raise an error if no valid email bodies were found
            raise ValueError("No valid email bodies found in the directory.")

        # Extract TF-IDF features for all email bodies
        tfidf_features = self.extract_tfidf_features(email_bodies)
        dense_features = tfidf_features.toarray()

        # Convert the TF-IDF feature matrix to a DataFrame
        df = pd.DataFrame(dense_features, columns=[f'term_{i}' for i in range(dense_features.shape[1])])
        return df

    def process_single_email(self, file_path, vectorizer_path):
        """
        Processes a single email and extracts its TF-IDF features using a pre-fitted vectorizer.
        :param file_path: Path to the email file.
        :param vectorizer_path: Path to the saved vectorizer model.
        :return: DataFrame containing the TF-IDF features for the email.
        """
        # Load the pre-trained vectorizer from the specified path
        self.vectorizer = joblib.load(vectorizer_path)

        # Parse the single email file to extract the body content
        headers, body = self.parse_eml(file_path)
        if not body:
            # Raise an error if the email body is empty
            raise ValueError("The email body is empty. Ensure there is valid email content to process.")

        # Transform the email body into TF-IDF features using the loaded vectorizer
        tfidf_features = self.vectorizer.transform([body])
        dense_features = tfidf_features.toarray()

        # Convert the TF-IDF features into a DataFrame
        df = pd.DataFrame(dense_features, columns=[f'term_{i}' for i in range(dense_features.shape[1])])
        return df

    def save_combined_csv(self, phishing_dir, legitimate_dir, csv_file_path, vectorizer_path):
        """
        Processes phishing and legitimate email directories, extracts features, and saves the combined data to a CSV.
        Uses a pre-fitted TF-IDF vectorizer to ensure consistent feature extraction.
        """
        # Load the pre-trained vectorizer
        self.vectorizer = joblib.load(vectorizer_path)

        # Process phishing emails (label=1) and legitimate emails (label=0)
        phishing_df = self.process_directory(phishing_dir, label=1)
        legitimate_df = self.process_directory(legitimate_dir, label=0)

        # Combine phishing and legitimate data into a single DataFrame
        combined_df = pd.concat([phishing_df, legitimate_df], ignore_index=True)

        # Extract features for all email bodies using the pre-fitted vectorizer
        email_bodies = combined_df['body'].tolist()
        tfidf_features = self.vectorizer.transform(email_bodies)

        # Convert TF-IDF features into a DataFrame
        feature_df = pd.DataFrame(tfidf_features.toarray(),
                                  columns=[f'term_{i}' for i in range(tfidf_features.shape[1])])
        # Add the labels to the DataFrame
        feature_df['label'] = combined_df['label'].values

        # Save the final feature matrix with labels to the specified CSV file
        feature_df.to_csv(csv_file_path, index=False)
        print(f"TF-IDF features and labels saved to {csv_file_path}")

    def extract_features_from_plain_text(self, text):
        """
        Extracts TF-IDF features from a plain text string.
        :param text: Plhain text from which to extract features.
        :return: DataFrame containing TF-IDF features for te text.
        """
        if not text:
            # Raise an error if the text is empty
            raise ValueError("The text is empty. Ensure there is valid content to process.")

        # Transform the plain text into TF-IDF features using the initialized vectorizer
        tfidf_features = self.vectorizer.transform([text])
        dense_features = tfidf_features.toarray()

        # Convert the TF-IDF features into a DataFrame
        df = pd.DataFrame(dense_features, columns=[f'term_{i}' for i in range(dense_features.shape[1])])
        return df

    def process_plain_text_directory(self, directory_path, output_csv_path):
        """
        Processes all .txt files in a directory and extracts TF-IDF features from their contents.
        Saves the TF-IDF features to a CSV file.
        :param directory_path: Path to the directory containing plain text files.
        :param output_csv_path: Path to save the extracted TF-IDF features.
        """
        email_bodies = []

        # Iterate over all .txt files in the directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if file_path.endswith('.txt'):
                try:
                    # Read the content of the text file
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                    if text.strip():
                        # Add non-empty text content to the list
                        email_bodies.append(text.strip())
                    else:
                        # Warn if the file is empty
                        print(f"Warning: Empty text file {filename}. Skipping...")
                except Exception as e:
                    # Handle any errors encountered during file processing
                    print(f"Error processing file {filename}: {e}. Skipping...")

        if not email_bodies:
            # Raise an error if no valid text files were found
            raise ValueError("No valid text files found in the directory.")

        # Fit and transform the vectorizer on the collected email bodies
        tfidf_matrix = self.vectorizer.fit_transform(email_bodies)

        # Convert the TF-IDF feature matrix into a DataFrame with feature names as columns
        feature_names = self.vectorizer.get_feature_names_out()
        df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

        # Save the extracted features to the specified CSV file
        df.to_csv(output_csv_path, index=False)
        print(f"TF-IDF features saved to {output_csv_path}")


if __name__ == "__main__":
    # Path to the pre-fitted vectorizer model (used for consistent feature extraction)
    vectorizer_path = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\code\backend\trained_models\vectorizer.pkl'

    # # Paths for directories used in combined CSV creation
    # phishing_dir = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\data\final_testing\AI_legit_phish_pred\phishing_emails'
    # legitimate_dir = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\data\final_testing\AI_legit_phish_pred\legitimate_emails'
    # output_csv = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\data\final_testing\AI_legit_phish_pred\predictions.csv'
    #
    # # Create the combined CSV for training data
    # extractor.save_combined_csv(phishing_dir, legitimate_dir, output_csv, vectorizer_path)

    # Initialize the feature extractor
    extractor = EmailFeatureExtractor()

    # Path for the single email to be processed
    single_email_path = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\data\final_testing\AI_legit_phish_pred\phishing_emails\email_5.txt'

    # Path to save the TF-IDF features from the single email
    single_email_output_csv = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\data\single_tests\single6.csv'

    # Process the single email and extract features
    single_email_features_df = extractor.process_single_email(single_email_path, vectorizer_path)

    # Save the features to the specified CSV file
    single_email_features_df.to_csv(single_email_output_csv, index=False)
    print(f"TF-IDF features for the single email saved to {single_email_output_csv}")

    ## PLAINTEXT DIRECTORY PROCESSING
    # directory_path = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\data\AI_phishing_emails'
    # output_csv_path = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\output\AI_emails_features.csv'
    # extractor.process_plain_text_directory(directory_path, output_csv_path)
