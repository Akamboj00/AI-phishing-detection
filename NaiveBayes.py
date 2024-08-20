import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB

# Load the dataset
df = pd.read_csv('../../Data/dataset_small.csv')

# Plot the distribution of phishing vs legitimate
sns.countplot(x='phishing', data=df)
plt.title('Distribution of Phishing vs Legitimate')
plt.xlabel('Legitimate (0) vs Phishing (1)')
plt.ylabel('Count')
plt.show()

# Define features and target
features = df.columns[:-1]
x = df[features]
y = df['phishing']

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Train the Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(x_train, y_train)

# Make predictions
y_pred = nb_model.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate classification report
report = classification_report(y_test, y_pred, output_dict=True)

# Convert the classification report to a DataFrame
df_report = pd.DataFrame(report).transpose()

# Display the classification report as a well-formatted table
print("Classification Report:\n", df_report)

print(f"\nAccuracy: {accuracy:.2f}")

# Plot the metrics
metrics = ['precision', 'recall', 'f1-score']

plt.figure(figsize=(12, 6))
for i, metric in enumerate(metrics, 1):
    plt.subplot(1, 3, i)
    sns.barplot(x=df_report.index[:-3], y=metric, data=df_report[:-3])
    plt.title(f'{metric.capitalize()}')
    plt.ylim(0, 1)
    plt.xlabel('Legitimate (0) vs Phishing (1)')
    plt.ylabel(metric.capitalize())
plt.tight_layout()
plt.show()
