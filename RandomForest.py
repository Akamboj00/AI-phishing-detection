import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('../../Data/dataset_small.csv')

sns.countplot(x='phishing', data=df)
plt.title('Distribution of Phishing vs Legitimate')
plt.xlabel('Legitimate (0) vs Phishing (1) ')
plt.ylabel('Count')
plt.show()

#features = ['qty_dot_url', 'qty_hyphen_url', 'url_google_index']
features = df.columns[:-1]
x = df[features]
y = df['phishing']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

report = classification_report(y_test, y_pred, output_dict=True)
print(report)

df_report = pd.DataFrame(report).transpose()

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



