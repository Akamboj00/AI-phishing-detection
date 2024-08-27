import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from scipy.sparse import csr_matrix
from sklearn.impute import SimpleImputer


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    features = df.columns[:-1]
    X = df[features].values
    y = df['label']

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    X = csr_matrix(X)

    return train_test_split(X, y, test_size=0.3, random_state=42)


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.2f}")

    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    print("Classification Report:\n", df_report)

    metrics = ['precision', 'recall', 'f1-score']

    plt.figure(figsize=(12, 6))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        sns.barplot(x=df_report.index[:-3], y=metric, data=df_report[:-3])
        plt.title(f'{metric.capitalize()}')
        plt.ylim(0, 1)
        plt.xlabel('Class')
        plt.ylabel(metric.capitalize())
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    filepath = '../output/combined_email_features.csv'
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    train_and_evaluate_model(xgb_model, X_train, X_test, y_train, y_test)
