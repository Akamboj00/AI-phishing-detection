import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_fscore_support


# Load and preprocess the data (assuming the same for all models)
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop_duplicates()
    X = df.drop('label', axis=1).values
    y = df['label'].values
    return X, y


# Filepath to dataset
filepath = r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\data\final_testing\AI_legit_phish_train\training.csv'  # Update this path

# Load dataset
X, y = load_and_preprocess_data(filepath)
y = label_binarize(y, classes=[0, 1])  # Adjust if necessary (0: Legitimate, 1: Phishing)

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Load trained models
rf_model = joblib.load('trained_models/random_forest_model.pkl')
nb_model = joblib.load('trained_models/naive_bayes_model.pkl')
svm_model = joblib.load('trained_models/svm_model.pkl')
xgb_model = joblib.load('trained_models/xgboost_model.pkl')


# Function to evaluate and plot metrics
def evaluate_model(model, model_name):
    # Predict
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # For ROC Curve

    # Accuracy, Precision, Recall, F1-Score
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    print(
        f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Label for ROC curve
    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

    return accuracy, precision, recall, f1, roc_auc


# Plot all ROC curves on the same figure
plt.figure(figsize=(8, 6))

for model, model_name in [(rf_model, "Random Forest"), (nb_model, "Naive Bayes"), (svm_model, "SVM"),
                          (xgb_model, "XGBoost")]:
    evaluate_model(model, model_name)

# Add the diagonal reference line for random performance
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')  # Diagonal for reference

# Set the axis limits
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

# Label the axes
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# Title for the plot
plt.title('ROC Curve Comparison Across Models')

# Add legend to the bottom right of the plot
plt.legend(loc="lower right")

# Show the plot
plt.show()

# Feature importance for Random Forest
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plotting Feature Importance for Random Forest
plt.figure(figsize=(8, 5))
plt.barh([f'Feature {i}' for i in indices[:10]], importances[indices[:10]], color='lightgreen')
plt.xlabel('Importance')
plt.title('Random Forest - Top 10 Feature Importance')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest feature on top
plt.show()

# Summary table for Precision, Recall, F1-Score, and AUC
summary_metrics = []
for model, model_name in [(rf_model, "Random Forest"), (nb_model, "Naive Bayes"), (svm_model, "SVM"),
                          (xgb_model, "XGBoost")]:
    acc, prec, rec, f1, roc_auc = evaluate_model(model, model_name)
    summary_metrics.append(
        {'Model': model_name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1, 'AUC': roc_auc})

# Create DataFrame for summary
df_summary = pd.DataFrame(summary_metrics)
print(df_summary)

# Grouped bar chart for Precision, Recall, and F1-Score comparison
df_metrics = df_summary.melt(id_vars="Model", value_vars=["Precision", "Recall", "F1-Score"], var_name="Metric",
                             value_name="Score")

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Score', hue='Metric', data=df_metrics, palette='Blues_d')
plt.ylim(0.0, 1.05)
plt.title('Precision, Recall, and F1-Score Comparison')
plt.ylabel('Score')
plt.xlabel('Model')
plt.legend(loc='lower right')
plt.show()


# Function to plot ROC curve for each model separately
def plot_individual_roc_curves(models, X_test, y_test):
    for model, model_name in models:
        plt.figure(figsize=(8, 6))  # Create a new figure for each model

        # Predict and calculate the ROC curve and AUC
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve
        plt.plot(fpr, tpr, lw=2, color='blue', label=f'{model_name} (AUC = {roc_auc:.2f})')

        # Plot a diagonal reference line (random guessing)
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=2)

        # Set axis limits
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        # Labels and title
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model_name}')

        # Show legend
        plt.legend(loc="lower right")

        # Show the plot
        plt.show()

# Example usage with trained models
models = [(rf_model, "Random Forest"),
          (nb_model, "Naive Bayes"),
          (svm_model, "SVM"),
          (xgb_model, "XGBoost")]

# Call the function to plot individual ROC curves for each model
plot_individual_roc_curves(models, X_test, y_test)
