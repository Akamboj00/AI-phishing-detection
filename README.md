# AI-Phishing-Detection
## Master's Project

### Introduction
Welcome to the AI-Phishing-Detection repository. This project is part of a Master's thesis and focuses on the development of an application that uses advanced machine learning and large language model (LLM) techniques to detect phishing attacks. Our goal is to provide a robust tool to enhance cybersecurity measures against phishing.

### File Structure
The project is organized as follows:

AI-Phishing-Detection/<br>
•	Feature_extraction.py – Handles data preprocessing and feature extraction.<br>
•	RandomForest.py – Implements the Random Forest model.<br>
•	NaïveBayes.py – Implements the Naive Bayes model.<br>
•	SVM.py – Implements the Support Vector Machine model.<br>
•	XGBoost.py – Implements the XGBoost model.<br>
•	Ensemble.py – Combines individual models into an ensemble.<br>
•	ChatGPT.py – Integrates the ChatGPT API for phishing detection.<br>
•	app.py – Main application logic and routing for the web tool.<br>
•	Detect.html – Frontend interface for detecting phishing emails.<br>
•	Metrics.html – Displays model metrics.<br>
•	Results.html – Presents detection results.<br>
•	Train.html – Interface for model training.<br>
•	MetricsForReport.py – Script for generating model performance metrics for the report.<br>



<h3>How to Run the Application</h3><br>
The easiest way to run the application is:<br>
-	Clone the GitHub repository:<br>
-	Run the app.py script <br> 
- Open a browser and go to localhost:5000 to start using the application.<br>
<br><br>
The backend scripts (e.g., RandomForest.py, NaiveBayes.py etc.) can also be run individually, but they require a valid dataset to be routed to the script; otherwise, nothing will happen. Running these scripts individually is only necessary for testing batch processing or similar tasks. For real-time detection, running app.py is sufficient.<br><br>
To route a dataset to the script, you to edit the path connected to the “filepath” variables present in the scripts.<br><br>
To view the exact datasets used in the project, you can navigate to data/final_testing, and view both the evaluation(AI_legit_phish_pred) and training(AI_legit_phish_train) datasets, respectively.<br><br>
<br>Note: The ChatGPT model is configured with my personal API key, so this file cannot be fully tested by others unless they set up their own API key.<br>


