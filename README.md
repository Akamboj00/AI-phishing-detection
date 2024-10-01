# AI-Phishing-Detection
## Master's Project

**Status:** 🚧 Still in development

### Introduction
Welcome to the AI-Phishing-Detection repository. This project is part of a Master's thesis and focuses on the development of an application that uses advanced machine learning and large language model (LLM) techniques to detect phishing attacks. Our goal is to provide a robust tool to enhance cybersecurity measures against phishing.

### File Structure
The project is organized as follows:

AI-Phishing-Detection/<br>
│<br>
├── Code/ # All code files <br>
&nbsp;&nbsp;&nbsp;&nbsp;── backend/ # Backend scripts and modules<br>
&nbsp;&nbsp;&nbsp;&nbsp;── static/ # graphs and static files(images etc)<br>
&nbsp;&nbsp;&nbsp;&nbsp;── templates/ # html templates<br>
&nbsp;&nbsp;&nbsp;&nbsp;── app/ # Application runner<br>
├── data/ # Datasets <br>
└── output/ # Output files from scripts and models<br>


<h3>How to Run the Application</h3><br>
The easiest way to run the application is:
-	Clone the GitHub repository:
-	Run the application:
o	python app.py
Open a browser and go to localhost:5000 to start using the application.
<br>
The backend scripts (e.g., RandomForest.py, NaiveBayes.py etc.) can also be run individually, but they require a valid dataset to be routed to the script; otherwise, nothing will happen. Running these scripts individually is only necessary for testing batch processing or similar tasks. For real-time detection, running app.py is sufficient.
To route a dataset to the script, you to edit the path connected to the “filepath” variables present in the scripts.
To view the exact datasets used in the project, you can navigate to data/final_testing, and view both the evaluation(AI_legit_phish_pred) and training(AI_legit_phish_train) datasets, respectively.
Note: The ChatGPT model is configured with my personal API key, so this file cannot be fully tested by others unless they set up their own API key.


