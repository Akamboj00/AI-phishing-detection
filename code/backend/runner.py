import subprocess

# Paths to each of your model training scripts
script_paths = [
    r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\code\backend\NaiveBayes.py',
    r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\code\backend\SVM.py',
    r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\code\backend\XGBoost.py',
    r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\code\backend\RandomForest.py',
    r'C:\Users\Abhi\OneDrive - City, University of London\Cyber Security MSc\Main\Project\03 Software\Code\AI-phishing-detection\code\backend\DNN.py'
]

# Function to run a script and display its output in the console
def run_script(script_path):
    try:
        print(f"\nRunning {script_path}...\n")
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        print(result.stdout)
        print(f"\nFinished running {script_path}\n")
    except Exception as e:
        print(f"Error running {script_path}: {e}")

# Main runner
if __name__ == "__main__":
    for script in script_paths:
        run_script(script)
