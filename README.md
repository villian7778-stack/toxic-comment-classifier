# Toxic Comment Classifier

A Python-based NLP pipeline to classify short text comments as **toxic** or **non-toxic**. Uses NLTK for text preprocessing and scikit-learn for model development. Suitable for monitoring social media comments or customer feedback.

---

## 📌 Project Overview

- **Objective:** Classify short text comments as `Toxic` or `Non-Toxic`.
- **Use Case:** Monitoring social media, customer feedback, or any text input for toxic content.
- **Tech Stack:**
  - Python
  - NLTK (Text preprocessing: tokenization, lemmatization, stopwords removal)
  - scikit-learn (TF-IDF Vectorizer, Logistic Regression)
  - pandas (Data handling)
  - langdetect (Detect non-English comments)

- **Dataset:**  
  20 sample comments (10 toxic, 10 non-toxic) used for demonstration.  
  Labels: `0` → Non-toxic, `1` → Toxic.

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/villian7778-stack/toxic-comment-classifier.git
cd toxic-comment-classifier


#2. Create and activate a virtual environment (optional but recommended)
# Using conda
conda create -n toxic-nlp python=3.10 -y
conda activate toxic-nlp

# OR using venv
python -m venv toxic-nlp
source toxic-nlp/Scripts/activate   # Windows
source toxic-nlp/bin/activate       # Linux/Mac

#3. Install dependencies
pip install -r requirements.txt


#🚀4 Running the Project
1. Run the classifier script
python toxic_classifier.py




#2. Sample interaction
Enter a comment to check toxicity (or 'exit' to quit): This is garbage.
---------------------------------------------------
🗨️ Comment: This is garbage.
Prediction: Toxic (1)
Probability [Non-Toxic, Toxic]: [0.40228726 0.59771274]

3. Sample Output File

The script generates a file sample_output.txt with predictions for sample comments:

---------------------------------------------------
🗨️ Comment: Thanks for your help!
Prediction: Non-Toxic (0)
Probability [Non-Toxic, Toxic]: [0.8, 0.2]
---------------------------------------------------
🗨️ Comment: This is garbage.
Prediction: Toxic (1)
Probability [Non-Toxic, Toxic]: [0.4, 0.6]



#🛠️ Code Structure
toxic-comment-classifier/
│
├─ toxic_classifier.py       # Main Python script
├─ requirements.txt          # Python dependencies
├─ README.md                 # Project documentation
├─ sample_output.txt         # Sample predictions
├─ toxic_model/              # (ignored) Saved models
└─ .gitignore                # Files/folders to ignore


#📝 Dependencies

All required libraries are listed in requirements.txt:

nltk
scikit-learn
pandas
numpy
langdetect


#Install them using:

pip install -r requirements.txt

#⚠️ Notes

The dataset is small; for real-world usage, consider using a larger, labeled dataset.

Non-English comments will display a warning.

Predictions include probability scores for transparency.

#Usage

Run toxic_classifier.py.

Enter any comment.

See if it is Toxic or Non-Toxic with probability.

Check sample_output.txt for sample results.

#Future Enhancements

---Add more training data for better accuracy.

Support multiple languages.

Integrate with social media APIs for real-time monitoring.

Save and load models for faster predictions.