import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import string
import langdetect


# 1 Download NLTK data

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


# 2 Dataset

non_toxic_comments = [
    "Great job on the project!", "Thanks for the help.", "This is informative.",
    "I appreciate your input.", "Well done everyone.", "Interesting perspective.",
    "Helpful advice here.", "Positive feedback.", "Good discussion.", "Keep it up!"
]

toxic_comments = [
    "This is garbage.", "You're an idiot.", "Shut up already.", "Hate this nonsense.",
    "Total waste of time.", "Annoying and stupid.", "This sucks badly", "Go away loser.",
    "Useless crap.", "Pathetic attempt.", "Disgusting behavior."
]

texts = non_toxic_comments + toxic_comments
labels = [0]*len(non_toxic_comments) + [1]*len(toxic_comments)  # 0 = Non-toxic, 1 = Toxic

df = pd.DataFrame({"text": texts, "label": labels})

# 3 Preprocessing

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

df["processed_text"] = df["text"].apply(preprocess)

# 4 Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    df["processed_text"], df["label"], test_size=0.2, random_state=42
)


# 5 Feature Extraction (TF-IDF)

vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6 Model Training (Logistic Regression)

model = LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")
model.fit(X_train_vec, y_train)


# 7 Model Evaluation

y_pred = model.predict(X_test_vec)
y_prob = model.predict_proba(X_test_vec)

print("📊 Model Evaluation Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.2f}")
print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred, zero_division=0):.2f}\n")
print("Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Non-Toxic", "Toxic"], zero_division=0))


# 8 Prediction Function

def predict_comment(comment, threshold=0.55):
    try:
        if langdetect.detect(comment) != "en":
            print("⚠️ Warning: Non-English text detected!")
    except:
        pass

    processed = preprocess(comment)
    vec = vectorizer.transform([processed])
    prob = model.predict_proba(vec)[0]
    pred = 1 if prob[1] >= threshold else 0

    label = "Toxic" if pred == 1 else "Non-Toxic"
    print("---------------------------------------------------")
    print(f"🗨️ Comment: {comment}")
    print(f"Prediction: {label} ({pred})")
    print(f"Probability [Non-Toxic, Toxic]: {prob}")


# 9 Generate Sample Output File

sample_comments = [
    "Thanks for your help!",
    "This is garbage.",
    "I really enjoyed this.",
    "This sucks badly."
]

with open("sample_output.txt", "w", encoding="utf-8") as f:
    for comment in sample_comments:
        try:
            if langdetect.detect(comment) != "en":
                f.write("⚠️ Warning: Non-English text detected!\n")
        except:
            pass

        processed = preprocess(comment)
        vec = vectorizer.transform([processed])
        prob = model.predict_proba(vec)[0]
        pred = 1 if prob[1] >= 0.55 else 0
        label = "Toxic" if pred == 1 else "Non-Toxic"

        f.write("---------------------------------------------------\n")
        f.write(f"🗨️ Comment: {comment}\n")
        f.write(f"Prediction: {label} ({pred})\n")
        f.write(f"Probability [Non-Toxic, Toxic]: {prob}\n")

print("\n✅ Sample output saved to sample_output.txt")


# 10 Interactive Loop

while True:
    user_input = input("Enter a comment to check toxicity (or 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    predict_comment(user_input)