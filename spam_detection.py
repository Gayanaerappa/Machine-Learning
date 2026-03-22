# 📧 Spam Email Detection using NLP + ML

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset (you can replace with CSV later)
data = {
    "message": [
        "Win money now!!!",
        "Hi, how are you?",
        "Claim your free prize",
        "Let's meet tomorrow",
        "Congratulations, you won a lottery!",
        "Call me when you are free",
        "Exclusive offer just for you",
        "Are we still meeting today?"
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Spam, 0 = Ham
}

df = pd.DataFrame(data)

# Features and target
X = df["message"]
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression()

# Train
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Test custom message
sample = ["Congratulations! You won a free ticket"]
sample_vec = vectorizer.transform(sample)
prediction = model.predict(sample_vec)

print("\n📩 Message:", sample[0])
print("Prediction:", "Spam" if prediction[0] == 1 else "Not Spam")
