# Step 1: Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, confusion_matrix

# Step 2: Load dataset
# Dataset should have columns: 'text' and 'label'
data = pd.read_csv("spam.csv")

# Step 3: Features and target
X = data['text']
y = data['label']   # spam or ham

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Convert text to numbers (TF-IDF)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Model
model = MultinomialNB()

# Step 7: Train
model.fit(X_train_vec, y_train)

# Step 8: Predict
y_pred = model.predict(X_test_vec)

# Step 9: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 10: Test with new message
sample = ["Congratulations! You won a lottery"]
sample_vec = vectorizer.transform(sample)

print("Prediction:", model.predict(sample_vec)[0])
