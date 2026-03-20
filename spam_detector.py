# Spam Email Detection using NLP

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create Dataset
data = {
    'Message': [
        "Win money now", "Free offer just for you", "Call me now",
        "Let's meet tomorrow", "Project meeting at 10am",
        "Congratulations you won a prize", "Hello how are you",
        "Get free coupons now", "Important update", "Limited time offer"
    ],
    'Label': [1,1,0,0,0,1,0,1,0,1]  # 1 = Spam, 0 = Not Spam
}

df = pd.DataFrame(data)

# Step 2: Split Data
X = df['Message']
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 3: Text Vectorization
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 4: Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 5: Prediction
y_pred = model.predict(X_test_vec)

# Step 6: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# Step 7: User Input
msg = input("Enter a message: ")
msg_vec = vectorizer.transform([msg])

prediction = model.predict(msg_vec)

if prediction[0] == 1:
    print("Spam Message ❌")
else:
    print("Not Spam Message ✅")
