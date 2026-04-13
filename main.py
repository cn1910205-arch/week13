import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data/spam.csv")

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Test detection
test_messages = [
    "Win a free iPhone",
    "Let's go to school",
    "Claim your prize now",
    "See you later"
]

X_test = vectorizer.transform(test_messages)
predictions = model.predict(X_test)

# Print results
for msg, pred in zip(test_messages, predictions):
    print(f"{msg} --> {'Spam' if pred == 1 else 'Not Spam'}")