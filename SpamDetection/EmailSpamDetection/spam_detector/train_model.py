import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'spam.csv')
data = pd.read_csv(dataset_path, encoding='latin-1')

# Preprocess data
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Custom Spam Rule Function
def custom_spam_rule(text):
    spam_keywords = ['join my group', 'earn money', 'click here', 'make money', 'free gift', 'limited offer']
    for keyword in spam_keywords:
        if keyword in text.lower():
            return 1  # Flag as Spam
    return 0  # Flag as Ham

# Apply custom rule to label data
data['custom_spam_label'] = data['message'].apply(custom_spam_rule)

# Prepare features and labels
X = data['message']
y = data['label']

# Vectorize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save model and vectorizer
model_path = os.path.join(os.path.dirname(__file__), 'spam_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print("Model and vectorizer saved successfully!")
