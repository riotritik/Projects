# spam_detector/train_model.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

# --- Set up Path and Imports ---
# Assumes feature_extractor.py is in the same directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Import custom feature extractor
from feature_extractor import NonTextualFeatures 

# Define paths relative to the current script
dataset_path = os.path.join(BASE_DIR, 'dataset', 'spam.csv')
model_path = os.path.join(BASE_DIR, 'spam_model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'vectorizer.pkl') # Note: We save the full pipeline here

# --- Data Loading and Preprocessing ---
data = pd.read_csv(dataset_path, encoding='latin-1')

# Keep only the required columns and rename
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Encode labels: 'ham' as 0, 'spam' as 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

X = data['message']
y = data['label']

# Split data into training and testing sets (stratify ensures equal label proportions)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# --- Define the Hybrid Pipeline Components ---

# 1. Text-based Pipeline (TF-IDF Vectorization)
text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),  # Use 1-grams and 2-grams (e.g., 'click here')
        max_features=5000     # Keep top 5000 features
    ))
])

# 2. Non-Textual Feature Pipeline
non_text_pipeline = Pipeline([
    ('non_text_features', NonTextualFeatures()),
    ('scaler', StandardScaler())  # Scale numerical features (important for LogReg)
])

# 3. Combine Features using FeatureUnion
feature_union = FeatureUnion([
    ('text_features', text_pipeline),
    ('extra_features', non_text_pipeline)
])

# 4. Final Classifier Pipeline
model_pipeline = Pipeline([
    ('features', feature_union),
    ('clf', LogisticRegression(
        solver='liblinear',
        class_weight='balanced',  # CRUCIAL for optimizing RECALL on imbalanced data
        random_state=42
    ))
])

# --- Training and Evaluation ---
print("Starting hybrid model training...")
model_pipeline.fit(X_train, y_train)
print("Training complete.")

# Predict and Evaluate
predictions = model_pipeline.predict(X_test)

# Calculate key metrics
accuracy = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f'\nHybrid Model Accuracy: {accuracy * 100:.2f}%')
print(f'Hybrid Model RECALL (Catching Spam): {recall * 100:.2f}%')
print(f'Hybrid Model F1-Score: {f1:.4f}')

# --- Save Model Pipeline ---
# Save the entire pipeline, including the feature extractors, for prediction.
joblib.dump(model_pipeline, model_path)
print("\nModel pipeline saved successfully to spam_model.pkl!")
# Note: The vectorizer_path variable is unused but kept for consistency.