# spam_detector/feature_extractor.py

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# Custom transformer to extract non-textual features
class NonTextualFeatures(BaseEstimator, TransformerMixin):
    """
    Extracts numerical features from text messages to enhance spam detection.
    Features: length, number of digits, ratio of uppercase characters, and punctuation count.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        punctuation_marks = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        
        for text in X:
            text = str(text) # Ensure input is treated as a string
            length = len(text)
            
            # Avoid division by zero for ratio calculations
            safe_length = length + 1e-6 
            
            # 1. Number of digits
            num_digits = sum(c.isdigit() for c in text)
            
            # 2. Percentage of uppercase letters
            upper_ratio = sum(c.isupper() for c in text) / safe_length
            
            # 3. Number of punctuation marks
            num_punc = sum(c in punctuation_marks for c in text)
            
            # 4. Total length of message
            
            features.append([length, num_digits, upper_ratio, num_punc])
            
        return np.array(features)