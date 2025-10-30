import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import pickle

# Set the paths for model and vectorizer
model_path = os.path.join(os.path.dirname(__file__), 'spam_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')
2
# Load dataset
dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'spam.csv')
data = pd.read_csv(dataset_path, encoding='latin-1')

# Preprocess data
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Tokenization using torchtext
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

# Build vocabulary
vocab = build_vocab_from_iterator(yield_tokens(data['message']), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Tokenize and convert to indices
def text_pipeline(text):
    return [vocab[token] for token in tokenizer(text)]

# Prepare dataset
X = data['message'].apply(text_pipeline)

# Use pad_sequence to pad the tokenized sequences
X_padded = [torch.tensor(seq) for seq in X]
X_padded = pad_sequence(X_padded, batch_first=True, padding_value=0)

y = data['label'].values

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.tolist(), dtype=torch.long)
X_test_tensor = torch.tensor(X_test.tolist(), dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for batching
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Define LSTM Model in PyTorch
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.fc(hn[-1])
        return self.sigmoid(out)

# Instantiate the model
model = LSTMModel(vocab_size=len(vocab), embed_dim=128, hidden_dim=100, output_dim=1)

# Loss and Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Train the model
train_model(model, train_loader, criterion, optimizer, epochs=5)

# Save the model and vectorizer
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

with open(vectorizer_path, 'wb') as f:
    pickle.dump(vocab, f)

# Evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (outputs.squeeze() > 0.5).long()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total * 100
    return accuracy

# Test the model
accuracy = evaluate_model(model, test_loader)
print(f'LSTM Model Accuracy: {accuracy:.2f}%')
