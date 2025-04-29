import pandas as pd
import numpy as np
import re
import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
print("Dataset size:", df.shape)
print(df['label'].value_counts(), "\n")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\s+', ' ', text).strip()

df['clean_msg'] = df['message'].apply(clean_text)
print(df[['message', 'clean_msg']].head(), "\n")

y = df['label'].map({'ham': 0, 'spam': 1}).values
X = df['clean_msg'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=5)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

X_train_tensor = torch.tensor(X_train_tfidf, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_tfidf, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class SpamClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
        super(SpamClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SpamClassifier(input_dim=X_train_tfidf.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 13
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * inputs.size(0)

    avg_loss = epoch_loss / len(train_dataset)
    print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")

def evaluate(model, X_tensor, y_true, dataset_name='Set'):
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor.to(device))
        _, preds = torch.max(outputs, 1)
        y_pred = preds.cpu().numpy()
        print(f"\n{dataset_name} Evaluation:")
        print(classification_report(y_true, y_pred, target_names=['ham', 'spam']))
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred):.4f}")
        print(f"Recall: {recall_score(y_true, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    return y_pred

_ = evaluate(model, X_train_tensor, y_train, dataset_name='Training Set')
_ = evaluate(model, X_test_tensor, y_test, dataset_name='Test Set')

# Save Model and Vectorizer for script
torch.save(model.state_dict(), 'spam_classifier.pt')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Model and vectorizer saved.")
